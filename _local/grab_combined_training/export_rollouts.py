# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export deterministic validation rollouts for honest pilot-result videos.

This does not train or evaluate the test split. The original vector wrapper
automatically resets failed simulations; those reset states are never exported
as motion. Instead, the last valid state is held with a persistent failure flag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from train_policy import (
    EVALUATION_PROTOCOL,
    OBSERVATION_SIZE,
    ObservationMoments,
    TrackingBatch,
    TrackingPolicy,
    collect,
    make_windows,
    sha256,
    summarize,
    write_json,
)


class RecordingBatch(TrackingBatch):
    """Capture physical states [m, rad] while preserving the learner's rollout."""

    def reset(self, sequences: list[dict], seed: int, dt: float) -> None:
        super().reset(sequences, seed, dt)
        self.raw.get_global_state(self.state)
        self.states = [self.state.copy()]
        self.failure_flags = [np.zeros(self.num_envs, dtype=np.bool_)]
        self.reset_seed = seed

    def step(self, actions: np.ndarray) -> None:
        super().step(actions)
        failed = self.failure_flags[-1] | self.done
        current = self.state.copy()
        current[failed] = self.states[-1][failed]
        self.states.append(current)
        self.failure_flags.append(failed.copy())


def read_clip(prepared: Path, entry: dict) -> dict:
    """Read a single verified reference; arrays use metres and radians."""
    path = Path(entry["path"])
    if not path.is_absolute():
        path = prepared / path
    if entry.get("sha256") and sha256(path) != entry["sha256"]:
        raise ValueError(f"Reference hash changed: {path}")
    with np.load(path) as archive:
        return {key: archive[key].copy() for key in archive.files}


def capture(
    checkpoint: dict,
    windows: list[dict],
    references: dict,
    manifest: dict,
    config: dict,
    device: torch.device,
) -> tuple[list[dict], dict]:
    """Replay validation in the same order and seeds used for model selection."""
    policy = TrackingPolicy(OBSERVATION_SIZE).to(device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    moments = ObservationMoments(OBSERVATION_SIZE)
    moments.load_state_dict(checkpoint["moments"])
    moments_before = moments.state_dict()
    rng = np.random.default_rng(config["seed"] + 1_000_000)
    batch = RecordingBatch(manifest, config, False)
    if batch.binary_sha256 != checkpoint["environment_sha256"]:
        batch.close()
        raise ValueError("Simulator binary differs from the saved training checkpoint")
    results, metric_rows = [], []
    try:
        for start in range(0, len(windows), batch.num_envs):
            selected = windows[start : start + batch.num_envs]
            _, rows = collect(batch, selected, references, policy, moments, rng, device, config, False)
            states = np.stack(batch.states)
            failed = np.stack(batch.failure_flags)
            for lane, (window, metrics) in enumerate(zip(selected, rows)):
                length = window["end"] - window["start"]
                indices = np.minimum(np.arange(length), len(states) - 1)
                result = {
                    "state": states[indices, lane].copy(),
                    "failed": failed[indices, lane].copy(),
                    "metrics": metrics,
                    "reset_seed": batch.reset_seed,
                }
                if not np.isfinite(result["state"]).all():
                    raise RuntimeError("Refusing to export nonfinite state")
                failure_indices = np.flatnonzero(result["failed"])
                if len(failure_indices):
                    first = int(failure_indices[0])
                    if first == 0 or not np.array_equal(
                        result["state"][first:],
                        np.broadcast_to(result["state"][first - 1], result["state"][first:].shape),
                    ):
                        raise RuntimeError("Failed world was not held at its last valid state")
                results.append(result)
            metric_rows.extend(rows)
            print(f"Checkpoint epoch {checkpoint['epoch']}: {len(results)}/{len(windows)} windows", flush=True)
    finally:
        batch.close()
    if moments.count != moments_before["count"] or not np.array_equal(moments.mean, moments_before["mean"]):
        raise RuntimeError("Validation modified normalization statistics")
    return results, {"summary": summarize(metric_rows), "windows": metric_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--original_prepared", type=Path, default=Path(__file__).with_name("prepared"))
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = (args.output_dir or run / "video_export" / "rollouts").resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((run / "report.json").read_text())
    prepared = Path(report["prepared"])
    manifest_path = prepared / "manifest.json"
    if sha256(manifest_path) != report["manifest_sha256"]:
        raise ValueError("Prepared data manifest differs from the pilot report")
    manifest = json.loads(manifest_path.read_text())
    original_manifest_path = args.original_prepared / "manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text())
    original_entries = {item["clip_id"]: item for item in original_manifest["clips"]}
    references, originals, episodes, clip_metadata = {}, {}, [], {}
    for entry in manifest["clips"]:
        if entry["split"] != "validation":
            continue
        clip_id = entry["clip_id"]
        references[clip_id] = read_clip(prepared, entry)
        originals[clip_id] = read_clip(args.original_prepared, original_entries[clip_id])
        if not np.array_equal(references[clip_id]["source_frame_indices"], originals[clip_id]["source_frame_indices"]):
            raise ValueError(f"Original and refined frame indices differ: {clip_id}")
        if not np.allclose(references[clip_id]["object_pose"], originals[clip_id]["object_pose"], atol=1e-6):
            raise ValueError(f"Original and refined mug motion differs: {clip_id}")
        clip_metadata[clip_id] = entry
        for index, episode in enumerate(entry["episodes"]):
            if episode.get("supported", True) and episode["end"] - episode["start"] >= 2:
                episodes.append(dict(episode, clip_id=clip_id, episode_index=index))
    config = report["config"]
    windows = make_windows(episodes, config["window_frames"], references)
    if len(windows) != report["split_windows"]["validation"]:
        raise ValueError("Validation window count changed")
    for key in ("hand_urdf", "object_urdf"):
        if manifest.get(f"{key}_sha256") and sha256(Path(manifest[key])) != manifest[f"{key}_sha256"]:
            raise ValueError(f"Physical asset changed: {key}")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run with GPU permission or explicitly use --device cpu")
    checkpoint_paths = {"selected": run / "best.pt", "latest": run / "last.pt"}
    captured, summaries, provenance = {}, {}, {}
    for name, path in checkpoint_paths.items():
        checkpoint = torch.load(path, map_location=device)
        for key in ("manifest_sha256", "environment_sha256", "evaluation_protocol"):
            if checkpoint[key] != report[key]:
                raise ValueError(f"Checkpoint provenance mismatch: {name}/{key}")
        if checkpoint["evaluation_protocol"] != EVALUATION_PROTOCOL:
            raise ValueError("Evaluation code differs from the checkpoint's metric protocol")
        if {k: v for k, v in checkpoint["config"].items() if k != "epochs"} != {
            k: v for k, v in config.items() if k != "epochs"
        }:
            raise ValueError(f"Checkpoint config differs beyond epoch target: {name}")
        captured[name], summaries[name] = capture(checkpoint, windows, references, manifest, config, device)
        provenance[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "epoch": checkpoint["epoch"],
            "validation_selected": name == "selected",
            "policy_kind": "zero-residual reference-PD baseline"
            if checkpoint["epoch"] == 0
            else "trained residual PPO",
        }
    exported = []
    for index, window in enumerate(windows):
        clip_id = window["clip_id"]
        reference, original = references[clip_id], originals[clip_id]
        start, end = window["start"], window["end"]
        frames = slice(start, end)
        arrays = {
            "frame_indices": np.arange(start, end, dtype=np.int64),
            "source_frames": reference["source_frame_indices"][frames],
            "table_dimensions": reference["table_dimensions"],
            "table_pose": reference["table_pose"],
            "expected_contact": reference["contact"][frames],
            "world_to_training_rotation": reference["world_to_training_rotation"],
            "world_to_training_translation": reference["world_to_training_translation"],
        }
        for key in ("hand_qpos", "object_pose", "joints"):
            arrays[f"reference_{key}"] = reference[key][frames]
            arrays[f"original_source_{key}"] = original[key][frames]
        arrays["original_recorded_joints"] = original["recorded_joints"][frames]
        item = dict(window, window_index=index, fps=manifest["fps"], frames=end - start)
        for name in captured:
            result = captured[name][index]
            arrays[f"{name}_state"] = result["state"]
            arrays[f"{name}_failed"] = result["failed"]
            failure_indices = np.flatnonzero(result["failed"])
            item[f"{name}_first_failed_frame"] = int(failure_indices[0]) if len(failure_indices) else None
            item[f"{name}_metrics"] = result["metrics"]
            item[f"{name}_reset_seed"] = result["reset_seed"]
        path = output / f"window_{index:02d}_{clip_id}_f{start:04d}-{end:04d}.npz"
        np.savez_compressed(path, **arrays)
        item.update(npz_path=str(path), sha256=sha256(path), source_frame_indices=arrays["source_frames"].tolist())
        exported.append(item)
    receipt = {
        "schema_version": 1,
        "split": "validation",
        "test_split_used": False,
        "trained_during_export": False,
        "run_report": str(run / "report.json"),
        "run_report_sha256": sha256(run / "report.json"),
        "prepared": str(prepared),
        "manifest_sha256": report["manifest_sha256"],
        "original_prepared": str(args.original_prepared.resolve()),
        "original_manifest_sha256": sha256(original_manifest_path),
        "environment_sha256": report["environment_sha256"],
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "exporter_sha256": sha256(Path(__file__)),
        "device": str(device),
        "checkpoint_provenance": provenance,
        "clip_count": len({item["clip_id"] for item in windows}),
        "window_count": len(windows),
        "frame_count_with_resets": sum(item["frames"] for item in exported),
        "planned_transitions": sum(item["frames"] - 1 for item in exported),
        "frames_include_initial_reset": True,
        "window_boundaries": (
            "Each window starts a separately reset physical world; do not imply continuous rollout across windows."
        ),
        "failure_display": (
            "At the first terminal failure the wrapper's reset state is discarded; "
            "the last valid state is held through the rest of the window and failed=true."
        ),
        "state_layout": {"hand_qpos": [0, 51], "object_pose_xyz_wxyz": [102, 109], "joints_21xyz": [115, 178]},
        "units": {"positions": "m", "angles": "rad", "fps": manifest["fps"]},
        "summaries": summaries,
        "windows": exported,
    }
    write_json(output / "manifest.json", receipt)
    print(
        json.dumps(
            {
                "manifest": str(output / "manifest.json"),
                "windows": len(windows),
                "summaries": {k: v["summary"] for k, v in summaries.items()},
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
