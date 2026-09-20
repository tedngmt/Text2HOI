# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure warmed Text2HOI refiner inference on existing GraspXL source motions.

Run with /home/nmt/miniconda3/envs/text2hoi/bin/python. All timed inputs and
outputs stay on the GPU; checkpoint loading, rendering, and export are excluded.
This benchmarks source-motion cleanup, not text-conditioned motion generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir


def digest(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def measure(function, frames: int, warmups: int, repeats: int) -> dict:
    """Measure synchronized wall-clock latency [s] with GPU-resident tensors."""
    for _ in range(warmups):
        result = function()
        torch.cuda.synchronize()
        del result
    durations = []
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        del result
    values = np.asarray(durations)
    return {
        "unique_output_frames": frames,
        "warmups": warmups,
        "repeats": repeats,
        "seconds_mean": float(values.mean()),
        "seconds_p50": float(np.percentile(values, 50)),
        "seconds_p95": float(np.percentile(values, 95)),
        "seconds_min": float(values.min()),
        "seconds_max": float(values.max()),
        "fps_total_frames_over_total_seconds": float(frames * repeats / values.sum()),
        "fps_at_p50_latency": float(frames / np.percentile(values, 50)),
        "fps_at_p95_latency": float(frames / np.percentile(values, 95)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated() - baseline_allocated,
        "sample_seconds": durations,
    }


def main() -> None:  # noqa: C901 - standalone benchmark shares loaded model and assets
    script_path = Path(__file__).resolve()
    local = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=local.parent.parent / "Text2HOI")
    parser.add_argument("--prepared", type=Path, default=local / "graspxl_mug_refinement" / "prepared")
    parser.add_argument(
        "--checkpoint", type=Path, default=local / "graspxl_mug_refinement" / "runs" / "all18_v1" / "checkpoint.pth"
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("refiner_results.json"))
    parser.add_argument("--clip_id", default="mano_dataset_1__1")
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.window < 8 or args.window > 150 or args.warmups < 1 or args.repeats < 1:
        parser.error("window must be 8..150; warmups and repeats must be positive")
    args.repo = args.repo.resolve()
    args.prepared = args.prepared.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    os.environ.setdefault("WANDB_MODE", "disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("Run with Text2HOI's original Conda environment and GPU access")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    from lib.models.mano import build_mano_aa
    from lib.utils.model_utils import build_refiner
    from lib.utils.proc import proc_refiner_input
    from lib.utils.proc_output import get_hand_verts

    checkpoint_hash = digest(args.checkpoint)
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    entry = next(item for item in manifest["clips"] if item["clip_id"] == args.clip_id)
    clip_path = args.prepared / "clips" / f"{args.clip_id}.npz"
    with np.load(clip_path) as archive:
        arrays = dict(archive)
    with np.load(args.prepared / "object.npz") as archive:
        objects = dict(archive)
    with np.load(args.prepared / "subjects" / f"{entry['subject']}.npz") as archive:
        assets = dict(archive)

    def tensor(value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device="cuda")

    points = tensor(objects["object_points"])[None]
    normals = tensor(objects["object_normals"])[None]
    hands = [tensor(arrays[f"x_{side}"])[None] for side in ("lhand", "rhand")]
    obj = tensor(arrays["x_obj"])[None]
    nframes = obj.shape[1]
    if nframes < args.window:
        parser.error("selected clip is shorter than the benchmark window")
    coverage_array = np.zeros(1024, dtype=np.float32)
    active_points = arrays["rhand_contact_point_indices"][arrays["rhand_contact_mask"]]
    coverage_array[np.unique(active_points)] = 1
    coverage = tensor(coverage_array)[None]
    layers = []
    for side in ("lhand", "rhand"):
        layer = build_mano_aa(is_rhand=side == "rhand", flat_hand=True).cuda()
        layer.v_template.copy_(tensor(assets[f"{side}_v_template"]))
        layer.requires_grad_(False)
        layers.append(layer)

    with initialize_config_dir(version_base=None, config_dir=str(args.repo / "configs")):
        config = compose(config_name="config", overrides=["dataset=grab"])
    model = build_refiner(config, test=True)
    saved = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(saved["model"], strict=True)
    training_step = int(saved.get("step", 0))
    upstream_checkpoint_hash = saved.get("upstream_checkpoint_sha256")
    del saved
    model.eval().requires_grad_(False)
    limits = tensor([0.015] * 3 + [0.15] * 96)
    starts = list(range(0, max(nframes - args.window + 1, 1), args.window // 2))
    starts = sorted(set(starts + [max(0, nframes - args.window)]))

    def features(start: int, end: int):
        mask = torch.ones((1, end - start), dtype=torch.bool, device="cuda")
        return proc_refiner_input(
            hands[0][:, start:end],
            hands[1][:, start:end],
            obj[:, start:end],
            *layers,
            points,
            normals,
            torch.zeros_like(mask),
            mask,
            mask,
            coverage,
            "grab",
        )[:2]

    def clip_features():
        parts = [[], []]
        for start in range(0, nframes, 128):
            result = features(start, min(nframes, start + 128))
            for side in range(2):
                parts[side].append(result[side])
        return tuple(torch.cat(part, dim=1) for part in parts)

    def predict(inputs):
        mask = torch.ones(inputs[0].shape[:2], dtype=torch.bool, device="cuda")
        results = model(*inputs, valid_mask_lhand=torch.zeros_like(mask), valid_mask_rhand=mask)
        return tuple(
            x[..., :99] + limits * torch.tanh(0.1 * (y - x[..., :99]) / limits) for x, y in zip(inputs, results)
        )

    def window_pipeline():
        params = predict(features(0, args.window))
        return get_hand_verts(params[1], layers[1])

    def clip_predict(inputs):
        accum = [torch.zeros((nframes, 99), device="cuda") for _ in range(2)]
        denominator = torch.zeros((nframes, 1), device="cuda")
        for start in starts:
            end = min(nframes, start + args.window)
            params = predict(tuple(value[:, start:end] for value in inputs))
            weights = torch.hann_window(end - start, periodic=False, device="cuda").clamp_min(0.05)[:, None]
            for side in range(2):
                accum[side][start:end] += params[side][0] * weights
            denominator[start:end] += weights
        return tuple((value / denominator)[None] for value in accum)

    def clip_vertices(params):
        chunks = []
        for start in range(0, nframes, 128):
            end = min(nframes, start + 128)
            chunks.append(get_hand_verts(params[1][:, start:end], layers[1])[0])
        return torch.cat(chunks)

    def clip_pipeline():
        return clip_vertices(clip_predict(clip_features()))

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Existing GraspXL source-motion refinement; not text-to-motion generation or playback FPS",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "upstream_checkpoint_sha256": upstream_checkpoint_hash,
        "training_step": training_step,
        "prepared_manifest_sha256": digest(args.prepared / "manifest.json"),
        "clip_path": str(clip_path),
        "clip_sha256": digest(clip_path),
        "clip_id": args.clip_id,
        "batch_size": 1,
        "window_frames": args.window,
        "clip_frames": nframes,
        "clip_window_starts": starts,
        "clip_transformer_frames_including_overlap": sum(min(args.window, nframes - start) for start in starts),
        "feature_chunk_frames": 128,
        "mano_output_chunk_frames": 128,
        "object_points": points.shape[1],
        "hand_policy": "Absent left masked; both inputs retained by upstream refiner; only right output MANO decoded",
        "bounded_wrapper": "x + limit * tanh(0.1 * (prediction - x) / limit); 0.015 m translation, 0.15 rotation6d",
        "clip_stitching": (
            "Configured windows, half-window stride and anchored final window; Hann weights clamped to 0.05"
        ),
        "timer": "time.perf_counter, torch.cuda.synchronize before and after each timed call",
        "grad_mode": "torch.no_grad",
        "model_mode": "eval",
        "dtype": str(next(model.parameters()).dtype),
        "mixed_precision": False,
        "torch_compile": False,
        "cuda_graphs": False,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "excluded_from_all_timings": [
            "checkpoint and asset loading",
            "CPU-to-GPU input staging",
            "source contact coverage construction and prepared-data construction",
            "SDF/contact-quality evaluation",
            "rendering, GPU-to-CPU transfer, export and disk writes",
            "text encoder, diffusion generator, training",
        ],
        "stages": {},
    }
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,pstate,power.draw,temperature.gpu", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    report["nvidia_smi_before"] = query.stdout.strip()
    with torch.no_grad():
        cached_clip_features = clip_features()
        cached_window_features = tuple(value[:, : args.window] for value in cached_clip_features)
        cached_window_params = predict(cached_window_features)
        mask = torch.ones((1, args.window), dtype=torch.bool, device="cuda")
        left_mask = torch.zeros_like(mask)
        report["refiner_input_shapes"] = [list(value.shape) for value in cached_window_features]
        report["validation"] = {
            "source_mano_max_abs_error_m": float(
                (get_hand_verts(hands[1], layers[1])[0] - tensor(arrays["rhand_vertices"])).abs().max()
            ),
            "window_output_finite": bool(torch.isfinite(window_pipeline()).all()),
            "clip_output_finite": bool(torch.isfinite(clip_pipeline()).all()),
        }
        export_path = args.checkpoint.parent / "geometry" / f"{args.clip_id}.npz"
        if export_path.exists():
            with np.load(export_path) as archive:
                expected = tensor(archive["vertices_r_after"])
            report["validation"]["prior_export_max_abs_error_m"] = float((clip_pipeline() - expected).abs().max())
        stages = [
            (
                "window_network_only_cached_features",
                lambda: model(*cached_window_features, valid_mask_lhand=left_mask, valid_mask_rhand=mask),
                args.window,
            ),
            ("window_bounded_refiner_cached_features", lambda: predict(cached_window_features), args.window),
            ("window_features_only", lambda: features(0, args.window), args.window),
            ("window_right_mano_only", lambda: get_hand_verts(cached_window_params[1], layers[1]), args.window),
            ("window_features_bounded_refiner_right_mano", window_pipeline, args.window),
            ("clip_bounded_refiner_stitching_cached_features", lambda: clip_predict(cached_clip_features), nframes),
            ("clip_features_bounded_refiner_stitching_right_mano", clip_pipeline, nframes),
        ]
        for name, function, frames in stages:
            values = measure(function, frames, args.warmups, args.repeats)
            report["stages"][name] = values
            summary = {key: value for key, value in values.items() if key != "sample_seconds"}
            print(json.dumps({"stage": name, **summary}))
            sys.stdout.flush()
    report["checkpoint_unchanged"] = digest(args.checkpoint) == checkpoint_hash
    report["benchmark_script_sha256"] = digest(script_path)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"RESULTS {args.output}", flush=True)


if __name__ == "__main__":
    main()
