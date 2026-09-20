# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Apply a trained refiner to released GRAB recordings and export before/after videos.

This is recorded-motion cleanup, not the official text-generation benchmark.
Both columns use the same standard MANO shape used by upstream training.

Each video shows, for every frame and for both columns, how many hand vertices
penetrate the object, the deepest penetration, how many vertices touch it, and the
hand acceleration (a smoothness measure). Clip summaries go to the per-clip JSON and
to metrics_summary.json / metrics_summary.csv in the output folder.

Use --split validation (or test) with the held-out experiment to look only at clips
from subjects the refiner never trained on.
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import trimesh
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--clip_limit", type=int, default=0)
    parser.add_argument("--frame_limit", type=int, default=0)
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    parser.add_argument("--split_manifest", type=Path, default=None)
    parser.add_argument("--objects", default="", help="Comma-separated object names; empty means every object")
    parser.add_argument(
        "--allow_sealed_test", action="store_true", help="Export test-subject clips before the final test evaluation"
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    local = Path(__file__).resolve().parent
    repo = local.parents[1]
    experiment = local.parent / "grab_unseen_refinement"
    split_of = {}
    manifest_path = None
    if args.split != "all":
        manifest_path = (args.split_manifest or experiment / "split_manifest.json").resolve()
        split_of = {row["index"]: row for row in json.loads(manifest_path.read_text())["clips"]}
        if args.split == "test" and not args.allow_sealed_test and not (experiment / "run/test_results.json").exists():
            raise SystemExit(
                "The test subject stays sealed until the final test evaluation exists "
                "(grab_unseen_refinement/run/test_results.json). Use --split validation, "
                "or pass --allow_sealed_test if you accept looking at test clips early."
            )
    wanted_objects = {name for name in args.objects.split(",") if name}
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(local))
    from hand_object_metrics import ObjectSurface, frame_metrics, summarize
    from lib.datasets.datasets import get_dataset
    from lib.models.mano import build_mano_aa
    from lib.utils.model_utils import build_refiner
    from lib.utils.proc import proc_refiner_input
    from lib.utils.proc_output import get_hand_verts
    from lib.utils.rot import rot6d_to_rotmat

    torch.set_num_threads(4)
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["dataset=grab"])
    config = EasyDict(OmegaConf.to_container(cfg, resolve=True))
    dataset = get_dataset("Motiongrab", config.dataset)
    left = build_mano_aa(is_rhand=False, flat_hand=True).cuda()
    right = build_mano_aa(is_rhand=True, flat_hand=True).cuda()
    refiner = build_refiner(config)
    state = torch.load(checkpoint, map_location="cpu")
    refiner.load_state_dict(state["model"])
    refiner.eval()
    checkpoint_hash = digest(checkpoint)
    reports = []
    mesh_archive = zipfile.ZipFile("/home/nmt/Projects/Grab Dataset/tools__object_meshes__contact_meshes.zip")
    selected = [
        index
        for index in range(len(dataset))
        if (args.split == "all" or split_of[index]["split"] == args.split)
        and (not wanted_objects or str(dataset.obj_name[index]) in wanted_objects)
    ]
    if args.clip_limit:
        selected = selected[: args.clip_limit]
    if not selected:
        raise SystemExit("No clips match the requested split and objects.")
    if args.split == "all":
        relation, footer = "unknown", "Recorded-motion cleanup | clip may be in the training set"
    elif args.split == "train":
        relation, footer = "in-sample", "TRAINING subject (seen by the refiner) | recorded-motion cleanup"
    else:
        # Only a checkpoint from the held-out experiment proves the refiner never saw this subject.
        # It records the hash of the split manifest it was trained with.
        trained_with = (state.get("protocol") or {}).get("manifest_sha256")
        if trained_with is not None and trained_with == digest(manifest_path):
            relation = "held-out"
            footer = f"HELD-OUT {args.split} subject (never trained on) | recorded-motion cleanup"
        else:
            relation = "unverified: checkpoint is not from the held-out experiment and may have trained on this subject"
            footer = f"{args.split} split subject | this checkpoint may have trained on it (e.g. released weights)"
    benchmark_done = False
    for index in selected:
        name = str(dataset.obj_name[index])
        action = str(dataset.action_name[index]).replace(" ", "_")
        identifier = f"{index:04d}_{name}_{action}"
        movie = output / name / (identifier + ".mp4")
        report_path = movie.with_suffix(".json")
        if report_path.exists() and movie.exists():
            previous = json.loads(report_path.read_text())
            if previous["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Existing video uses another checkpoint: " + str(movie))
            if previous["video_sha256"] != digest(movie):
                raise ValueError("Existing video checksum mismatch: " + str(movie))
            reports.append(previous)
            continue
        count = int(dataset.nframes[index])
        if args.frame_limit:
            count = min(count, args.frame_limit)
        originals = [
            torch.as_tensor(np.asarray(values[index][:count], dtype=np.float32), device="cuda")
            for values in (dataset.x_lhand, dataset.x_rhand, dataset.x_obj)
        ]
        present = [bool(dataset.is_lhand[index]), bool(dataset.is_rhand[index])]
        _, points, normals, _ = dataset.object_model(name)
        pc = torch.as_tensor(points, dtype=torch.float32, device="cuda")[None]
        normal = torch.as_tensor(normals, dtype=torch.float32, device="cuda")[None]
        coverage = torch.zeros(1, 1024, device="cuda")
        for side in ("l", "r"):
            indices = np.asarray(getattr(dataset, side + "cov_idx")[index], dtype=np.int64)
            coverage[0, torch.as_tensor(indices, device="cuda")] = 1
        predicted = [torch.zeros_like(originals[0]), torch.zeros_like(originals[1])]
        denominator = torch.zeros(count, 1, device="cuda")
        starts = sorted(set(list(range(0, max(count - 150, 0) + 1, 75)) + [max(count - 150, 0)]))
        for start in starts:
            length = min(150, count - start)
            chunks = []
            for value in originals:
                chunk = torch.zeros(1, 150, value.shape[-1], device="cuda")
                chunk[0, :length] = value[start : start + length]
                chunks.append(chunk)
            masks = [torch.arange(150, device="cuda")[None] < length for _ in range(3)]
            for side in range(2):
                masks[side] &= present[side]
            il, ir, _ = proc_refiner_input(*chunks, left, right, pc, normal, *masks, coverage, "grab")
            results = refiner(il, ir, valid_mask_lhand=masks[0], valid_mask_rhand=masks[1])
            blend = torch.hann_window(length + 2, periodic=False, device="cuda")[1:-1, None]
            for side in range(2):
                predicted[side][start : start + length] += results[side][0, :length] * blend
            denominator[start : start + length] += blend
        predicted = [value / denominator for value in predicted]
        if not benchmark_done:
            benchmark_done = True

            def infer_window():
                inputs_left, inputs_right, _ = proc_refiner_input(
                    *chunks, left, right, pc, normal, *masks, coverage, "grab"
                )
                return refiner(inputs_left, inputs_right, valid_mask_lhand=masks[0], valid_mask_rhand=masks[1])

            for _ in range(5):
                infer_window()
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            for _ in range(30):
                infer_window()
            torch.cuda.synchronize()
            seconds = (time.perf_counter() - inference_start) / 30
            (output / "inference_benchmark.json").write_text(
                json.dumps(
                    dict(
                        seconds_per_window=seconds,
                        valid_frames_per_window=length,
                        padded_frames_per_window=150,
                        valid_frames_per_second=length / seconds,
                        includes="GPU refiner input preparation (MANO joints/proximity) and refiner forward",
                        excludes="diffusion generation, disk loading, final mesh export, video rendering",
                        repetitions=30,
                        warmup=5,
                        checkpoint_sha256=checkpoint_hash,
                    ),
                    indent=2,
                )
                + "\n"
            )
        before, after, faces = [], [], []
        displacement = []
        for side, layer in enumerate((left, right)):
            if not present[side]:
                continue
            vertices = []
            for params in (originals[side], predicted[side]):
                vertices.append(
                    torch.cat(
                        [
                            get_hand_verts(params[start : start + 150][None], layer)[0].cpu()
                            for start in range(0, count, 150)
                        ]
                    ).numpy()
                )
            if not all(np.isfinite(v).all() for v in vertices):
                raise FloatingPointError("Nonfinite exported hand geometry: " + identifier)
            faces.append(np.asarray(layer.faces, dtype=np.int32) + 778 * len(before))
            before.append(vertices[0])
            after.append(vertices[1])
            displacement.append(np.linalg.norm(vertices[1] - vertices[0], axis=-1))
        if not before:
            raise ValueError("No active hands: " + identifier)
        mesh = trimesh.load(
            io.BytesIO(mesh_archive.read("contact_meshes/" + name + ".ply")), file_type="ply", process=False
        )
        rotation = rot6d_to_rotmat(originals[2][:, 3:9]).transpose(1, 2).cpu().numpy()
        translation = originals[2][:, :3].cpu().numpy()
        surface = ObjectSurface(mesh.vertices, mesh.faces)
        frame_table, clip_metrics = {}, {}
        for label, hands in (("before", before), ("after", after)):
            stacked = np.concatenate(hands, axis=1)
            stats = frame_metrics(surface, stacked, rotation, translation)
            # Smoothness: mean hand-vertex acceleration from central differences at 30 FPS.
            acceleration = np.zeros(count, dtype=np.float32)
            if count >= 3:
                second = (stacked[2:] - 2 * stacked[1:-1] + stacked[:-2]) * 30.0**2
                acceleration[1:-1] = np.linalg.norm(second, axis=-1).mean(1)
                acceleration[0], acceleration[-1] = acceleration[1], acceleration[-2]
            frame_table[label] = np.stack(
                [
                    stats["penetrating_vertices"],
                    stats["max_penetration_mm"],
                    stats["contact_vertices"],
                    acceleration,
                    stats["hand_vertices"],
                ],
                axis=1,
            ).astype(np.float32)
            clip_metrics[label] = dict(summarize(stats), mean_vertex_acceleration_m_s2=float(acceleration.mean()))
        summary_line = (
            "Clip mean penetrating verts {:.1f} -> {:.1f} | frames with penetration {:.0f}% -> {:.0f}% | "
            "deepest {:.1f} -> {:.1f} mm | accel {:.2f} -> {:.2f} m/s2"
        ).format(
            clip_metrics["before"]["mean_penetrating_vertices"],
            clip_metrics["after"]["mean_penetrating_vertices"],
            clip_metrics["before"]["percent_frames_with_penetration"],
            clip_metrics["after"]["percent_frames_with_penetration"],
            clip_metrics["before"]["max_penetration_mm"],
            clip_metrics["after"]["max_penetration_mm"],
            clip_metrics["before"]["mean_vertex_acceleration_m_s2"],
            clip_metrics["after"]["mean_vertex_acceleration_m_s2"],
        )
        geometry = output / "render_input.npz"
        np.savez(
            geometry,
            metrics_before=frame_table["before"],
            metrics_after=frame_table["after"],
            original_vertices=np.concatenate(before, axis=1),
            soma_vertices=np.concatenate(after, axis=1),
            original_faces=np.concatenate(faces),
            soma_faces=np.concatenate(faces),
            object_vertices_canonical=np.asarray(mesh.vertices, dtype=np.float32),
            object_faces=np.asarray(mesh.faces, dtype=np.int32),
            object_rotation=rotation,
            object_translation=translation,
        )
        movie.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "/home/nmt/miniconda3/envs/graspxl/bin/python",
                str(local / "render_video.py"),
                "--geometry",
                str(geometry),
                "--output",
                str(movie),
                "--label",
                identifier,
                "--footer",
                footer,
                "--summary",
                summary_line,
            ],
            check=True,
        )
        report = dict(
            clip_index=index,
            clip_id=identifier,
            object=name,
            frames=count,
            fps=30,
            checkpoint_sha256=checkpoint_hash,
            checkpoint_epoch=state.get("epoch"),
            video_sha256=digest(movie),
            video=str(movie.relative_to(output)),
            mean_displacement_mm=float(np.concatenate(displacement, axis=1).mean() * 1000),
            mode="recorded-motion cleanup; standard zero-beta MANO; unchanged object trajectory",
            temporal_scope="entire retained contact interval from upstream preprocessing; raw lead-in/tail excluded",
            split=args.split,
            subject=split_of[index]["subject"] if split_of else None,
            relation_to_refiner_training=relation,
            object_mesh_watertight=surface.watertight,
            metrics=clip_metrics,
            metric_definition=(
                "Geometry on the full GRAB contact mesh: a hand vertex penetrates when more than 1 mm inside "
                "and touches when within 5 mm outside. before = recorded motion, after = refined motion. "
                "This is not the training penetration loss."
            ),
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        (output / "manifest.json").write_text(json.dumps(dict(completed=False, clips=reports), indent=2) + "\n")
        print(json.dumps(report), flush=True)
    (output / "manifest.json").write_text(json.dumps(dict(completed=True, clips=reports), indent=2) + "\n")
    measured = [r for r in reports if "metrics" in r]
    if measured:
        keys = [
            "mean_penetrating_vertices",
            "percent_frames_with_penetration",
            "max_penetration_mm",
            "mean_of_frame_max_penetration_mm",
            "mean_contact_vertices",
            "mean_vertex_acceleration_m_s2",
        ]
        weights = np.array([r["frames"] for r in measured], dtype=np.float64)
        overall = {}
        for label in ("before", "after"):
            overall[label] = {}
            for key in keys:
                values = [r["metrics"][label][key] for r in measured]
                if key == "max_penetration_mm":
                    overall[label][key] = float(max(values))
                else:
                    overall[label][key] = float(np.average(values, weights=weights))
        (output / "metrics_summary.json").write_text(
            json.dumps(
                dict(
                    clips=len(measured),
                    frames=int(weights.sum()),
                    split=args.split,
                    checkpoint_sha256=checkpoint_hash,
                    weighting="frame-weighted mean over clips; max_penetration_mm is the maximum over clips",
                    before_recorded=overall["before"],
                    after_refined=overall["after"],
                ),
                indent=2,
            )
            + "\n"
        )
        with (output / "metrics_summary.csv").open("w") as stream:
            header = ["clip_id", "object", "subject", "split", "frames"]
            for key in keys:
                header += ["before_" + key, "after_" + key]
            stream.write(",".join(header) + "\n")
            for r in measured:
                cells = [r["clip_id"], r["object"], str(r.get("subject")), str(r.get("split")), str(r["frames"])]
                for key in keys:
                    cells.append("{:.4f}".format(r["metrics"]["before"][key]))
                    cells.append("{:.4f}".format(r["metrics"]["after"][key]))
                stream.write(",".join(cells) + "\n")
    (output / "render_input.npz").unlink(missing_ok=True)
    mesh_archive.close()


if __name__ == "__main__":
    main()
