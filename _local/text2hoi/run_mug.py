# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the upstream GRAB mug demo with reproducible output and bounded rendering memory."""

import argparse
import importlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import trimesh
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from pytorch3d.ops import knn_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Lift a mug with the right hand.")
    parser.add_argument("--render_batch_size", type=int, default=8)
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[3] / "Text2HOI")
    args = parser.parse_args()
    if args.render_batch_size < 1:
        parser.error("render_batch_size must be positive")
    repo = args.repo_root.resolve()
    local = Path(__file__).resolve().parent
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_seed_" + str(args.seed)
    output = local / "outputs" / run_name
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("The Text2HOI environment cannot access CUDA.")
    points = torch.randn(1, 16, 3, device="cuda")
    knn_points(points, points, K=1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    demo = importlib.import_module("demo.demo")
    demo_utils = importlib.import_module("lib.utils.demo_utils")
    report = {
        "prompt": args.prompt,
        "seed": args.seed,
        "repo": str(repo),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "output": str(output),
        "render_batch_size": args.render_batch_size,
        "fps": 30,
        "mano_flat_hand_mean": True,
        "mano_betas": "zero",
        "motion_coordinates": "Upstream GRAB world coordinates before visualization recentering; meters.",
        "scope": "One released-checkpoint GRAB mug demo; not a GraspXL benchmark or physical-stability test.",
    }
    original_search_object = demo_utils.search_object
    original_search_hand = demo_utils.search_hand
    original_render = demo.Renderer.render_video
    original_proc_results = demo.proc_results

    def checked_object(*call_args, **kwargs):
        selected = original_search_object(*call_args, **kwargs)
        report["selected_object"] = selected
        if selected != "mug":
            raise ValueError("Prompt selected %r; this run requires GRAB mug." % selected)
        return selected

    def checked_hand(*call_args, **kwargs):
        selected = original_search_hand(*call_args, **kwargs)
        report["selected_hands"] = {"left": bool(selected[0]), "right": bool(selected[1])}
        if tuple(selected) != (0, 1):
            raise ValueError("Prompt did not select the right hand only.")
        return selected

    def render_chunks(renderer, vertices, faces, mesh_kind, return_numpy=True):
        chunks = []
        for start in range(0, vertices[0].shape[0], args.render_batch_size):
            part = [value[start : start + args.render_batch_size] for value in vertices]
            chunks.append(original_render(renderer, part, faces, mesh_kind, return_numpy=return_numpy))
        return np.concatenate(chunks) if return_numpy else torch.cat(chunks)

    def save_parameters(*call_args, **kwargs):
        result = original_proc_results(*call_args, **kwargs)
        values = {
            "left_hand_parameters": call_args[0],
            "right_hand_parameters": call_args[1],
            "object_parameters": call_args[2],
            "object_vertices": result[0],
            "right_hand_vertices": result[3],
            "right_hand_faces": result[4],
        }
        arrays = {key: value.detach().cpu().numpy() for key, value in values.items() if value is not None}
        arrays["object_vertices_canonical"] = call_args[3].detach().cpu().numpy()
        arrays["object_faces"] = trimesh.load(
            repo / "data/grab/processed_object_meshes/mug.ply", process=False, maintain_order=True
        ).faces
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Generated parameters or vertices contain non-finite values.")
        np.savez_compressed(output / "generated_motion.npz", **arrays)
        report["frames"] = int(arrays["right_hand_parameters"].shape[0])
        report["all_arrays_finite"] = True
        return result

    demo_utils.search_object = checked_object
    demo_utils.search_hand = checked_hand
    demo.Renderer.render_video = render_chunks
    demo.proc_results = save_parameters
    demo.make_save_folder = lambda _: str(output)
    with initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        config = compose(config_name="config", overrides=["dataset=grab"])
    with open_dict(config):
        config.test_text = [args.prompt]
        config.nsamples = 1
        config.save_obj = True
    OmegaConf.save(config, output / "config.yaml")
    report["completed"] = False
    report["phase"] = "running"
    (output / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (local / "last_run.json").write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    try:
        demo.main.__wrapped__(config)
        torch.cuda.synchronize()
        report["completed"] = True
        report["phase"] = "completed"
    except Exception as exc:
        report["completed"] = False
        report["phase"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        report["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        (output / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
        (local / "last_run.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
