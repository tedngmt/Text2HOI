# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time the released Text2HOI mug generation pipeline without rendering/export."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict


def main() -> None:
    local = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=local.parents[3] / "Text2HOI")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    assert args.repeats >= 3 and args.warmups >= 1
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA access is required")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2)
    with initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        config = compose(config_name="config", overrides=["dataset=grab"])
    with open_dict(config):
        config.test_text = ["Lift a mug with the right hand."]
        config.nsamples = args.warmups + args.repeats
        config.save_obj = False
    demo = importlib.import_module("demo.demo")
    records = []
    current = {}

    def clock():
        torch.cuda.synchronize()
        return time.perf_counter()

    def timed(stage, function):
        @functools.wraps(function)
        def call(*call_args, **kwargs):
            start = clock()
            value = function(*call_args, **kwargs)
            current[stage] = clock() - start
            return value

        return call

    original_seq = demo.build_seq_cvae
    original_diffusion = demo.build_model_and_diffusion
    original_refiner = demo.build_refiner
    original_masks = demo.get_valid_mask_bunch
    original_results = demo.proc_results

    def sequence_model(*call_args, **kwargs):
        model = original_seq(*call_args, **kwargs)
        original_decode = model.decode

        def decode(*decode_args, **decode_kwargs):
            current.clear()
            current["started"] = clock()
            return timed("duration_seconds", original_decode)(*decode_args, **decode_kwargs)

        model.decode = decode
        return model

    def diffusion_model(*call_args, **kwargs):
        model, diffusion = original_diffusion(*call_args, **kwargs)
        diffusion.sampling = timed("motion_diffusion_seconds", diffusion.sampling)
        return model, diffusion

    def refinement_model(*call_args, **kwargs):
        model = original_refiner(*call_args, **kwargs)
        model.forward = timed("refiner_network_seconds", model.forward)
        return model

    def masks(*call_args, **kwargs):
        current["valid_frames"] = int(call_args[3][0].item())
        return original_masks(*call_args, **kwargs)

    def results(*call_args, **kwargs):
        value = timed("mesh_decoding_seconds", original_results)(*call_args, **kwargs)
        current["total_seconds"] = clock() - current.pop("started")
        current["output_frames"] = int(value[3].shape[0])
        assert current["valid_frames"] == current["output_frames"]
        assert all(
            torch.isfinite(item).all().item() for item in (call_args[0], call_args[1], call_args[2], value[0], value[3])
        )
        current["all_output_arrays_finite"] = True
        records.append(dict(current))
        print("TIMING " + json.dumps(records[-1]), flush=True)
        return value

    demo.build_seq_cvae = sequence_model
    demo.build_model_and_diffusion = diffusion_model
    demo.build_refiner = refinement_model
    demo.get_valid_mask_bunch = masks
    demo.proc_obj_feat_final = timed("contact_prediction_seconds", demo.proc_obj_feat_final)
    demo.proc_refiner_input = timed("refiner_features_seconds", demo.proc_refiner_input)
    demo.proc_results = results
    # Keep upstream generation unchanged; omit the independent display/export stages.
    demo.Renderer = lambda **kwargs: None
    demo.render_videos = lambda *call_args, **kwargs: None
    demo.save_video = lambda *call_args, **kwargs: None
    demo.make_save_folder = lambda *call_args: str(local)
    # tqdm display is irrelevant to inference and otherwise floods the timing log.
    diffusion_module = importlib.import_module("lib.networks.diffusion")
    diffusion_module.tqdm = SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable)
    started = clock()
    demo.main.__wrapped__(config)
    complete_seconds = clock() - started
    assert len(records) == args.repeats + args.warmups
    measured = records[args.warmups :]
    total_seconds = sum(row["total_seconds"] for row in measured)
    total_frames = sum(row["output_frames"] for row in measured)
    stage_names = [key for key in measured[0] if key.endswith("seconds")]
    report = {
        "complete": True,
        "prompt": list(config.test_text),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model_dtype": "float32; upstream CLIP uses its own evaluation dtype",
        "batch_size": 1,
        "cpu_threads": torch.get_num_threads(),
        "diffusion_steps": int(config.diffusion.T),
        "padded_frames_per_sample": int(config.dataset.max_nframes),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "generated_valid_frames": total_frames,
        "measured_seconds": total_seconds,
        "valid_generated_frames_per_second": total_frames / total_seconds,
        "padded_frames_per_second": args.repeats * config.dataset.max_nframes / total_seconds,
        "stage_statistics": {
            stage: {
                "mean_seconds": float(np.mean([row[stage] for row in measured])),
                "median_seconds": float(np.median([row[stage] for row in measured])),
                "min_seconds": min(row[stage] for row in measured),
                "max_seconds": max(row[stage] for row in measured),
            }
            for stage in stage_names
        },
        "timing_scope": (
            "Warm loaded models, cached text/object features; duration and contact prediction, "
            "1000-step diffusion, refiner feature construction, refiner, MANO/object mesh decoding. "
            "Synchronized wall-clock with synchronization at stage boundaries. "
            "Excludes model loading, initial object selection/mesh loading/text and PointNet encoding, "
            "rendering, video/mesh export. Whole-sequence throughput, not a causal streaming controller."
        ),
        "all_trials_including_warmup": records,
        "overall_seconds_including_model_setup_and_warmup": complete_seconds,
        "script_sha256": hashlib.sha256((local / "benchmark_generation.py").read_bytes()).hexdigest(),
    }
    (local / "generation_benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
