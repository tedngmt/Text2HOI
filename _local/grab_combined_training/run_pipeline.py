# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run separate Text2HOI refinement and GraspXL policy training stages, with resume."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    local = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=local / "runs/grab_mug_100")
    parser.add_argument("--config", type=Path, default=local / "config.json")
    parser.add_argument("--epochs", type=int, default=100, help="Dataset passes for EACH trainable stage")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 100:
        parser.error("The full pipeline requires at least 100 epochs; use the stage scripts for short tests")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another pipeline process is already using this output directory")
    status_path = output / "pipeline_status.json"
    if status_path.exists() and not args.resume:
        parser.error("This run already exists; add --resume or choose another output directory")
    status = json.loads(status_path.read_text()) if status_path.exists() else {"stages": {}}
    if args.resume and status.get("epochs_per_stage") not in (None, args.epochs):
        parser.error("Resume with the original epoch target; make a new run for a changed experiment")
    config = json.loads(args.config.read_text())
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if args.resume and status.get("config_sha256") not in (None, config_hash):
        parser.error("The pipeline configuration changed; resume requires the original configuration")
    status["config_sha256"] = config_hash
    status.update(epochs_per_stage=args.epochs, pid=os.getpid(), state="running", updated=time.time())
    save(status_path, status)
    environments = {
        "text2hoi": Path("/home/nmt/miniconda3/envs/text2hoi"),
        "graspxl": Path("/home/nmt/miniconda3/envs/graspxl"),
    }
    shared_prepared = local / "prepared_source"
    collision_assets = local / "prepared/assets/mug"
    # Fail before the long refiner stage if either environment or the compiled
    # tracking module is unavailable. No training or data writes occur here.
    for name, code in (
        ("text2hoi", "import torch, hydra, pytorch3d; assert torch.cuda.is_available()"),
        (
            "graspxl",
            "import torch; from raisimGymTorch.env.bin import ours_grab_tracking; assert torch.cuda.is_available()",
        ),
    ):
        preflight_env = dict(os.environ, LD_LIBRARY_PATH="/home/nmt/raisim_build/lib:/usr/lib/wsl/lib")
        subprocess.run([str(environments[name] / "bin/python"), "-c", code], env=preflight_env, check=True, timeout=60)

    def run(
        stage: str, environment: str, script: str, arguments: list[str], resume_checkpoint: Path | None = None
    ) -> None:
        if status["stages"].get(stage, {}).get("state") == "complete":
            print(f"Already complete: {stage}", flush=True)
            return
        prefix = environments[environment]
        command = [str(prefix / "bin/python"), "-u", str(local / script), *arguments]
        if args.resume and resume_checkpoint and resume_checkpoint.exists():
            command += ["--resume", str(resume_checkpoint)]
        env = dict(os.environ)
        env.update(
            PATH=str(prefix / "bin") + os.pathsep + env.get("PATH", ""),
            LD_LIBRARY_PATH="/home/nmt/raisim_build/lib:/usr/lib/wsl/lib"
            + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
            MANO_ASSETS_ROOT="/home/nmt/manotorch/assets/mano",
            WANDB_MODE="disabled",
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="4",
        )
        status["active_stage"] = stage
        status["stages"][stage] = {"state": "running", "command": command, "started": time.time()}
        save(status_path, status)
        print(f"Starting {stage}; log: {output / (stage + '.log')}", flush=True)
        with (output / (stage + ".log")).open("a") as log:
            process = subprocess.run(command, cwd=local, env=env, stdout=log, stderr=subprocess.STDOUT)
        status["stages"][stage].update(returncode=process.returncode, finished=time.time())
        if process.returncode:
            status["stages"][stage]["state"] = "failed"
            status["state"] = "failed"
            save(status_path, status)
            raise RuntimeError(f"{stage} failed with code {process.returncode}; see its log and resume after fixing")
        status["stages"][stage]["state"] = "complete"
        save(status_path, status)

    run(
        "prepare_source",
        "graspxl",
        "prepare.py",
        ["--output_dir", str(shared_prepared), "--collision_assets", str(collision_assets)],
    )
    refiner = output / "refiner"
    run(
        "train_refiner",
        "text2hoi",
        "train_refiner.py",
        [
            "--output_dir",
            str(refiner),
            "--splits",
            str(shared_prepared / "splits.json"),
            "--epochs",
            str(args.epochs),
            "--window",
            str(config["refiner"]["window"]),
            "--learning_rate",
            str(config["refiner"]["learning_rate"]),
            "--validation_every",
            str(config["refiner"]["validation_interval"]),
            "--seed",
            str(config["refiner"]["seed"]),
            "--contact_weight",
            str(config["refiner"]["contact_weight"]),
            "--contact_retention_max_drop",
            str(config["refiner"]["contact_retention_max_drop"]),
        ],
        refiner / "last.pth",
    )
    refined = output / "prepared_refined"
    run(
        "prepare_refined",
        "graspxl",
        "prepare.py",
        [
            "--output_dir",
            str(refined),
            "--refined_dir",
            str(refiner / "references"),
            "--collision_assets",
            str(collision_assets),
        ],
    )
    policy = output / "policy"
    feasible = output / "prepared_feasible"
    run(
        "retarget_references",
        "graspxl",
        "retarget.py",
        [
            "--source_dir",
            str(refined),
            "--output_dir",
            str(feasible),
            "--iterations",
            str(config["retarget"]["iterations"]),
            "--batch_size",
            str(config["retarget"]["batch_size"]),
        ],
    )
    run(
        "train_policy",
        "graspxl",
        "train_policy.py",
        [
            "--prepared",
            str(feasible),
            "--output_dir",
            str(policy),
            "--config",
            str(args.config.resolve()),
            "--epochs",
            str(args.epochs),
        ],
        policy / "last.pt",
    )
    status.update(state="complete", active_stage=None, finished=time.time())
    save(status_path, status)
    print(f"Both {args.epochs}-epoch stages completed: {output}", flush=True)


if __name__ == "__main__":
    main()
