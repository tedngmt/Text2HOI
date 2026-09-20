# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Run resumable official-setting training, then export every GRAB clip."""

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

local = Path(__file__).resolve().parent
run = local / "run_official"
run.mkdir(exist_ok=True)
lock = (local / "pipeline.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
environment = dict(os.environ)
environment["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:/home/nmt/miniconda3/envs/text2hoi/lib"
environment["WANDB_MODE"] = "disabled"
environment["OMP_NUM_THREADS"] = "4"
python = "/home/nmt/miniconda3/envs/text2hoi/bin/python"


def status(stage, **extra):
    path = local / "pipeline_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **extra), indent=2) + "\n"
    )
    temporary.replace(path)


try:
    if not (run / "training_complete.json").exists():
        status("training", planned_epochs=650, effective_batch=64, physical_batch=8)
        command = [python, "-u", str(local / "train_official.py"), "--batch_size", "8", "--output_dir", str(run)]
        if (run / "latest.pth").exists():
            command.append("--resume")
        with (local / "training.log").open("a") as stream:
            subprocess.run(command, cwd=local, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    status("exporting_videos", planned_clips=1335)
    with (local / "export.log").open("a") as stream:
        subprocess.run(
            [
                python,
                "-u",
                str(local / "export_videos.py"),
                "--checkpoint",
                str(run / "best.pth"),
                "--output_dir",
                str(local / "share"),
            ],
            cwd=local,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    status("complete", manifest=str(local / "share/manifest.json"))
except BaseException as error:
    status("failed", error=repr(error))
    raise
