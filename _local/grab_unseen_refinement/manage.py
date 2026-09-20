# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Control this held-out experiment independently of the all-data video export."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / "refiner_tools"))
import contextlib

from cooling import read_duty, set_duty

PYTHON = "/home/nmt/miniconda3/envs/text2hoi/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "resume", "pause", "test", "cool", "eco", "full"))
    args = parser.parse_args()
    path = LOCAL / "process.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    pid = state.get("pid", 0)
    active = False
    if isinstance(pid, int) and pid > 1:
        with contextlib.suppress(FileNotFoundError):
            active = str(LOCAL / "train.py").encode() in (Path("/proc") / str(pid) / "cmdline").read_bytes().split(
                b"\0"
            )
    if args.action in ("cool", "eco", "full"):
        duty = {"cool": 50, "eco": 25, "full": 100}[args.action]
        set_duty(duty, LOCAL / "cooling_settings.json")
        print(f"This experiment's active-time target: {duty}%")
    elif args.action == "status":
        print("RUNNING" if active else "NOT RUNNING")
        print(f"Cooling active-time target: {read_duty(LOCAL / 'cooling_settings.json')}%")
        log = LOCAL / "run/steps.jsonl"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 16384))
                for line in reversed(stream.read().splitlines()):
                    try:
                        row = json.loads(line)
                        print(f"Last training update: epoch {row['epoch']}/650, batch {row['batch']}/16")
                        break
                    except ValueError:
                        continue
        print("Final test report:", (LOCAL / "run/test_results.json").exists())
    elif args.action == "pause":
        if not active:
            print("Already stopped")
            return
        if os.getpgid(pid) != pid:
            raise RuntimeError("Refusing to signal a process group not owned by this job")
        os.killpg(pid, signal.SIGTERM)
        print("Stop requested; latest completed-epoch checkpoint preserved")
    else:
        if active:
            print("Already running")
            return
        if args.action == "test" and not (LOCAL / "run/training_complete.json").exists():
            raise RuntimeError("Complete training and validation selection before opening the test set")
        if args.action == "resume" and (LOCAL / "run/training_complete.json").exists():
            print("Training complete; use test for the final held-out evaluation")
            return
        if not (LOCAL / "smoke/smoke_complete.json").exists():
            raise RuntimeError("Smoke validation must pass before launching the long experiment")
        mode = "test" if args.action == "test" else "train"
        command = [PYTHON, "-u", str(LOCAL / "train.py"), "--mode", mode]
        if mode == "train" and (LOCAL / "run/latest.pth").exists():
            command.append("--resume")
        environment = dict(
            os.environ,
            LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/nmt/miniconda3/envs/text2hoi/lib",
            OMP_NUM_THREADS="4",
            WANDB_MODE="disabled",
        )
        with (LOCAL / (mode + ".log")).open("a") as log:
            job = subprocess.Popen(
                command,
                cwd=LOCAL,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        path.write_text(json.dumps(dict(pid=job.pid, mode=mode), indent=2) + "\n")
        print(f"Started {mode}, PID {job.pid}")


if __name__ == "__main__":
    main()
