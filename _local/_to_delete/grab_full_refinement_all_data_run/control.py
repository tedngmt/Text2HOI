# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Inspect, stop, or resume the all-GRAB training/export background pipeline."""

import argparse
import json
import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path

from cooling import read_duty, set_duty

LOCAL = Path(__file__).resolve().parent
PYTHON = "/home/nmt/miniconda3/envs/text2hoi/bin/python"


def active_pipeline(state):
    pid = state.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        arguments = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        return str(LOCAL / "run_pipeline.py").encode() in arguments
    except FileNotFoundError:
        return False


def latest_step():
    path = LOCAL / "run_official/steps.jsonl"
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 16384))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            result = json.loads(line)
            if "step" in result:
                return result
        except (ValueError, UnicodeDecodeError):
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "pause", "resume", "cool", "eco", "full"))
    args = parser.parse_args()
    state_path = LOCAL / "pipeline_status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    active = active_pipeline(state)
    if args.action in ("cool", "eco", "full"):
        duty = {"cool": 50, "eco": 25, "full": 100}[args.action]
        set_duty(duty)
        print(f"Training active-time target: {duty}%. The remainder is cooling idle time.")
        print("Applies at the next microbatch and persists after pause/resume. Checkpoints are unchanged.")
    elif args.action == "status":
        print("Pipeline: " + ("RUNNING" if active else "NOT RUNNING"))
        print("Last recorded stage: " + state.get("stage", "unknown"))
        manifest_path = LOCAL / "share/manifest.json"
        if state.get("stage") in ("exporting_videos", "complete") and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                count = len(manifest.get("clips", []))
                print(f"Video export: {count}/1335 clips ({100 * count / 1335:.1f}%)")
            except ValueError:
                print("Video export manifest is being updated; check again shortly.")
        print(f"Cooling active-time target: {read_duty()}% (not a hardware utilization or temperature cap)")
        step = latest_step()
        if step:
            print(
                f"Last update: epoch {step['epoch']}/650, batch {step['batch']}/20, "
                f"update {step['step']}/13000 ({100 * step['step'] / 13000:.1f}%)"
            )
            print(f"Last update took {step['seconds']:.1f} seconds; loss {step['loss']:.6f}")
        checkpoint = LOCAL / "run_official/latest.pth"
        if checkpoint.exists():
            print("Resume checkpoint saved: " + str(datetime.fromtimestamp(checkpoint.stat().st_mtime).astimezone()))
        else:
            print("No completed-epoch checkpoint yet.")
    elif args.action == "pause":
        if not active:
            print("Pipeline is already stopped.")
            return
        pid = state["pid"]
        # The launcher creates a separate session for this pipeline. Never signal
        # an inherited shell's process group or a reused unrelated PID.
        if os.getpgid(pid) != pid:
            raise RuntimeError("Pipeline does not own its process group; refusing to stop unrelated processes")
        os.killpg(pid, signal.SIGTERM)
        print("Stop requested. Completed-epoch checkpoint is preserved.")
        print("The unfinished epoch will repeat on resume. Run 'status' to confirm it has stopped.")
    else:
        if active:
            print("Pipeline is already running; no duplicate was started.")
            return
        if (LOCAL / "share/manifest.json").exists():
            manifest = json.loads((LOCAL / "share/manifest.json").read_text())
            if manifest.get("completed") and len(manifest.get("clips", [])) == 1335:
                print("Training and all video exports are already complete.")
                return
        with (LOCAL / "pipeline.log").open("a") as log:
            job = subprocess.Popen(
                [PYTHON, "-u", str(LOCAL / "run_pipeline.py")],
                cwd=LOCAL,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"Started background pipeline PID {job.pid}. It resumes the last completed epoch automatically.")
        print("Run 'status' after a few seconds to confirm startup.")


if __name__ == "__main__":
    main()
