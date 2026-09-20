# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare zero-action GRAB tracking with and without planned PD velocity.

Uses only validation recordings and a fixed environment. The two conditions
share references, initial states, physical parameters, and zero residual policy.
This is a controller regression, not a training/generalization benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from raisimGymTorch.env.bin import ours_grab_tracking
from ruamel.yaml import RoundTripDumper, dump
from scipy.spatial.transform import Rotation


def velocity(current: np.ndarray, previous: np.ndarray, dt: float) -> np.ndarray:
    """Return finite difference generalized velocity [m/s, rad/s]."""
    delta = current - previous
    delta[3:] = np.arctan2(np.sin(delta[3:]), np.cos(delta[3:]))
    result = delta / dt
    result[:3] = np.clip(result[:3], -5, 5)
    result[3:] = np.clip(result[3:], -20, 20)
    return result


def evaluate(args: argparse.Namespace, enabled: bool) -> dict:
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    dt = 1 / 30
    cfg = {
        "num_envs": 1,
        "num_threads": 1,
        "render": False,
        "visualize": False,
        "simulation_dt": dt / 10,
        "control_dt": dt,
        "enable_table_collision": True,
        "domain_randomization": False,
    }
    table_scenes = json.loads(args.table_metadata.read_text()) if args.table_metadata else None
    if table_scenes:
        dimensions = np.asarray(next(iter(table_scenes.values()))["table_dimensions"])
        cfg.update(
            table_length=float(dimensions[0]), table_width=float(dimensions[1]), table_thickness=float(dimensions[2])
        )
    env = ours_grab_tracking.RaisimGymEnv(str(args.graspxl_root / "rsc"), dump(cfg, Dumper=RoundTripDumper))
    rows = []
    state = np.zeros((1, 199), dtype=np.float32)
    zeros = np.zeros((1, 51), dtype=np.float32)
    rewards = np.zeros(1, dtype=np.float32)
    left_rewards = np.zeros(1, dtype=np.float32)
    done = np.zeros(1, dtype=bool)
    observation = np.zeros((1, 383), dtype=np.float32)
    left_observation = np.zeros((1, 1), dtype=np.float32)
    try:
        env.load_multi_articulated([manifest["object_urdf"]])
        for clip in manifest["clips"]:
            if clip["split"] != "validation":
                continue
            with np.load(args.prepared / clip["path"]) as archive:
                data = {name: archive[name] for name in ("hand_qpos", "object_pose", "joints", "contact")}
            for episode in clip["episodes"]:
                for start in range(episode["start"], episode["end"] - 1, args.window_frames):
                    end = min(start + args.window_frames + 1, episode["end"])
                    q = np.ascontiguousarray(data["hand_qpos"][start : start + 1], dtype=np.float32)
                    initial_velocity = np.clip(velocity(data["hand_qpos"][start + 1], q[0], dt), -10, 10)[None]
                    object_state = np.zeros((1, 13), dtype=np.float32)
                    object_state[0, :7] = data["object_pose"][start]
                    object_state[0, 7:10] = (
                        data["object_pose"][start + 1, :3] - data["object_pose"][start, :3]
                    ) / dt
                    object_rotations = Rotation.from_quat(data["object_pose"][start : start + 2, [4, 5, 6, 3]])
                    object_state[0, 10:13] = (object_rotations[1] * object_rotations[0].inv()).as_rotvec() / dt
                    if table_scenes:
                        scene = table_scenes[clip["clip_id"]]
                        env.add_stage(
                            np.asarray(scene["table_dimensions"], dtype=np.float32)[None],
                            np.asarray(scene["table_pose"], dtype=np.float32)[None],
                        )
                    if args.center_base:
                        window_positions = data["hand_qpos"][start:end, :3]
                        span = np.ptp(window_positions, axis=0)
                        if np.max(span) > 1.5:
                            raise ValueError("This diagnostic requires shorter windows to preserve 1.5 m root spans")
                        origin = 0.5 * (window_positions.min(axis=0) + window_positions.max(axis=0))
                        q = np.concatenate((q, origin[None].astype(np.float32)), axis=1)
                    env.reset_state(q, zeros, initial_velocity.astype(np.float32), zeros, object_state)
                    row = {
                        "clip_id": clip["clip_id"],
                        "start": start,
                        "frames": 0,
                        "failed": False,
                        "wrist_error_sum_m": 0.0,
                        "joint_error_sum_m": 0.0,
                        "object_error_sum_m": 0.0,
                        "table_frames": 0,
                        "table_depth_max_m": 0.0,
                    }
                    for frame in range(start + 1, end):
                        hand = np.ascontiguousarray(data["hand_qpos"][frame : frame + 1], dtype=np.float32)
                        obj = np.ascontiguousarray(data["object_pose"][frame : frame + 1], dtype=np.float32)
                        joints = np.ascontiguousarray(data["joints"][frame].reshape(1, 63), dtype=np.float32)
                        extra = np.zeros((1, 53 if enabled else 2), dtype=np.float32)
                        extra[0, :2] = [(frame - start) / (end - start - 1), data["contact"][frame]]
                        if enabled:
                            extra[0, 2:] = velocity(hand[0], data["hand_qpos"][frame - 1], dt)
                        env.set_goals_r(obj, joints, hand, extra)
                        env.observe(observation, left_observation)
                        if enabled:
                            np.testing.assert_array_equal(observation[0, 322:373], extra[0, 2:])
                        else:
                            np.testing.assert_array_equal(observation[0, 322:373], 0)
                        env.step(zeros, zeros, rewards, left_rewards, done)
                        env.get_global_state(state)
                        if done.any():
                            row["failed"] = True
                            break
                        assert np.isfinite(state).all()
                        row["frames"] += 1
                        row["wrist_error_sum_m"] += float(np.linalg.norm(state[0, :3] - hand[0, :3]))
                        row["joint_error_sum_m"] += float(
                            np.linalg.norm(state[0, 115:178].reshape(21, 3) - joints.reshape(21, 3), axis=-1).mean()
                        )
                        row["object_error_sum_m"] += float(np.linalg.norm(state[0, 102:105] - obj[0, :3]))
                        row["table_frames"] += int(state[0, 194] > 0)
                        row["table_depth_max_m"] = max(row["table_depth_max_m"], float(state[0, 195]))
                    rows.append(row)
                    if args.window_limit and len(rows) >= args.window_limit:
                        break
                if args.window_limit and len(rows) >= args.window_limit:
                    break
            if args.window_limit and len(rows) >= args.window_limit:
                break
    finally:
        env.close()
    frames = sum(row["frames"] for row in rows)
    return {
        "planned_velocity": enabled,
        "windows": len(rows),
        "frames": frames,
        "failed_windows": sum(row["failed"] for row in rows),
        "mean_wrist_error_m": sum(row["wrist_error_sum_m"] for row in rows) / max(1, frames),
        "mean_joint_error_m": sum(row["joint_error_sum_m"] for row in rows) / max(1, frames),
        "mean_object_error_m": sum(row["object_error_sum_m"] for row in rows) / max(1, frames),
        "maximum_table_depth_m": max(row["table_depth_max_m"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--graspxl_root", type=Path, default=Path("/home/nmt/Projects/GraspXL"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("velocity_tracking_verification.json"))
    parser.add_argument("--window_frames", type=int, default=96)
    parser.add_argument("--window_limit", type=int, default=0)
    parser.add_argument("--table_metadata", type=Path, help="Use recorded GRAB tabletop dimensions and poses")
    parser.add_argument("--center_base", action="store_true", help="Centre virtual wrist actuators within each window")
    args = parser.parse_args()
    started = time.monotonic()
    before = evaluate(args, False)
    after = evaluate(args, True)
    report = {
        "scope": "Validation-only zero-residual controller comparison; no optimization or test-split evaluation.",
        "manifest_sha256": hashlib.sha256((args.prepared / "manifest.json").read_bytes()).hexdigest(),
        "table_scene": "recorded_grab" if args.table_metadata else "large_regression_proxy",
        "centered_wrist_base": args.center_base,
        "zero_desired_velocity": before,
        "planned_desired_velocity": after,
        "joint_error_change_fraction": after["mean_joint_error_m"] / before["mean_joint_error_m"] - 1,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = {k: v for k, v in report.items() if k not in ("zero_desired_velocity", "planned_desired_velocity")}
    print(json.dumps(summary))
    for value in (before, after):
        print(json.dumps({k: v for k, v in value.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
