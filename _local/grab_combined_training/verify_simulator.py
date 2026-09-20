# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the GRAB environment's physical constraints using a small rigid box.

This verifies collision handling and reference semantics independently of the
native mug asset, data preparation, and policy learning. It does not measure
grasp success. Run with the GraspXL environment after compiling the C++ module.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from raisimGymTorch.env.bin import ours_grab_tracking
from ruamel.yaml import RoundTripDumper, dump


def run_case(resources: Path, object_path: Path, table_collision: bool) -> dict:
    """Drive the same hand through a table with collision enabled or disabled."""
    config = {
        "num_envs": 1,
        "num_threads": 1,
        "render": False,
        "visualize": False,
        "simulation_dt": 1 / 300,
        "control_dt": 1 / 30,
        "enable_table_collision": table_collision,
        "domain_randomization": False,
    }
    env = ours_grab_tracking.RaisimGymEnv(str(resources), dump(config, Dumper=RoundTripDumper))
    try:
        env.load_multi_articulated([str(object_path)])
        hand = np.zeros((1, 51), dtype=np.float32)
        hand[0, :3] = [1.0, 0.0, 0.8]
        velocity = np.zeros_like(hand)
        mug = np.array([[1.6, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        env.reset_state(hand, hand, velocity, velocity, mug)
        state = np.zeros((1, 199), dtype=np.float32)
        obs = np.zeros((1, 383), dtype=np.float32)
        left_obs = np.zeros((1, 1), dtype=np.float32)
        env.get_global_state(state)
        np.testing.assert_allclose(state[:, :3], state[:, 115:118], atol=1e-6)
        initial = state.copy()
        joints = np.ascontiguousarray(state[:, 115:178])
        goals = np.zeros((1, 2), dtype=np.float32)
        changed_mug_goal = mug.copy()
        changed_mug_goal[0, :3] += [0.1, 0.05, 0.3]
        env.set_goals_r(changed_mug_goal, joints, hand, goals)
        env.get_global_state(state)
        env.observe(obs, left_obs)
        np.testing.assert_array_equal(state, initial)
        np.testing.assert_array_equal(obs[:, 250:257], changed_mug_goal)
        actions = np.zeros((1, 51), dtype=np.float32)
        rewards = np.zeros(1, dtype=np.float32)
        left_rewards = np.zeros(1, dtype=np.float32)
        done = np.zeros(1, dtype=bool)
        total_contacts = 0
        maximum_penetration = 0.0
        maximum_object_displacement = 0.0
        for step in range(100):
            target = hand.copy()
            target[0, 2] = 0.8 - 0.45 * min(1.0, step / 60)
            target_joints = joints.reshape(1, 21, 3).copy()
            target_joints[..., 2] += target[0, 2] - hand[0, 2]
            env.set_goals_r(mug, target_joints.reshape(1, 63), target, goals)
            env.step(actions, actions, rewards, left_rewards, done)
            env.get_global_state(state)
            assert np.isfinite(state).all(), f"Nonfinite state at step {step}"
            assert np.isfinite(rewards).all(), f"Nonfinite reward at step {step}"
            assert not done.any(), f"Unexpected terminal state at step {step}"
            total_contacts += int(state[0, 194])
            maximum_penetration = max(maximum_penetration, float(state[0, 195]))
            maximum_object_displacement = max(
                maximum_object_displacement, float(np.linalg.norm(state[0, 102:105] - mug[0, :3]))
            )
        return {
            "table_collision": table_collision,
            "reference_only_does_not_change_state": True,
            "wrist_frame_matches_state": True,
            "final_wrist_z_m": float(state[0, 2]),
            "target_wrist_z_m": float(target[0, 2]),
            "table_contacts": total_contacts,
            "maximum_table_contact_depth_m": maximum_penetration,
            "final_object_z_m": float(state[0, 104]),
            "maximum_object_displacement_m": maximum_object_displacement,
        }
    finally:
        env.close()


def verify_recorded_table(resources: Path, object_path: Path) -> dict:
    """Verify the thin table's pose, observable dimensions, support, and finite footprint."""
    dimensions = np.array([[0.45001498, 0.54001802, 0.005481]], dtype=np.float32)
    config = {
        "num_envs": 1,
        "num_threads": 1,
        "render": False,
        "visualize": False,
        "simulation_dt": 1 / 300,
        "control_dt": 1 / 30,
        "table_length": float(dimensions[0, 0]),
        "table_width": float(dimensions[0, 1]),
        "table_thickness": float(dimensions[0, 2]),
    }
    env = ours_grab_tracking.RaisimGymEnv(str(resources), dump(config, Dumper=RoundTripDumper))
    try:
        env.load_multi_articulated([str(object_path)])
        pose = np.array([[1.0, -0.08, 0.5 - dimensions[0, 2] / 2, np.cos(0.2), 0, 0, np.sin(0.2)]], np.float32)
        hand = np.zeros((1, 51), dtype=np.float32)
        hand[0, :3] = [1.7, 0.7, 0.8]
        zeros = np.zeros_like(hand)
        obj = np.array([[1.0, -0.08, 0.8, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        state = np.zeros((1, 199), dtype=np.float32)
        obs = np.zeros((1, 383), dtype=np.float32)
        left_obs = np.zeros((1, 1), dtype=np.float32)
        reward = np.zeros(1, dtype=np.float32)
        left_reward = np.zeros_like(reward)
        done = np.zeros(1, dtype=bool)
        for displaced in (False, True):
            current_pose = pose.copy()
            if displaced:
                current_pose[0, 0] += 1
            env.add_stage(dimensions, current_pose)
            env.reset_state(hand, zeros, zeros, zeros, obj)
            env.get_global_state(state)
            env.set_goals_r(obj, np.ascontiguousarray(state[:, 115:178]), hand, np.zeros((1, 2), np.float32))
            env.observe(obs, left_obs)
            np.testing.assert_allclose(obs[:, 373:376], current_pose[:, :3], atol=1e-7)
            np.testing.assert_allclose(obs[:, 376:379], dimensions, atol=1e-7)
            np.testing.assert_allclose(obs[:, 379:383], current_pose[:, 3:], atol=1e-7)
            for _ in range(120):
                env.step(zeros, zeros, reward, left_reward, done)
                env.get_global_state(state)
                if done.any():
                    break
            if displaced:
                assert done.any(), "Object outside the recorded tabletop footprint did not fall"
            else:
                assert not done.any(), "Object fell through the thin tabletop"
                supported_height = float(state[0, 104])
                assert 0.535 < supported_height < 0.545, supported_height
        return {
            "passed": True,
            "dimensions_m": dimensions[0].tolist(),
            "table_top_z_m": 0.5,
            "box_center_supported_z_m": supported_height,
            "outside_footprint_falls": True,
            "table_pose_and_dimensions_observable": True,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graspxl_root", type=Path, default=Path("/home/nmt/Projects/GraspXL"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("simulator_verification.json"))
    parser.add_argument("--prepared", type=Path, default=Path(__file__).with_name("prepared"))
    args = parser.parse_args()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="grab_tracking_verification_") as directory:
        box = Path(directory) / "rigid_box.urdf"
        box.write_text(
            '<?xml version="1.0"?><robot name="verification_box"><link name="box">'
            '<inertial><mass value="0.2"/><inertia ixx="0.000213333" ixy="0" ixz="0" '
            'iyy="0.000213333" iyz="0" izz="0.000213333"/></inertial>'
            '<collision><geometry><box size="0.08 0.08 0.08"/></geometry>'
            '<material name=""><contact name="object"/></material></collision>'
            '<visual><geometry><box size="0.08 0.08 0.08"/></geometry></visual>'
            '</link></robot>\n'
        )
        enabled = run_case(args.graspxl_root / "rsc", box, True)
        disabled = run_case(args.graspxl_root / "rsc", box, False)
        recorded_table = verify_recorded_table(args.graspxl_root / "rsc", box)
    assert enabled["table_contacts"] > 0, "Enabled hand-table collisions never produced contacts"
    assert disabled["table_contacts"] == 0, "Disabled hand-table collisions still produced contacts"
    assert enabled["final_wrist_z_m"] - disabled["final_wrist_z_m"] > 0.08, (
        "The table failed to block the descending hand"
    )
    for result in (enabled, disabled):
        assert 0.535 < result["final_object_z_m"] < 0.545, "The dynamic box did not settle on the static table"
        assert result["maximum_object_displacement_m"] > 0.2, "The box failed to fall under gravity"
    report = {
        "passed": True,
        "scope": "Mechanics regression with a rigid box; not native-mug grasp performance.",
        "enabled": enabled,
        "disabled_regression_control": disabled,
        "recorded_table": recorded_table,
        "elapsed_seconds": time.monotonic() - started,
    }
    if (args.prepared / "manifest.json").is_file():
        report["native_grab_pose_validation"] = verify_native_retarget(args.graspxl_root / "rsc", args.prepared)
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


def verify_native_retarget(resources: Path, prepared: Path) -> dict:
    """Compare native simulator joints to the retargeted references across all clips."""
    manifest = json.loads((prepared / "manifest.json").read_text())
    config = {
        "num_envs": 1,
        "num_threads": 1,
        "render": False,
        "visualize": False,
        "simulation_dt": 1 / 300,
        "control_dt": 1 / 30,
        "enable_table_collision": True,
        "domain_randomization": False,
    }
    env = ours_grab_tracking.RaisimGymEnv(str(resources), dump(config, Dumper=RoundTripDumper))
    checked = 0
    max_error = 0.0
    try:
        env.load_multi_articulated([manifest["object_urdf"]])
        state = np.zeros((1, 199), dtype=np.float32)
        zeros = np.zeros((1, 51), dtype=np.float32)
        for clip in manifest["clips"]:
            with np.load(prepared / clip["path"]) as values:
                samples = np.unique(np.linspace(0, len(values["hand_qpos"]) - 1, 3, dtype=int))
                for frame in samples:
                    hand = np.ascontiguousarray(values["hand_qpos"][frame : frame + 1], dtype=np.float32)
                    mug = np.ascontiguousarray(values["object_pose"][frame : frame + 1], dtype=np.float32)
                    env.reset_state(hand, zeros, zeros, zeros, mug)
                    env.get_global_state(state)
                    error = float(np.abs(state[0, 115:178].reshape(21, 3) - values["joints"][frame]).max())
                    assert error < 2e-6, f"Retarget FK mismatch {clip['clip_id']} frame {frame}: {error} m"
                    # The two representations must also agree after adding full turns.
                    hand[:, 3:] += np.float32(2 * np.pi)
                    # Recentring the virtual wrist-actuator base must preserve the
                    # same world pose, rather than teleporting the hand.
                    base_origin = hand[:, :3] + np.array([[-0.2, 0.1, -0.15]], dtype=np.float32)
                    hand_with_base = np.concatenate((hand, base_origin), axis=1)
                    env.reset_state(hand_with_base, zeros, zeros, zeros, mug)
                    env.get_global_state(state)
                    wrapped_error = float(np.abs(state[0, 115:178].reshape(21, 3) - values["joints"][frame]).max())
                    assert wrapped_error < 2e-6, f"Euler full-turn reset changed {clip['clip_id']}: {wrapped_error} m"
                    max_error = max(max_error, error, wrapped_error)
                    checked += 2
    finally:
        env.close()
    return {
        "passed": True,
        "clips": len(manifest["clips"]),
        "poses_including_equivalent_full_turns": checked,
        "maximum_absolute_joint_coordinate_error_m": max_error,
        "explicit_base_origin_preserves_world_pose": True,
        "object_urdf": manifest["object_urdf"],
    }


if __name__ == "__main__":
    main()
