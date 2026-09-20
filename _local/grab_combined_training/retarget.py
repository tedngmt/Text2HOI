# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fit GRAB references within the simulator's finger limits and recover its table.

This is deterministic per-pose kinematic fitting, not learning from held-out
recordings. The wrist, object motion, dataset split, and source timing stay fixed.
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from prepare import FINGERS, generic_hand_joints, sha256, write_json
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def read_hand(urdf: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read 45 angular bounds [rad] and finger segment offsets [m]."""
    root = ET.parse(urdf).getroot()
    movable = [joint for joint in root.findall("joint") if joint.attrib["type"] != "fixed"]
    lower = np.asarray([float(j.find("limit").attrib["lower"]) for j in movable[6:]], dtype=np.float32)
    upper = np.asarray([float(j.find("limit").attrib["upper"]) for j in movable[6:]], dtype=np.float32)
    assert lower.shape == upper.shape == (45,)
    offsets = []
    for finger in FINGERS:
        values = []
        for name in [f"right_{finger}{index}_x" for index in (1, 2, 3)] + [f"right_{finger}_tip"]:
            origin = root.find(f"joint[@name='{name}']/origin")
            assert np.allclose(np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" "), 0)
            values.append(np.fromstring(origin.attrib["xyz"], sep=" "))
        offsets.append(values)
    return lower, upper, np.asarray(offsets, dtype=np.float32)


def euler_matrices(angles: torch.Tensor) -> torch.Tensor:
    """Convert intrinsic XYZ angles [rad] to active rotation matrices."""
    sine, cosine = torch.sin(angles), torch.cos(angles)
    sx, sy, sz = sine.unbind(-1)
    cx, cy, cz = cosine.unbind(-1)
    values = (
        cy * cz,
        -cy * sz,
        sy,
        sx * sy * cz + cx * sz,
        -sx * sy * sz + cx * cz,
        -sx * cy,
        -cx * sy * cz + sx * sz,
        cx * sy * sz + sx * cz,
        cx * cy,
    )
    return torch.stack(values, dim=-1).reshape(*angles.shape[:-1], 3, 3)


def finger_positions(angles: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Return five fingers' 4 joint/tip positions [m] in wrist coordinates."""
    matrices = euler_matrices(angles.reshape(-1, 5, 3, 3))
    first = offsets[None, :, 0].expand(len(angles), -1, -1)
    rotation = matrices[:, :, 0]
    second = first + (rotation @ offsets[None, :, 1, :, None]).squeeze(-1)
    rotation = rotation @ matrices[:, :, 1]
    third = second + (rotation @ offsets[None, :, 2, :, None]).squeeze(-1)
    rotation = rotation @ matrices[:, :, 2]
    tip = third + (rotation @ offsets[None, :, 3, :, None]).squeeze(-1)
    return torch.stack((first, second, third, tip), dim=2)


def table_scene(item: dict, clip: dict) -> tuple[np.ndarray, np.ndarray]:
    """Recover the thin captured tabletop's dimensions [m] and xyz+wxyz pose."""
    dimensions = np.asarray([0.45001498, 0.54001802, 0.005481], dtype=np.float32)
    alignment = item["world_to_training_rotation"]
    offset = item["world_to_training_translation"]
    provenance = clip["table_alignment"]
    normal = np.asarray(provenance["source_table_normal"])
    center = np.asarray(provenance["source_table_top_point_m"]) - normal * float(dimensions[2]) * 0.5
    center = alignment @ center + offset
    source = Rotation.from_rotvec(item["source_table_global_orient"]).as_matrix().transpose(0, 2, 1)
    x_axis = np.median(source[min(30, len(source) // 4) :, :, 0] @ alignment.T, axis=0)
    yaw = np.arctan2(x_axis[1], x_axis[0])
    quaternion = Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
    return dimensions, np.r_[center, quaternion].astype(np.float32)


def feasible_angles(source: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, dict]:
    """Choose the less violating equivalent XYZ branch before constrained fitting."""
    principal = (source + np.pi) % (2 * np.pi) - np.pi
    value = principal.reshape(-1, 15, 3)
    alternative = np.stack((value[..., 0] + np.pi, np.pi - value[..., 1], value[..., 2] + np.pi), axis=-1)
    alternative = (alternative + np.pi) % (2 * np.pi) - np.pi
    lo, hi = lower.reshape(15, 3), upper.reshape(15, 3)
    excess = np.maximum(lo - value, 0) + np.maximum(value - hi, 0)
    alternative_excess = np.maximum(lo - alternative, 0) + np.maximum(alternative - hi, 0)
    replace = (alternative_excess**2).sum(-1) < (excess**2).sum(-1)
    chosen = np.where(replace[..., None], alternative, value).reshape(-1, 45)
    report = {
        "out_of_bounds_frame_fraction_before": float((excess > 1e-6).any(axis=(1, 2)).mean()),
        "out_of_bounds_coordinate_fraction_before": float((excess > 1e-6).mean()),
        "maximum_excess_before_rad": float(excess.max()),
        "equivalent_branch_replacements": int(replace.sum()),
        "joint_rotations_fully_fixed_by_equivalent_branch": int(
            ((excess > 1e-6).any(-1) & (alternative_excess <= 1e-6).all(-1)).sum()
        ),
    }
    return chosen, report


def fit_clip(
    qpos: np.ndarray,
    target_joints: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    offsets: np.ndarray,
    steps: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit finger angles within bounds while retaining wrist coordinates [m, rad]."""
    reference_angles, report = feasible_angles(qpos[:, 6:], lower, upper)
    principal = (qpos[:, 6:] + np.pi) % (2 * np.pi) - np.pi
    clipped = qpos.copy()
    clipped[:, 6:] = np.clip(principal, lower, upper)
    fitted = qpos.copy()
    root_matrices = Rotation.from_euler("XYZ", qpos[:, 3:6]).as_matrix()
    local_target = np.einsum("tji,tkj->tki", root_matrices, target_joints[:, 1:] - qpos[:, None, :3])
    offsets_t = torch.as_tensor(offsets, device=device)
    lower_t, upper_t = (torch.as_tensor(x, device=device) for x in (lower, upper))
    weights = torch.tensor([1.0, 1.0, 1.0, 2.0], device=device)[None, None, :, None]
    for start in range(0, len(qpos), batch_size):
        stop = min(start + batch_size, len(qpos))
        target = torch.as_tensor(local_target[start:stop].reshape(-1, 5, 4, 3), dtype=torch.float32, device=device)
        source_t = torch.as_tensor(reference_angles[start:stop], device=device)
        naive_t = torch.as_tensor(clipped[start:stop, 6:], device=device)
        initial = source_t.clamp(lower_t, upper_t)

        def errors(value: torch.Tensor) -> torch.Tensor:
            return (((finger_positions(value, offsets_t) - target) * 1000.0).square() * weights).mean((1, 2, 3))

        with torch.no_grad():
            choose = errors(initial) < errors(naive_t)
            initial = torch.where(choose[:, None], initial, naive_t)
            best = initial.clone()
            best_error = errors(best)
        variable = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([variable], lr=0.04)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            position_error = errors(variable)
            difference = variable - source_t
            angular_difference = torch.atan2(difference.sin(), difference.cos())
            loss = position_error.mean() + 0.01 * angular_difference.square().mean()
            loss.backward()
            if not torch.isfinite(variable.grad).all():
                raise ValueError("Nonfinite retarget gradient")
            optimizer.step()
            with torch.no_grad():
                variable.clamp_(lower_t, upper_t)
                candidate_error = errors(variable)
                improved = candidate_error < best_error
                best[improved] = variable[improved]
                best_error = torch.minimum(best_error, candidate_error)
        fitted[start:stop, 6:] = best.detach().cpu().numpy()
    if not np.isfinite(fitted).all() or np.any(fitted[:, 6:] < lower - 1e-6) or np.any(fitted[:, 6:] > upper + 1e-6):
        raise ValueError("Retargeting failed to enforce finite finger joint limits")
    report["out_of_bounds_frame_fraction_after"] = 0.0
    return fitted, clipped, report


def point_distances(joints: np.ndarray, object_pose: np.ndarray, tree: cKDTree) -> np.ndarray:
    """Return unsigned closest native-mesh vertex distances [m], not penetration."""
    rotations = Rotation.from_quat(object_pose[:, [4, 5, 6, 3]]).as_matrix()
    canonical = np.einsum("tji,tkj->tki", rotations, joints - object_pose[:, None, :3])
    return tree.query(canonical.reshape(-1, 3))[0].reshape(len(joints), 21)


def retarget_directory(
    source_dir: Path, output_dir: Path, steps: int = 100, batch_size: int = 512, device: str = "cuda"
) -> dict:
    """Create bounded references and matching original tabletop geometry."""
    started = time.monotonic()
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("Retargeting must write a new directory to preserve original references")
    torch.set_num_threads(2)
    manifest = json.loads((source_dir / "manifest.json").read_text())
    urdf = Path(manifest["hand_urdf"])
    lower, upper, offsets = read_hand(urdf)
    objects = np.load(Path(__file__).resolve().parents[1] / "grab_mug_refinement/prepared/object.npz")
    tree = cKDTree(objects["object_vertices_canonical"])
    (output_dir / "clips").mkdir(parents=True, exist_ok=True)
    records, metrics = [], []
    for clip in manifest["clips"]:
        with np.load(source_dir / clip["path"], allow_pickle=False) as source:
            item = {name: source[name].copy() for name in source.files}
        unbounded_qpos = item["hand_qpos"].copy()
        unbounded_joints = item["joints"].copy()
        qpos, clipped, report = fit_clip(
            unbounded_qpos, unbounded_joints, lower, upper, offsets, steps, batch_size, device
        )
        joints = generic_hand_joints(urdf, qpos).astype(np.float32)
        clipped_joints = generic_hand_joints(urdf, clipped).astype(np.float32)
        fitted_error = np.linalg.norm(joints - unbounded_joints, axis=-1)
        clipped_error = np.linalg.norm(clipped_joints - unbounded_joints, axis=-1)
        tips = [4, 8, 12, 16, 20]
        original_distance = point_distances(unbounded_joints, item["object_pose"], tree)
        fitted_distance = point_distances(joints, item["object_pose"], tree)
        clipped_distance = point_distances(clipped_joints, item["object_pose"], tree)
        proximity = (original_distance < 0.02) & item["contact"][:, None].astype(bool)
        report.update(
            {
                "clip_id": clip["clip_id"],
                "frames": len(qpos),
                "naive_clipping_mean_joint_error_mm": float(clipped_error.mean() * 1000),
                "fitted_mean_joint_error_mm": float(fitted_error.mean() * 1000),
                "fitted_max_joint_error_mm": float(fitted_error.max() * 1000),
                "naive_clipping_mean_tip_error_mm": float(clipped_error[:, tips].mean() * 1000),
                "fitted_mean_tip_error_mm": float(fitted_error[:, tips].mean() * 1000),
                "original_proximity_joint_samples": int(proximity.sum()),
                "fitted_proximity_retention": float((fitted_distance[proximity] < 0.02).mean())
                if proximity.any()
                else None,
                "clipped_proximity_retention": float((clipped_distance[proximity] < 0.02).mean())
                if proximity.any()
                else None,
                "fitted_mean_contact_distance_change_mm": float(
                    np.abs(fitted_distance - original_distance)[proximity].mean() * 1000
                )
                if proximity.any()
                else None,
                "clipped_mean_contact_distance_change_mm": float(
                    np.abs(clipped_distance - original_distance)[proximity].mean() * 1000
                )
                if proximity.any()
                else None,
            }
        )
        item.update(
            hand_qpos=qpos,
            joints=joints,
            hand_qpos_unbounded=unbounded_qpos,
            joints_unbounded=unbounded_joints,
            hand_qpos_naive_clipped=clipped,
            finger_limits_lower=lower,
            finger_limits_upper=upper,
        )
        item["table_dimensions"], item["table_pose"] = table_scene(item, clip)
        target = output_dir / "clips" / f"{clip['clip_id']}.npz"
        np.savez_compressed(target, **item)
        record = dict(
            clip,
            path=str(target.relative_to(output_dir)),
            sha256=sha256(target),
            retargeting=report,
            table_dimensions=item["table_dimensions"].tolist(),
            table_pose=item["table_pose"].tolist(),
        )
        records.append(record)
        metrics.append(report)
        print(
            json.dumps(
                {
                    "retargeted": clip["clip_id"],
                    "naive_mm": report["naive_clipping_mean_joint_error_mm"],
                    "fitted_mm": report["fitted_mean_joint_error_mm"],
                }
            ),
            flush=True,
        )
    manifest.update(
        clips=records,
        parent_preparation=str(source_dir / "manifest.json"),
        parent_preparation_sha256=sha256(source_dir / "manifest.json"),
        retarget_policy=(
            "Projected Adam with original URDF finger bounds; fixed wrist and generic unbounded FK targets; "
            "2x tip weight. This enforces joint bounds, not collision-free poses or physical holding."
        ),
        scene_limit=(
            "Original GRAB thin tabletop approximated by its oriented bounding box; "
            "no body, head, recipient, other hand or table legs."
        ),
        table_geometry="Per-clip captured center/yaw, 0.450015 x 0.540018 x 0.005481 m; upper plane z=0.5 m.",
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "splits.json", manifest["splits"])
    report = {
        "clips": metrics,
        "iterations": steps,
        "batch_size": batch_size,
        "device": device,
        "total_frames": sum(c["frames"] for c in metrics),
        "elapsed_seconds": time.monotonic() - started,
        "all_finger_limits_satisfied": True,
        "wrist_and_object_unchanged": True,
        "contact_metric_limit": (
            "Unsigned nearest native vertex proximity; not collision depth or proof of physical holding."
        ),
    }
    write_json(output_dir / "retarget_report.json", report)
    print(
        json.dumps(
            {"completed": len(records), "output": str(output_dir), "elapsed_seconds": report["elapsed_seconds"]}
        ),
        flush=True,
    )
    return report


def main() -> None:
    """Run bounds-aware reference fitting without changing source assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    retarget_directory(args.source_dir, args.output_dir, args.iterations, args.batch_size, args.device)


if __name__ == "__main__":
    main()
