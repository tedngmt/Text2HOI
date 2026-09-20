# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared overview and grip-detail cameras for paired mug videos."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def canonical_to_z_up(kind: str) -> np.ndarray:
    """Return the common rigid display rotation without changing mesh scale."""
    if kind == "GRAB":
        return np.eye(3, dtype=np.float64)
    if kind == "GraspXL":
        # A +90 degree rotation about X maps the mug's +Y axis to +Z.
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    raise ValueError(f"Unknown dataset: {kind}")


def object_local_vertices(
    vertices: np.ndarray,
    object_rotation: np.ndarray,
    object_translation: np.ndarray,
    kind: str,
) -> np.ndarray:
    """Transform world vertices [m] to object-stabilized, Z-up vertices [m].

    The same transform must be used for original and SOMA meshes. Inputs may
    describe one frame or a batch of frames, with active object rotation matrices.
    """
    local = np.matmul(
        np.asarray(vertices) - np.asarray(object_translation)[..., None, :],
        np.asarray(object_rotation),
    )
    return local @ canonical_to_z_up(kind).T


def _camera_spec(target, view_direction, height, aspect, depth_extent=0.0):
    """Build a JSON-compatible orthographic camera specification [m]."""
    direction = np.asarray(view_direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    target = np.asarray(target, dtype=np.float64)
    distance = max(2.0, depth_extent / 2.0 + 1.0)
    return {
        "target": target.tolist(),
        "view_direction": direction.tolist(),
        "up": [0.0, 0.0, 1.0],
        "height": float(height),
        "width": float(height * aspect),
        "eye": (target + direction * distance).tolist(),
        "znear": 0.01,
        "zfar": float(distance + depth_extent / 2.0 + 2.0),
    }


def _grab_detail_specs(rotation, translation, minimum, maximum, fps, aspect):
    """Follow the mug with a world-up camera while preserving its real tilt [m]."""
    count = len(rotation)
    center = (minimum + maximum) / 2.0
    grip_center = center + np.array([0.012, 0.0, 0.0])
    desired_targets = grip_center @ rotation.transpose(0, 2, 1) + translation
    centers_world = center @ rotation.transpose(0, 2, 1) + translation
    plane_normals = rotation[:, :, 1]
    handle_axes = rotation[:, :, 0]
    horizontal_norm = np.linalg.norm(plane_normals[:, :2], axis=1)
    elevation = np.arctan2(0.45, np.hypot(0.32, 0.85))
    cosine, sine = np.cos(elevation), np.sin(elevation)

    def candidate_yaw(frame, side):
        horizontal = 0.85 * side * plane_normals[frame, :2] + 0.32 * handle_axes[frame, :2]
        if np.linalg.norm(horizontal) < 1e-10:
            return None
        return np.arctan2(horizontal[1], horizontal[0])

    def wrapped_angle(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    # Select one side using object geometry only. This avoids the handle plane
    # becoming edge-on when the real drinking motion tilts the mug toward us.
    qualities = {}
    for side in (-1, 1):
        horizontal = 0.85 * side * plane_normals[:, :2] + 0.32 * handle_axes[:, :2]
        horizontal /= np.maximum(np.linalg.norm(horizontal, axis=1, keepdims=True), 1e-10)
        directions = np.column_stack([cosine * horizontal, np.full(count, sine)])
        alignment = np.abs(np.sum(directions * plane_normals, axis=1))
        qualities[side] = float(np.quantile(alignment, 0.10) + alignment.mean())
    side = max(qualities, key=qualities.get)
    initial_side = side
    initial_yaw = candidate_yaw(0, side)
    if initial_yaw is None:
        initial_yaw = np.arctan2(-0.85, 0.32)
    yaw = initial_yaw
    target = desired_targets[0].copy()
    position_tau, yaw_tau, maximum_lag = 0.10, 0.20, 0.035
    position_alpha = 1.0 - np.exp(-1.0 / (fps * position_tau))
    yaw_alpha = 1.0 - np.exp(-1.0 / (fps * yaw_tau))
    maximum_yaw_step = np.deg2rad(90.0) / fps
    frozen = horizontal_norm[0] < 0.15
    freeze_count, side_changes = 0, 0
    directions, targets, yaws = [], [], []
    for frame in range(count):
        if horizontal_norm[frame] < 0.15:
            frozen = True
        elif frozen and horizontal_norm[frame] > 0.30:
            # The projected normal can reverse through a vertical singularity.
            # Choose its equivalent viewing side closest to the previous camera.
            candidates = {sign: candidate_yaw(frame, sign) for sign in (-1, 1)}
            next_side = min(
                candidates,
                key=lambda sign: abs(wrapped_angle(candidates[sign] - yaw)),
            )
            side_changes += int(next_side != side)
            side = next_side
            frozen = False
        if frozen:
            freeze_count += 1
        else:
            desired_yaw = candidate_yaw(frame, side)
            if desired_yaw is not None:
                change = yaw_alpha * wrapped_angle(desired_yaw - yaw)
                yaw += np.clip(change, -maximum_yaw_step, maximum_yaw_step)
        if frame:
            target += position_alpha * (desired_targets[frame] - target)
            lag = target - desired_targets[frame]
            lag_length = np.linalg.norm(lag)
            if lag_length > maximum_lag:
                target = desired_targets[frame] + lag * (maximum_lag / lag_length)
        directions.append([cosine * np.cos(yaw), cosine * np.sin(yaw), sine])
        targets.append(target.copy())
        yaws.append(yaw)
    directions, targets, yaws = np.asarray(directions), np.asarray(targets), np.asarray(yaws)
    frames = [_camera_spec(target, direction, 0.36, aspect) for target, direction in zip(targets, directions)]
    right = np.cross(np.broadcast_to([0.0, 0.0, 1.0], directions.shape), directions)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    up = np.cross(directions, right)
    residual = centers_world - targets
    screen_motion = np.column_stack([np.sum(residual * right, axis=1), np.sum(residual * up, axis=1)])
    tracking = {
        "method": "World-space translation follow and yaw-only handle tracking; no object or hand vertex changes",
        "trajectory": "detail_frames contains the shared per-frame camera trajectory for both columns",
        "world_up": [0.0, 0.0, 1.0],
        "fixed_elevation_degrees": float(np.rad2deg(elevation)),
        "playback_fps": float(fps),
        "position_smoothing_time_constant_seconds": position_tau,
        "position_lag_limit_m": maximum_lag,
        "position_lag_max_m": float(np.linalg.norm(targets - desired_targets, axis=1).max()),
        "position_lag_mean_m": float(np.linalg.norm(targets - desired_targets, axis=1).mean()),
        "yaw_smoothing_time_constant_seconds": yaw_tau,
        "yaw_speed_limit_degrees_per_second": 90.0,
        "yaw_max_step_degrees": float(np.rad2deg(np.abs(np.diff(yaws))).max(initial=0.0)),
        "vertical_normal_freeze_enter_horizontal_norm": 0.15,
        "vertical_normal_freeze_exit_horizontal_norm": 0.30,
        "vertical_normal_frozen_frames": freeze_count,
        "initial_handle_plane_side": initial_side,
        "handle_plane_side_changes_after_vertical_degeneracy": side_changes,
        "canonical_target_offset_m": [0.012, 0.0, 0.0],
        "mug_center_screen_span_m": np.ptp(screen_motion, axis=0).tolist(),
        "object_pitch_roll": "Unmodified source world motion; camera never copies object pitch or roll",
    }
    return frames, tracking


def build_camera_specs(cache: Mapping[str, np.ndarray], kind: str, aspect: float = 960 / 464) -> dict:
    """Fit shared cameras to both representations in the complete clip [m].

    The overview preserves world motion and includes every cached hand vertex
    and the transformed object bounding box in every frame, with 10% padding.
    GRAB detail follows the world-space mug with a world-up camera. GraspXL
    detail stabilizes the mug. Both use a crop whose label must explain that
    hands may leave the detail view.
    """
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("Camera aspect ratio must be finite and positive")
    direction = np.array([0.65, -1.0, 0.65], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    screen_right = np.cross([0.0, 0.0, 1.0], direction)
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(direction, screen_right)
    basis = np.stack([screen_right, screen_up, direction], axis=1)
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)

    def include(vertices):
        nonlocal low, high
        vertices = np.asarray(vertices)
        if vertices.shape[-1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("Camera fitting requires finite 3D vertices")
        projected = vertices.reshape(-1, 3) @ basis
        low = np.minimum(low, projected.min(axis=0))
        high = np.maximum(high, projected.max(axis=0))

    original = np.asarray(cache["original_vertices"])
    soma = np.asarray(cache["soma_vertices"])
    rotation = np.asarray(cache["object_rotation"])
    translation = np.asarray(cache["object_translation"])
    canonical = np.asarray(cache["object_vertices_canonical"])
    frames = len(original)
    if frames == 0 or len(soma) != frames:
        raise ValueError("Both compared meshes need the same nonzero frame count")
    if rotation.shape != (frames, 3, 3) or translation.shape != (frames, 3):
        raise ValueError("Object transforms must correspond to every mesh frame")
    if canonical.ndim != 2 or canonical.shape[1] != 3 or not np.isfinite(canonical).all():
        raise ValueError("Canonical object vertices must be finite with shape (V, 3)")
    minimum, maximum = canonical.min(axis=0), canonical.max(axis=0)
    corners = np.array(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
    )
    for start in range(0, frames, 64):
        selected = slice(start, start + 64)
        include(original[selected])
        include(soma[selected])
        object_corners = np.matmul(corners, rotation[selected].transpose(0, 2, 1))
        include(object_corners + translation[selected, None, :])
    extent = high - low
    height = max(extent[1], extent[0] / aspect, 0.15) * 1.10
    target = basis @ ((low + high) / 2.0)
    overview = _camera_spec(target, direction, height, aspect, extent[2])
    overview["coordinate_frame"] = "world"
    overview["framing"] = "All original and SOMA hand vertices and object bounds, all cached frames; 10% padding"
    detail_target = (minimum + maximum) / 2.0 @ canonical_to_z_up(kind).T
    detail = _camera_spec(detail_target, [0.32, -0.85, 0.45], 0.36, aspect)
    detail["coordinate_frame"] = "object_stabilized_z_up"
    detail["framing"] = "Fixed grip close-up; hands may leave this view; consult full-hand overview"
    result = {
        "overview": overview,
        "detail": detail,
        "canonical_to_z_up": canonical_to_z_up(kind).tolist(),
    }
    if kind == "GRAB":
        fps = float(np.asarray(cache.get("playback_fps", 30.0)).item())
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Camera tracking requires a positive playback frame rate")
        detail_frames, tracking = _grab_detail_specs(rotation, translation, minimum, maximum, fps, aspect)
        result["detail"] = dict(detail_frames[0])
        result["detail"]["coordinate_frame"] = "world_follow_handle"
        result["detail"]["framing"] = (
            "Camera follows mug translation and handle yaw; source tilt is retained; hands may leave the close-up"
        )
        result["detail_frames"] = detail_frames
        result["detail_tracking"] = tracking
    return result
