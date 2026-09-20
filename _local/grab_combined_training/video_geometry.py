# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render geometry for the native segmented GraspXL simulation hand.

These are the URDF visual meshes, not a personalized MANO skin. The input is
the simulator's world-coordinate hand state, with translation in metres and
all rotation coordinates in radians. No licensed MANO model is loaded here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

DEFAULT_HAND_URDF = Path("/home/nmt/Projects/GraspXL/rsc/mano_double/rhand_mano_low_mass.urdf")


def _origin(element: ET.Element | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        angles = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    return result


class HandVisuals:
    """Preload visual triangles and forward kinematics for the 51-DoF right hand.

    Attributes:
        faces: Constant triangle vertex indices, shape ``(F, 3)``, int32.
        visual_link_names: Link names corresponding to the loaded mesh segments.
        vertex_count: Total number of combined visual vertices.
    """

    def __init__(self, urdf: str | Path = DEFAULT_HAND_URDF) -> None:
        self.urdf = Path(urdf).resolve()
        root = ET.parse(self.urdf).getroot()
        self._joints = []
        dof = 0
        children = set()
        for joint in root.findall("joint"):
            kind = joint.attrib["type"]
            if kind not in ("fixed", "revolute", "continuous", "prismatic"):
                raise ValueError(f"Unsupported URDF joint type: {kind}")
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            children.add(child)
            axis_element = joint.find("axis")
            axis = np.fromstring(axis_element.get("xyz", "1 0 0"), sep=" ") if axis_element is not None else None
            coordinate = None
            if kind != "fixed":
                coordinate = dof
                dof += 1
                if axis is None or not np.isclose(np.linalg.norm(axis), 1.0):
                    raise ValueError(f"Joint {joint.attrib['name']} needs a unit axis")
            self._joints.append(
                (joint.attrib["name"], parent, child, _origin(joint.find("origin")), kind, axis, coordinate)
            )
        if dof != 51:
            raise ValueError(f"Expected 51 hand coordinates, found {dof}")
        self._root_links = [link.attrib["name"] for link in root.findall("link") if link.attrib["name"] not in children]

        self._visuals = []
        self.visual_link_names = []
        all_faces = []
        offset = 0
        for link in root.findall("link"):
            for visual in link.findall("visual"):
                mesh_element = visual.find("geometry/mesh")
                if mesh_element is None:
                    raise ValueError(f"Expected a mesh visual for {link.attrib['name']}")
                filename = mesh_element.attrib["filename"]
                if filename.startswith("package://"):
                    raise ValueError(f"Package URIs are unsupported for this local hand: {filename}")
                mesh = trimesh.load(self.urdf.parent / filename, force="mesh", process=True)
                scale = np.fromstring(mesh_element.get("scale", "1 1 1"), sep=" ")
                if scale.shape != (3,) or np.any(scale <= 0):
                    raise ValueError(f"Expected three positive visual scale values: {scale}")
                vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
                # Inverse-transpose scaling preserves normals for nonuniform mesh scale.
                normals = np.asarray(mesh.vertex_normals, dtype=np.float64) / scale
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
                visual_origin = _origin(visual.find("origin"))
                vertices = vertices @ visual_origin[:3, :3].T + visual_origin[:3, 3]
                normals = normals @ visual_origin[:3, :3].T
                self._visuals.append((link.attrib["name"], slice(offset, offset + len(vertices)), vertices, normals))
                self.visual_link_names.append(link.attrib["name"])
                all_faces.append(np.asarray(mesh.faces, dtype=np.int32) + offset)
                offset += len(vertices)
        if not all_faces:
            raise ValueError(f"No visual meshes found in {self.urdf}")
        self.vertex_count = offset
        self.faces = np.ascontiguousarray(np.concatenate(all_faces), dtype=np.int32)

    def _forward(self, qpos: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        coordinates = np.asarray(qpos, dtype=np.float64)
        if coordinates.shape != (51,) or not np.isfinite(coordinates).all():
            raise ValueError("Expected 51 finite world-coordinate hand values")
        transforms = {name: np.eye(4) for name in self._root_links}
        frames = {}
        for name, parent, child, offset, kind, axis, coordinate in self._joints:
            world = transforms[parent] @ offset
            frames[name] = world[:3, 3].copy()
            if coordinate is not None:
                motion = np.eye(4)
                if kind == "prismatic":
                    motion[:3, 3] = coordinates[coordinate] * axis
                else:
                    motion[:3, :3] = Rotation.from_rotvec(coordinates[coordinate] * axis).as_matrix()
                world = world @ motion
            transforms[child] = world
        return transforms, frames

    def link_transforms(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        """Return local-link to world transforms, with translation components [m].

        Args:
            qpos: World wrist translation [m], then root and finger angles [rad], shape ``(51,)``.
        """
        return self._forward(qpos)[0]

    def joint_frames(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        """Return named joint-origin world positions [m] before each joint's motion.

        Args:
            qpos: World wrist translation [m], then root and finger angles [rad], shape ``(51,)``.
        """
        return self._forward(qpos)[1]

    def vertices(self, qpos: np.ndarray) -> np.ndarray:
        """Return combined world visual vertices [m], shape ``(V, 3)``, float32.

        Args:
            qpos: World wrist translation [m], then root and finger angles [rad], shape ``(51,)``.
        """
        transforms = self.link_transforms(qpos)
        result = np.empty((self.vertex_count, 3), dtype=np.float32)
        for link, ids, vertices, _ in self._visuals:
            transform = transforms[link]
            result[ids] = vertices @ transform[:3, :3].T + transform[:3, 3]
        return result

    def vertices_and_normals(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return world vertices [m] and unit vertex normals, each float32 ``(V, 3)``.

        Args:
            qpos: World wrist translation [m], then root and finger angles [rad], shape ``(51,)``.
        """
        transforms = self.link_transforms(qpos)
        result = np.empty((self.vertex_count, 3), dtype=np.float32)
        result_normals = np.empty_like(result)
        for link, ids, vertices, normals in self._visuals:
            transform = transforms[link]
            result[ids] = vertices @ transform[:3, :3].T + transform[:3, 3]
            result_normals[ids] = normals @ transform[:3, :3].T
        return result, result_normals
