# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare subject-separated GRAB references and a compound-collision mug.

Run with the ``graspxl`` environment. Original datasets and previous preparation
are read only. Distances are metres and angles are radians throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

SUBJECT_SPLITS = {
    "train": ["s1", "s2", "s3", "s4", "s6", "s7"],
    "validation": ["s5", "s8"],
    "test": ["s9", "s10"],
}
FINGERS = ("index", "middle", "pinky", "ring", "thumb")
FRAME_NAMES = ["right_wrist_0rz"] + [
    name for finger in FINGERS for name in [f"right_{finger}{j}_x" for j in (1, 2, 3)] + [f"right_{finger}_tip"]
]
MANO_JOINT_ORDER = [0, 1, 2, 3, 16, 4, 5, 6, 17, 7, 8, 9, 18, 10, 11, 12, 19, 13, 14, 15, 20]


def sha256(path: Path) -> str:
    """Return the SHA-256 of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Write an indented JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def decode_rot6d(value: np.ndarray) -> np.ndarray:
    """Decode upstream interleaved first-two-column rotation vectors."""
    columns = value.reshape(*value.shape[:-1], 3, 2)
    first = columns[..., 0]
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = columns[..., 1] - np.sum(first * columns[..., 1], axis=-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def generic_hand_joints(hand_urdf: Path, qpos: np.ndarray) -> np.ndarray:
    """Compute the simulation hand's 21 frame positions [m] from 51 coordinates."""
    root = ET.parse(hand_urdf).getroot()
    count = len(qpos)
    transforms = {"world": np.broadcast_to(np.eye(4), (count, 4, 4)).copy()}
    frames = {}
    dof_index = 0
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        offset = np.eye(4)
        if origin is not None:
            offset[:3, 3] = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            angles = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            offset[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
        world = transforms[parent] @ offset
        frames[joint.attrib["name"]] = world[:, :3, 3].copy()
        if joint.attrib["type"] != "fixed":
            axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
            motion = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            values = qpos[:, dof_index, None]
            if joint.attrib["type"] == "prismatic":
                motion[:, :3, 3] = values * axis
            else:
                motion[:, :3, :3] = Rotation.from_rotvec(values * axis).as_matrix()
            world = world @ motion
            dof_index += 1
        transforms[child] = world
    if dof_index != 51:
        raise ValueError(f"Expected 51 simulation hand coordinates, got {dof_index}")
    return np.stack([frames[name] for name in FRAME_NAMES], axis=1)


def table_alignment(table: dict, table_vertices: np.ndarray, object_translation: np.ndarray) -> tuple:
    """Map the robust source tabletop to horizontal z=0.5 m and mug xy=(1,0) m."""
    params = table["params"]
    # GRAB applies source Rodrigues matrices to row vectors, including the table.
    rotations = Rotation.from_rotvec(params["global_orient"]).as_matrix().transpose(0, 2, 1)
    normals = rotations[:, :, 1].copy()  # The thin canonical table axis is local Y.
    normals[normals[:, 2] < 0] *= -1
    # The first few capture frames can contain gross table rotation transients.
    stable = slice(min(120, len(normals) // 4), None)
    normal = np.median(normals[stable], axis=0)
    normal /= np.linalg.norm(normal)
    center = np.median(params["transl"][stable], axis=0)
    half_thickness = float(np.max(np.abs(table_vertices[:, 1])))
    top_point = center + normal * half_thickness
    axis = np.cross(normal, np.asarray([0.0, 0.0, 1.0]))
    sine = np.linalg.norm(axis)
    angle = np.arctan2(sine, normal[2])
    matrix = Rotation.from_rotvec(axis * angle / sine).as_matrix() if sine > 1e-12 else np.eye(3)
    initial = matrix @ object_translation[0]
    offset = np.asarray([1.0 - initial[0], -initial[1], 0.5 - (matrix @ top_point)[2]])
    angles = np.degrees(np.arccos(np.clip(normals @ normal, -1, 1)))
    report = {
        "source_table_normal": normal.tolist(),
        "source_table_top_point_m": top_point.tolist(),
        "world_to_training_rotation": matrix.tolist(),
        "world_to_training_translation_m": offset.tolist(),
        "source_table_normal_deviation_median_deg": float(np.median(angles)),
        "source_table_normal_deviation_max_deg": float(angles.max()),
        "table_top_m": 0.5,
        "table_geometry_change": (
            "Simulator uses a larger 2 m by 1 m static tabletop; original is about 0.45 m by 0.54 m."
        ),
    }
    return matrix, offset, report


def contiguous(mask: np.ndarray) -> list:
    """Return half-open runs of true values."""
    transitions = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)))


def make_episodes(right: np.ndarray, left: np.ndarray, bottom: np.ndarray, fps: float) -> tuple:
    """Select right-only grasp phases with prior reach; reject unsupported airborne starts."""
    eligible = right & ~left
    # Bridge brief contact-label dropout, but never bridge a left-hand contact.
    for start, end in contiguous(~eligible):
        if start > 0 and end < len(eligible) and end - start <= 3 and not left[start:end].any():
            eligible[start:end] = True
    episodes, excluded = [], []
    for contact_start, end in contiguous(eligible):
        start = max(0, contact_start - int(fps))
        blocked = np.flatnonzero(left[start:contact_start])
        if len(blocked):
            start += int(blocked[-1]) + 1
        record = {"start": int(start), "end": int(end), "contact_start": int(contact_start)}
        if end - contact_start < 15:
            excluded.append(dict(record, reason="right_only_contact_shorter_than_0.5_seconds"))
        elif bottom[start] > 0.52:
            excluded.append(
                dict(record, reason="object_already_airborne_at_episode_start", initial_bottom_m=float(bottom[start]))
            )
        else:
            episodes.append(record)
    return episodes, excluded


def convex_contains(points: np.ndarray, hulls: list) -> np.ndarray:
    """Classify points inside the union of convex collision meshes."""
    occupied = np.zeros(len(points), dtype=bool)
    for mesh in hulls:
        equations = ConvexHull(mesh.vertices).equations
        candidates = np.flatnonzero(~occupied & (points >= mesh.bounds[0]).all(1) & (points <= mesh.bounds[1]).all(1))
        for start in range(0, len(candidates), 1024):
            ids = candidates[start : start + 1024]
            occupied[ids] = (points[ids] @ equations[:, :3].T + equations[:, 3] <= 1e-8).all(1)
    return occupied


def native_contains(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Classify exact native-mesh occupancy with bounded ray-intersection memory."""
    return np.concatenate([mesh.contains(points[start : start + 64]) for start in range(0, len(points), 64)])


def prepare_object(mesh: trimesh.Trimesh, output: Path, force: bool = False) -> dict:
    """Build dynamic compound convex collision geometry preserving mug openings."""
    import coacd

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "collision_report.json"
    geometry_hash = hashlib.sha256(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest()
    if manifest_path.exists() and not force:
        report = json.loads(manifest_path.read_text())
        if report.get("source_geometry_sha256") == geometry_hash and Path(report["object_urdf"]).exists():
            if sha256(Path(report["object_urdf"])) != report["urdf_sha256"]:
                raise ValueError("Cached mug URDF hash mismatch")
            cached = json.loads((output / "decomposition.json").read_text())
            for record in cached["parts"]:
                if sha256(output / record["filename"]) != record["sha256"]:
                    raise ValueError("Cached collision part hash mismatch")
            return report
    visual = output / "mug.obj"
    mesh.export(visual)
    coacd.set_log_level("warn")
    partial_path = output / "decomposition.json"
    partial = json.loads(partial_path.read_text()) if partial_path.exists() else {}
    if not force and partial.get("source_geometry_sha256") == geometry_hash:
        for record in partial["parts"]:
            if sha256(output / record["filename"]) != record["sha256"]:
                raise ValueError("Cached collision decomposition hash mismatch")
        hulls = [trimesh.load(output / record["filename"], process=False) for record in partial["parts"]]
    else:
        parts = coacd.run_coacd(
            coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)),
            threshold=0.01,
            max_convex_hull=128,
            preprocess_mode="off",
            resolution=3000,
            mcts_nodes=20,
            mcts_iterations=150,
            mcts_max_depth=3,
            merge=True,
            seed=20260917,
        )
        hulls = [trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in parts]
        records = []
        for index, hull in enumerate(hulls):
            path = output / f"collision_{index:03d}.obj"
            hull.export(path)
            records.append({"filename": path.name, "sha256": sha256(path)})
        write_json(partial_path, {"source_geometry_sha256": geometry_hash, "parts": records})
    mass = 0.25
    inertia = mesh.moment_inertia * mass / mesh.volume
    robot = ET.Element("robot", name="grab_mug")
    link = ET.SubElement(robot, "link", name="mug")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(map(str, mesh.center_mass)), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=str(mass))
    ET.SubElement(
        inertial,
        "inertia",
        **{
            name: str(inertia[i, j])
            for name, i, j in [("ixx", 0, 0), ("iyy", 1, 1), ("izz", 2, 2), ("ixy", 0, 1), ("ixz", 0, 2), ("iyz", 1, 2)]
        },
    )
    visual_node = ET.SubElement(link, "visual")
    ET.SubElement(ET.SubElement(visual_node, "geometry"), "mesh", filename=visual.name, scale="1 1 1")
    material = ET.SubElement(visual_node, "material", name="mug_blue")
    ET.SubElement(material, "color", rgba="0.2 0.5 0.8 1")
    for index in range(len(hulls)):
        collision = ET.SubElement(link, "collision", name=f"convex_{index:03d}")
        ET.SubElement(
            ET.SubElement(collision, "geometry"),
            "mesh",
            filename=f"collision_{index:03d}.obj",
            scale="1 1 1",
        )
        material = ET.SubElement(collision, "material", name="")
        ET.SubElement(material, "contact", name="mug")
    urdf = output / "mug.urdf"
    ET.ElementTree(robot).write(str(urdf), encoding="utf-8", xml_declaration=True)
    rng = np.random.default_rng(20260917)
    probes = rng.uniform(mesh.bounds[0], mesh.bounds[1], size=(12000, 3))
    native_inside = native_contains(mesh, probes)
    collision_inside = convex_contains(probes, hulls)
    # Probe empty cross-sections through the body and handle as well as random volume.
    grid_x, grid_z = np.meshgrid(np.linspace(*mesh.bounds[:, 0], 121), np.linspace(*mesh.bounds[:, 2], 111))
    section = np.column_stack((grid_x.ravel(), np.zeros(grid_x.size), grid_z.ravel()))
    section_native = native_contains(mesh, section)
    section_collision = convex_contains(section, hulls)
    special_points = np.asarray([[-0.015, 0, 0], [-0.015, 0, 0.04], [0.04, 0, 0], [0.05, 0, 0]])
    special_expected = np.asarray([False, False, False, True])
    special_native = native_contains(mesh, special_points)
    special_collision = convex_contains(special_points, hulls)
    if not np.array_equal(special_native, special_expected) or not np.array_equal(special_collision, special_expected):
        raise ValueError("Collision approximation closes a mug opening or removes the outer handle")
    np.savez_compressed(
        output / "collision_probe_audit.npz",
        points=probes,
        native_inside=native_inside,
        collision_inside=collision_inside,
        section_points=section,
        section_native_inside=section_native,
        section_collision_inside=section_collision,
        opening_points=special_points,
        opening_native_inside=special_native,
        opening_collision_inside=special_collision,
    )
    report = {
        "object_urdf": str(urdf.resolve()),
        "visual_mesh": str(visual.resolve()),
        "source_vertices": len(mesh.vertices),
        "source_faces": len(mesh.faces),
        "source_watertight": bool(mesh.is_watertight),
        "source_geometry_sha256": geometry_hash,
        "source_mesh_volume_m3": float(mesh.volume),
        "collision_algorithm": "CoACD 1.0.5 compound convex decomposition, not one convex hull",
        "normalized_concavity_threshold": 0.01,
        "maximum_parts": 128,
        "convex_parts": len(hulls),
        "mass_kg": mass,
        "mass_policy": "Experiment parameter, not measured GRAB mug mass",
        "random_probe_count": len(probes),
        "opening_probes_passed": True,
        "opening_probe_names": ["cup_cavity_center", "cup_cavity_near_rim", "handle_opening", "solid_outer_handle"],
        "native_void_false_positive_fraction": float(
            (collision_inside & ~native_inside).sum() / (~native_inside).sum()
        ),
        "native_material_false_negative_fraction": float(
            (~collision_inside & native_inside).sum() / native_inside.sum()
        ),
        "central_section_native_void_preserved_fraction": float(
            (~section_collision & ~section_native).sum() / (~section_native).sum()
        ),
        "geometry_limit": (
            "Convex decomposition is approximate; inspect collision_probe_audit.npz and contact behavior."
        ),
        "urdf_sha256": sha256(urdf),
    }
    write_json(manifest_path, report)
    return report


def main() -> None:
    """Prepare reusable reference arrays, subject splits, and collision geometry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir", type=Path, default=Path(__file__).resolve().parents[1] / "grab_mug_refinement/prepared"
    )
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).with_name("prepared"))
    parser.add_argument("--projects_root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--refined_dir", type=Path)
    parser.add_argument(
        "--collision_assets", type=Path, help="Reuse a previously verified compound mug asset directory."
    )
    parser.add_argument("--skip_collision", action="store_true")
    parser.add_argument("--force_collision", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = args.source_dir.resolve()
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    if source_manifest["sequences"] != 44:
        raise ValueError("Expected all 44 original GRAB mug recordings")
    splits = {
        name: {
            "subjects": subjects,
            "clip_ids": [c["clip_id"] for c in source_manifest["clips"] if c["subject"] in subjects],
        }
        for name, subjects in SUBJECT_SPLITS.items()
    }
    write_json(output / "splits.json", splits)
    with zipfile.ZipFile(args.projects_root / "Grab Dataset/tools__object_meshes__contact_meshes.zip") as archive:
        mug = trimesh.load(io.BytesIO(archive.read("contact_meshes/mug.ply")), file_type="ply", process=False)
        table_mesh = trimesh.load(io.BytesIO(archive.read("contact_meshes/table.ply")), file_type="ply", process=False)
    hand_urdf = args.projects_root / "GraspXL/rsc/mano_double/rhand_mano_low_mass.urdf"
    records = []
    for clip in source_manifest["clips"]:
        clip_id = clip["clip_id"]
        data = dict(np.load(source_root / clip["clip_path"], allow_pickle=False))
        reference_path = source_root / clip["clip_path"]
        if args.refined_dir:
            reference_path = args.refined_dir / f"{clip_id}.npz"
            refined = np.load(reference_path, allow_pickle=False)
            for key in ("x_rhand", "rhand_joints", "object_rotation", "object_translation", "source_frame_indices"):
                if key not in refined:
                    raise KeyError(f"Refined reference {reference_path} missing {key}")
            if not np.array_equal(refined["source_frame_indices"], data["source_frame_indices"]):
                raise ValueError(f"Refined reference changed timeline: {clip_id}")
            if not np.allclose(refined["object_translation"], data["object_translation"], atol=1e-7):
                raise ValueError(f"Refined reference changed mug trajectory: {clip_id}")
            if not np.allclose(refined["object_rotation"], data["object_rotation"], atol=1e-7):
                raise ValueError(f"Refined reference changed mug orientation: {clip_id}")
            if refined["x_rhand"].shape != data["x_rhand"].shape:
                raise ValueError(f"Refined reference changed the hand representation: {clip_id}")
            if refined["rhand_joints"].shape != data["rhand_joints"].shape:
                raise ValueError(f"Refined reference changed the hand joint ordering: {clip_id}")
            data["x_rhand"] = refined["x_rhand"]
            data["rhand_joints"] = refined["rhand_joints"]
        with zipfile.ZipFile(clip["source_archive"]) as archive:
            native = dict(np.load(io.BytesIO(archive.read(clip["source_member"])), allow_pickle=True))
        frame_ids = data["source_frame_indices"]
        nframes = len(frame_ids)
        alignment, offset, alignment_report = table_alignment(
            native["table"].item(), table_mesh.vertices, data["object_translation"]
        )
        object_rotations = alignment @ data["object_rotation"]
        object_translations = data["object_translation"] @ alignment.T + offset
        quaternions = Rotation.from_matrix(object_rotations).as_quat()[:, [3, 0, 1, 2]]
        # A continuous sign is convenient for downstream interpolation and differences.
        for frame in range(1, nframes):
            if np.dot(quaternions[frame], quaternions[frame - 1]) < 0:
                quaternions[frame] *= -1
        rotations = decode_rot6d(data["x_rhand"][:, 3:].reshape(nframes, 16, 6))
        rotations[:, 0] = alignment @ rotations[:, 0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Gimbal lock detected")
            euler = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_euler("XYZ").reshape(nframes, 16, 3)
        euler = np.unwrap(euler, axis=0)
        recorded_joints = data["rhand_joints"][:, MANO_JOINT_ORDER] @ alignment.T + offset
        qpos = np.column_stack((recorded_joints[:, 0], euler.reshape(nframes, 48)))
        simulated_joints = generic_hand_joints(hand_urdf, qpos)
        retarget_error = np.linalg.norm(simulated_joints - recorded_joints, axis=-1)
        # Minimum over native mug vertices, in batches to avoid large intermediate arrays.
        bottom = np.concatenate(
            [
                (
                    np.einsum("tij,vj->tvi", object_rotations[start : start + 32], mug.vertices)
                    + object_translations[start : start + 32, None]
                )[:, :, 2].min(1)
                for start in range(0, nframes, 32)
            ]
        )
        right = data["rhand_source_contact"].any(1)
        left = data["lhand_source_contact"].any(1)
        episodes, excluded = make_episodes(right, left, bottom, float(data["playback_fps"]))
        split = next(name for name, subjects in SUBJECT_SPLITS.items() if clip["subject"] in subjects)
        target = output / "clips" / f"{clip_id}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "hand_qpos": qpos.astype(np.float32),
            "object_pose": np.column_stack((object_translations, quaternions)).astype(np.float32),
            "joints": simulated_joints.astype(np.float32),
            "recorded_joints": recorded_joints.astype(np.float32),
            "contact": right.astype(np.float32),
            "left_contact": left,
            "source_frame_indices": frame_ids,
            "fps": np.asarray(30.0, dtype=np.float32),
            "source_fps": np.asarray(120.0, dtype=np.float32),
            "object_bottom": bottom.astype(np.float32),
            "world_to_training_rotation": alignment,
            "world_to_training_translation": offset,
            "source_table_translation": native["table"].item()["params"]["transl"][frame_ids],
            "source_table_global_orient": native["table"].item()["params"]["global_orient"][frame_ids],
        }
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"Nonfinite reference {clip_id}")
        np.savez_compressed(target, **arrays)
        record = {
            "clip_id": clip_id,
            "subject": clip["subject"],
            "split": split,
            "action": clip["action"],
            "path": str(target.relative_to(output)),
            "nframes": nframes,
            "fps": 30,
            "episodes": episodes,
            "excluded_episodes": excluded,
            "right_contact_frames": int(right.sum()),
            "left_contact_frames": int(left.sum()),
            "simultaneous_hand_contact_frames": int((right & left).sum()),
            "rl_supported_frames": int(
                np.unique(np.concatenate([np.arange(x["start"], x["end"]) for x in episodes])).size
            )
            if episodes
            else 0,
            "unsupported_reason": None
            if episodes
            else "No right-only grasp episode starting with mug on table; retained for refiner and reporting.",
            "retarget_joint_difference_mean_m": float(retarget_error.mean()),
            "retarget_joint_difference_max_m": float(retarget_error.max()),
            "table_alignment": alignment_report,
            "source_reference": str(reference_path.resolve()),
            "source_reference_sha256": sha256(reference_path),
            "sha256": sha256(target),
            "source_archive": clip["source_archive"],
            "source_member": clip["source_member"],
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "prepared": clip_id,
                    "split": split,
                    "episodes": len(episodes),
                    "retarget_mean_mm": float(retarget_error.mean() * 1000),
                }
            ),
            flush=True,
        )
    collision_dir = args.collision_assets.resolve() if args.collision_assets else output / "assets/mug"
    report = {
        "schema_version": 1,
        "splits": splits,
        "clips": records,
        "sequences": len(records),
        "frames": sum(c["nframes"] for c in records),
        "fps": 30,
        "object_urdf": str((collision_dir / "mug.urdf").resolve()),
        "hand_urdf": str(hand_urdf.resolve()),
        "hand_urdf_sha256": sha256(hand_urdf),
        "reference_kind": "validation_selected_text2hoi_refinement" if args.refined_dir else "original_grab",
        "source_manifest": str((source_root / "manifest.json").resolve()),
        "source_manifest_sha256": sha256(source_root / "manifest.json"),
        "coordinate_convention": (
            "World wrist joint xyz + intrinsic XYZ Euler16; object xyz+wxyz. Metres/radians. Table top z=0.5."
        ),
        "retarget_policy": (
            "Same MANO local rotations on generic GraspXL hand; reference joints are generic URDF FK. "
            "Personalized recorded joints retained separately; no shape fitting or IK contact correction."
        ),
        "episode_policy": (
            "Right-only source contact, bridging <=3-frame label gaps; up to 1 s prior reach without left contact; "
            ">=0.5 s contact; start mug bottom <=table+0.02 m. Half-open start/end."
        ),
        "scene_limit": (
            "Only hand+mug+table, no body, head, other hand or recipient; "
            "retained phases of drink/pass clips do not simulate mouth or recipient contact."
        ),
        "evaluation_limit": (
            "Subjects split before windows. Existing all44 fine-tuned checkpoints excluded; released pretrained "
            "Text2HOI GRAB checkpoint exposure is unknown, so this is adaptation holdout, "
            "not proof of unseen pretraining data."
        ),
    }
    write_json(output / "manifest.json", report)
    if not args.skip_collision:
        report["collision"] = prepare_object(mug, collision_dir, args.force_collision)
        write_json(output / "manifest.json", report)
    print(
        json.dumps(
            {
                "prepared": len(records),
                "rl_supported_clips": sum(bool(c["episodes"]) for c in records),
                "manifest": str(output / "manifest.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
