# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Link existing licensed/downloaded assets and prepare a GRAB mug in WSL."""

import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh


def link_existing(source: Path, target: Path):
    """Create a link without replacing an existing unrelated file."""
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise FileExistsError(target)
        return
    target.symlink_to(source, target_is_directory=source.is_dir())


def main():
    root = Path(__file__).resolve().parents[3]
    repo = root / "Text2HOI"
    downloads = root / "Text2HOI_Download"
    local = Path(__file__).resolve().parent
    link_existing(downloads / "Checkpoints/grab", repo / "checkpoints/grab")
    for name in ("balance_weights.pkl", "data.npz", "obj.pkl", "text.json", "text_count.json", "text_length.json"):
        link_existing(downloads / "Preprocessing/grab" / name, repo / "data/grab" / name)
    link_existing(root / "SOMA-X/assets/MANO", repo / "data/mano/mano_v1_2/models")

    archive = root / "Grab Dataset/tools__object_meshes__contact_meshes.zip"
    original = repo / "data/grab/contact_meshes/mug.ply"
    original.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        payload = source.read("contact_meshes/mug.ply")
    if original.exists() and original.read_bytes() != payload:
        raise FileExistsError("Existing mug differs from the licensed source archive.")
    original.write_bytes(payload)

    processed = repo / "data/grab/processed_object_meshes/mug.ply"
    processed.parent.mkdir(parents=True, exist_ok=True)
    if not processed.exists():
        meshes = pymeshlab.MeshSet(verbose=0)
        meshes.load_new_mesh(str(original))
        target_vertices = 4000
        target_faces = 100 + 2 * target_vertices
        for _ in range(100):
            if meshes.current_mesh().vertex_number() <= target_vertices:
                break
            meshes.apply_filter(
                "simplification_quadric_edge_collapse_decimation",
                targetfacenum=target_faces,
                preservenormal=True,
            )
            target_faces -= meshes.current_mesh().vertex_number() - target_vertices
        if meshes.current_mesh().vertex_number() > target_vertices:
            raise RuntimeError("Mesh simplification failed to converge.")
        meshes.save_current_mesh(str(processed))

    # The pickle is the user's download from the official Text2HOI asset folder.
    with (repo / "data/grab/obj.pkl").open("rb") as stream:
        objects = pickle.load(stream)
    mesh = trimesh.load(processed, maintain_order=True, process=False)
    points = objects["obj_pcs"]["mug"]
    indices = objects["point_sets"]["mug"]
    normals = objects["obj_pc_normals"]["mug"]
    assert points.shape == (1024, 3)
    assert np.isfinite(points).all() and np.isfinite(normals).all()
    assert np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5)
    index_error = np.linalg.norm(mesh.vertices[indices] - points, axis=1)
    _, surface_distance, _ = trimesh.proximity.closest_point(mesh, points)
    report = {
        "source_archive": str(archive),
        "source_member": "contact_meshes/mug.ply",
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "processed_sha256": hashlib.sha256(processed.read_bytes()).hexdigest(),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "point_count": len(points),
        "point_index_max_error_m": float(index_error.max()),
        "point_surface_max_error_m": float(surface_distance.max()),
        "point_surface_mean_error_m": float(surface_distance.mean()),
        "mesh_bounds_m": mesh.bounds.tolist(),
        "point_bounds_m": [points.min(0).tolist(), points.max(0).tolist()],
    }
    (local / "mesh_preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if surface_distance.max() > 0.001:
        raise ValueError("Cached points differ from the prepared mug surface by over 1 mm.")


if __name__ == "__main__":
    main()
