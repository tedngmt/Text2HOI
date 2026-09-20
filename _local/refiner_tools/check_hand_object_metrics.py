"""Check the signed-distance metrics against shapes with known answers."""

import io
import zipfile
from pathlib import Path

import numpy as np
import torch
import trimesh

from hand_object_metrics import ObjectSurface, closest_points_on_triangles, frame_metrics

PROJECTS = Path(__file__).resolve().parents[3]


def main():
    rng = np.random.default_rng(0)
    # 1. Sphere: the analytic signed distance is |p| - r.
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=0.05)
    surface = ObjectSurface(sphere.vertices, sphere.faces)
    points = rng.uniform(-0.075, 0.075, size=(20000, 3))
    got = surface.signed_distance(torch.as_tensor(points)).cpu().numpy()
    expected = np.linalg.norm(points, axis=1) - 0.05
    finite = np.isfinite(got)
    error = np.abs(got[finite] - expected[finite])
    print(f"sphere: {finite.sum()} near points, max error {error.max() * 1000:.4f} mm")
    assert error.max() < 2e-4, "sphere distance error above 0.2 mm (faceting alone is about 0.06 mm)"
    clear = np.abs(expected[finite]) > 2e-4
    assert ((got[finite] < 0) == (expected[finite] < 0))[clear].all(), "sphere sign"

    # 2. Box with sharp edges: exercises edge and corner regions of the closest-point code.
    box = trimesh.creation.box(extents=(0.10, 0.06, 0.04))
    box = box.subdivide().subdivide()
    surface = ObjectSurface(box.vertices, box.faces)
    points = rng.uniform(-0.07, 0.07, size=(20000, 3))
    got = surface.signed_distance(torch.as_tensor(points)).cpu().numpy()
    half = np.array([0.05, 0.03, 0.02])
    q = np.abs(points) - half
    expected = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    finite = np.isfinite(got)
    error = np.abs(got[finite] - expected[finite])
    print(f"box: {finite.sum()} near points, max error {error.max() * 1000:.5f} mm")
    assert error.max() < 1e-5, "box distance or sign error"

    # 3. Real GRAB mug: the sign must agree with the ray-casting containment test in trimesh.
    archive = zipfile.ZipFile(PROJECTS / "Grab Dataset/tools__object_meshes__contact_meshes.zip")
    mug = trimesh.load(io.BytesIO(archive.read("contact_meshes/mug.ply")), file_type="ply", process=False)
    surface = ObjectSurface(mug.vertices, mug.faces)
    low, high = mug.bounds
    points = rng.uniform(low, high, size=(600, 3))
    got = surface.signed_distance(torch.as_tensor(points)).cpu().numpy()
    # The pure-numpy ray caster in trimesh needs gigabytes for large batches on a
    # 50k-face scan (a 4,000-point batch exhausted 15 GB of RAM), so query in small batches.
    reference = np.concatenate([mug.contains(points[i : i + 50]) for i in range(0, len(points), 50)])
    clear = np.abs(got) > 5e-4  # skip points within half a millimeter of the surface
    agreement = ((got < 0) == reference)[clear].mean()
    print(
        f"mug: watertight={surface.watertight}, inside fraction {reference.mean():.3f}, "
        f"sign agreement {agreement:.5f} on {clear.sum()} points"
    )
    assert agreement > 0.999, "mug inside/outside sign disagrees with ray casting"

    # 3b. Distances must equal an exhaustive search over every triangle on real scans,
    #     including meshes that are not watertight. (trimesh.proximity.closest_point is not
    #     used as the reference: it was measured up to 0.22 mm away from exhaustive search.)
    for name in ("mug", "bowl", "camera", "pyramidsmall", "cylindersmall"):
        mesh = trimesh.load(io.BytesIO(archive.read(f"contact_meshes/{name}.ply")), file_type="ply", process=False)
        surface = ObjectSurface(mesh.vertices, mesh.faces)
        base, _ = trimesh.sample.sample_surface(mesh, 300)
        points = base + rng.normal(size=base.shape) * rng.uniform(0, 0.01, size=(len(base), 1))
        got = surface.signed_distance(torch.as_tensor(points)).abs().cpu().numpy()
        triangles = torch.as_tensor(mesh.vertices[mesh.faces], dtype=torch.float64, device="cuda")
        query = torch.as_tensor(points.astype(np.float32), dtype=torch.float64, device="cuda")
        exhaustive = []
        for begin in range(0, len(query), 20):
            part = query[begin : begin + 20]
            flat = part[:, None].expand(-1, len(triangles), -1).reshape(-1, 3)
            tri = triangles[None].expand(len(part), -1, -1, -1).reshape(-1, 3, 3)
            closest, _ = closest_points_on_triangles(flat, tri[:, 0], tri[:, 1], tri[:, 2])
            exhaustive.append((flat - closest).norm(dim=-1).reshape(len(part), -1).min(1).values)
        exhaustive = torch.cat(exhaustive).cpu().numpy()
        error = np.abs(got - exhaustive)
        print(f"{name}: watertight={surface.watertight}, max error vs exhaustive search {error.max() * 1000:.6f} mm")
        assert error.max() < 1e-6, name + " distance differs from exhaustive search"

    # 4. Moving-object transform: one vertex 3 mm inside a rotated, translated sphere,
    #    one 3 mm outside (touching), one far away.
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=0.05)
    surface = ObjectSurface(sphere.vertices, sphere.faces)
    rotation = trimesh.transformations.rotation_matrix(0.7, [0.2, 1.0, -0.4])[:3, :3]
    translation = np.array([0.3, -0.2, 0.9])
    canonical = np.array([[0.047, 0, 0], [0.053, 0, 0], [0.2, 0, 0]])
    world = canonical @ rotation.T + translation
    metrics = frame_metrics(surface, world[None], rotation[None], translation[None])
    print({key: value.tolist() for key, value in metrics.items()})
    assert metrics["penetrating_vertices"][0] == 1 and metrics["contact_vertices"][0] == 1
    assert abs(metrics["max_penetration_mm"][0] - 3.0) < 0.2
    print("all metric checks passed")


if __name__ == "__main__":
    main()
