"""Per-frame hand/object penetration and contact counts from the real object mesh.

A hand vertex counts as penetrating when it lies inside the object surface by more
than PENETRATION_TOLERANCE_M, and as touching when it lies within CONTACT_DISTANCE_M
of the surface without penetrating. Distances are exact point-to-triangle distances
on the full GRAB contact mesh. Candidate triangles are those touching the nearest
mesh vertices plus those with the nearest centers (the latter catches large flat
triangles whose corners are all far away); the PyTorch3D point-to-face CUDA kernel is
deliberately not used because it picks wrong triangles on meter-scale scans
(errors above 100 mm were measured on the GRAB mug). The inside/outside sign uses
angle-weighted vertex normals blended at the closest surface point, which stays
usable on the few GRAB meshes that are not watertight.

This measures geometry only. It is not the upstream training penetration loss, which
uses a 1,024-point proxy with normals.
"""

import numpy as np
import torch
import trimesh
from pytorch3d.ops import knn_points

PENETRATION_TOLERANCE_M = 0.001
CONTACT_DISTANCE_M = 0.005
NEAR_MARGIN_M = 0.03
NEAREST_VERTICES = 8
NEAREST_CENTROIDS = 8
CANDIDATE_BUDGET = 1_500_000  # point-triangle pairs evaluated per chunk (double precision)


def closest_points_on_triangles(p, a, b, c):
    """Closest point on each triangle (a, b, c) to each point p, with barycentric weights.

    Follows Ericson, Real-Time Collision Detection, section 5.1.5, vectorized.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    tiny = 1e-20
    # Interior of the face by default.
    denominator = (va + vb + vc).clamp_min(tiny)
    v = vb / denominator
    w = vc / denominator
    # Edge BC.
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(tiny)
    v = torch.where(edge_bc, 1 - t, v)
    w = torch.where(edge_bc, t, w)
    # Edge AC.
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / (d2 - d6).clamp_min(tiny)
    v = torch.where(edge_ac, torch.zeros_like(v), v)
    w = torch.where(edge_ac, t, w)
    # Edge AB.
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / (d1 - d3).clamp_min(tiny)
    v = torch.where(edge_ab, t, v)
    w = torch.where(edge_ab, torch.zeros_like(w), w)
    # Vertex regions override the edges.
    at_c = (d6 >= 0) & (d5 <= d6)
    v = torch.where(at_c, torch.zeros_like(v), v)
    w = torch.where(at_c, torch.ones_like(w), w)
    at_b = (d3 >= 0) & (d4 <= d3)
    v = torch.where(at_b, torch.ones_like(v), v)
    w = torch.where(at_b, torch.zeros_like(w), w)
    at_a = (d1 <= 0) & (d2 <= 0)
    v = torch.where(at_a, torch.zeros_like(v), v)
    w = torch.where(at_a, torch.zeros_like(w), w)
    u = 1 - v - w
    closest = u[:, None] * a + v[:, None] * b + w[:, None] * c
    return closest, torch.stack([u, v, w], dim=-1)


class ObjectSurface:
    """Signed distance queries against one object mesh in its canonical frame."""

    def __init__(self, vertices, faces, device="cuda"):
        mesh = trimesh.Trimesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces), process=False)
        self.watertight = bool(mesh.is_watertight)
        self.device = torch.device(device)
        self.vertices = torch.as_tensor(np.asarray(vertices), dtype=torch.float32, device=self.device)
        self.faces = torch.as_tensor(np.asarray(faces).astype(np.int64), device=self.device)
        # Closest-point math runs in float64: float32 loses up to 0.2 mm on sliver triangles.
        self.triangles = torch.as_tensor(np.asarray(vertices, dtype=np.float64), device=self.device)[self.faces]
        # trimesh weights vertex normals by face angle: the pseudonormal needed for a reliable sign.
        self.vertex_normals = torch.as_tensor(
            np.array(mesh.vertex_normals), dtype=torch.float32, device=self.device
        )
        # Padded vertex -> incident faces table, so candidates can be gathered on the GPU.
        faces_np = np.asarray(faces).astype(np.int64)
        order = np.argsort(faces_np.reshape(-1), kind="stable")
        owner = np.repeat(np.arange(len(faces_np)), 3)[order]
        vertex = faces_np.reshape(-1)[order]
        counts = np.bincount(vertex, minlength=len(self.vertices))
        table = np.full((len(self.vertices), max(int(counts.max()), 1)), -1, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        table[vertex, np.arange(len(vertex)) - starts[vertex]] = owner
        self.vertex_faces = torch.as_tensor(table, device=self.device)
        self.centroids = self.triangles.mean(1).float().contiguous()
        self.low = self.vertices.min(0).values - NEAR_MARGIN_M
        self.high = self.vertices.max(0).values + NEAR_MARGIN_M

    @torch.no_grad()
    def signed_distance(self, points):
        """Signed distance in meters for (N, 3) canonical-frame points; negative means inside.

        Points outside the object bounding box padded by NEAR_MARGIN_M return +inf:
        they can neither touch nor penetrate, and skipping them keeps long clips fast.
        """
        points = points.to(self.device, torch.float32).contiguous()
        result = torch.full((len(points),), float("inf"), device=self.device)
        near = ((points >= self.low) & (points <= self.high)).all(-1)
        indices = torch.nonzero(near)[:, 0]
        neighbors = min(NEAREST_VERTICES, len(self.vertices))
        central = min(NEAREST_CENTROIDS, len(self.faces))
        per_point = neighbors * self.vertex_faces.shape[1] + central
        step = max(1, CANDIDATE_BUDGET // per_point)
        for start in range(0, len(indices), step):
            chunk = indices[start : start + step]
            query = points[chunk].contiguous()
            nearest_vertices = knn_points(query[None], self.vertices[None], K=neighbors).idx[0]
            nearest_centers = knn_points(query[None], self.centroids[None], K=central).idx[0]
            candidates = torch.cat([self.vertex_faces[nearest_vertices].reshape(len(query), -1), nearest_centers], 1)
            valid = candidates >= 0
            tri = self.triangles[candidates.clamp_min(0)]
            flat_query = query.double()[:, None].expand(-1, per_point, -1).reshape(-1, 3)
            closest, weights = closest_points_on_triangles(
                flat_query, tri[..., 0, :].reshape(-1, 3), tri[..., 1, :].reshape(-1, 3), tri[..., 2, :].reshape(-1, 3)
            )
            distance = (flat_query - closest).norm(dim=-1).reshape(len(query), per_point)
            distance = torch.where(valid, distance, torch.full_like(distance, float("inf")))
            best = distance.argmin(1)
            rows = torch.arange(len(query), device=self.device)
            face = candidates[rows, best]
            closest = closest.reshape(len(query), per_point, 3)[rows, best]
            weights = weights.reshape(len(query), per_point, 3)[rows, best]
            normals = (self.vertex_normals[self.faces[face]].double() * weights[..., None]).sum(1)
            offset = query.double() - closest
            negative = (offset * normals).sum(-1) < 0
            nearest = distance[rows, best]
            result[chunk] = torch.where(negative, -nearest, nearest).float()
        return result


@torch.no_grad()
def frame_metrics(surface, hand_vertices_world, object_rotation, object_translation):
    """Per-frame metrics for hand vertices (T, V, 3) against a moving object.

    object_rotation (T, 3, 3) and object_translation (T, 3) follow the renderer
    convention: world = canonical @ rotation.T + translation.
    Returns a dict of numpy arrays with one value per frame.
    """
    device = surface.device
    hands = torch.as_tensor(np.asarray(hand_vertices_world), dtype=torch.float32, device=device)
    rotation = torch.as_tensor(np.asarray(object_rotation), dtype=torch.float32, device=device)
    translation = torch.as_tensor(np.asarray(object_translation), dtype=torch.float32, device=device)
    frames, count = hands.shape[:2]
    canonical = torch.einsum("tvi,tij->tvj", hands - translation[:, None], rotation)
    signed = surface.signed_distance(canonical.reshape(-1, 3)).reshape(frames, count)
    inside = signed < -PENETRATION_TOLERANCE_M
    depth = torch.where(inside, -signed, torch.zeros_like(signed))
    touching = (signed.abs() <= CONTACT_DISTANCE_M) & ~inside
    return dict(
        penetrating_vertices=inside.sum(1).cpu().numpy().astype(np.int32),
        max_penetration_mm=(depth.max(1).values * 1000).cpu().numpy().astype(np.float32),
        mean_penetration_mm=((depth.sum(1) / inside.sum(1).clamp_min(1)) * 1000).cpu().numpy().astype(np.float32),
        contact_vertices=touching.sum(1).cpu().numpy().astype(np.int32),
        hand_vertices=np.full(frames, count, dtype=np.int32),
    )


def summarize(metrics):
    """Clip-level summary of frame_metrics output, as plain Python numbers."""
    penetrating = metrics["penetrating_vertices"]
    frames = len(penetrating)
    if not frames:
        return dict(frames=0)
    return dict(
        frames=int(frames),
        hand_vertices_per_frame=int(metrics["hand_vertices"][0]),
        frames_with_penetration=int((penetrating > 0).sum()),
        percent_frames_with_penetration=float(100 * (penetrating > 0).mean()),
        mean_penetrating_vertices=float(penetrating.mean()),
        max_penetrating_vertices=int(penetrating.max()),
        max_penetration_mm=float(metrics["max_penetration_mm"].max()),
        mean_of_frame_max_penetration_mm=float(metrics["max_penetration_mm"].mean()),
        mean_contact_vertices=float(metrics["contact_vertices"].mean()),
        frames_with_contact=int((metrics["contact_vertices"] > 0).sum()),
        penetration_tolerance_mm=PENETRATION_TOLERANCE_M * 1000,
        contact_distance_mm=CONTACT_DISTANCE_M * 1000,
    )
