# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local paired GraspXL and GRAB mesh viewer; run in the soma-x environment."""

import argparse
import io
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SOMA-X"))
from soma._smpl_family_loader import ensure_chumpy_compat
from soma.hand import SOMAHandLayer

from tools.replay_grab_soma import load_personalized_layer, replay

ASSETS = ROOT / "SOMA-X/assets"
MANO_ROOT = ROOT / "GraspXL_MANO_51Objects_GRABMatched"
SOMA_ROOT = ROOT / "GraspXL_SOMA_51Objects_GRABMatched"
GRAB_ROOT = ROOT / "GRAB_SOMA_51Objects"


def _object_rotation(kind: str, rotvec: np.ndarray) -> np.ndarray:
    """Return active object rotation matrices from each dataset's convention.

    GRAB's official ObjectModel applies row vertices @ Rodrigues(rotvec),
    whereas GraspXL applies row vertices @ Rodrigues(rotvec).T. The returned
    matrices always use world_vertices = vertices @ rotation.T + translation.
    """
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    return np.swapaxes(rotation, -1, -2) if kind == "GRAB" else rotation


class Sequence:
    def __init__(self, kind, entry, device):
        self.kind, self.device = kind, device
        if kind == "GraspXL":
            self.motion = dict(np.load(SOMA_ROOT / entry["soma_path"], allow_pickle=False))
            self.source = np.load(MANO_ROOT / entry["mano_path"], allow_pickle=True).item()
            self.layer = SOMAHandLayer(
                data_root=str(ASSETS), hand_type="right", identity_model_type="mano", device=device
            ).to(device)
            self.layer.prepare_identity(torch.zeros(1, 10, device=device))
            self.original = smplx.MANO(
                str(ASSETS / "MANO/MANO_RIGHT.pkl"),
                is_rhand=True,
                use_pca=False,
                flat_hand_mean=False,
                create_transl=False,
            ).to(device)
            mesh_path = MANO_ROOT / entry["mesh_path"]
        else:
            self.motion = dict(np.load(GRAB_ROOT / entry, allow_pickle=False))
            self.layer = load_personalized_layer(GRAB_ROOT, self.motion, ASSETS, device)
            subject = Path(entry).parts[1]
            archive = ROOT / "Grab Dataset" / f"grab__{subject}.zip"
            with zipfile.ZipFile(archive) as zf:
                candidates = [n for n in zf.namelist() if Path(n).name == Path(entry).name]
                if len(candidates) != 1:
                    raise ValueError(f"Expected one original clip, found {candidates}")
                self.source = dict(np.load(io.BytesIO(zf.read(candidates[0])), allow_pickle=True))
            gender = str(self.motion["gender"].item())
            self.original = smplx.SMPLX(
                str(ASSETS / "SMPLX" / f"SMPLX_{gender.upper()}.npz"),
                gender=gender,
                use_pca=True,
                num_pca_comps=24,
                flat_hand_mean=False,
                v_template=self.layer.identity_model.identity_model.v_template,
                batch_size=1,
            ).to(device)
            mesh_path = GRAB_ROOT / str(self.motion["package_mesh_path"].item())
        self.mesh = trimesh.load(mesh_path, process=False, force="mesh")
        self.count = len(self.motion["poses"])
        self.faces = self.layer.faces.detach().cpu().numpy()
        self.original_faces = self.original.faces
        if kind == "GRAB":
            with zipfile.ZipFile(ROOT / "Grab Dataset/tools__smplx_correspondence.zip") as archive:
                ids = []
                for side in ("lhand", "rhand"):
                    name = next(n for n in archive.namelist() if Path(n).name == f"{side}_smplx_ids.npy")
                    ids.append(np.load(io.BytesIO(archive.read(name)), allow_pickle=False))
                original_ids = np.unique(np.concatenate(ids))
            with np.load(ASSETS / "SOMAHand.npz", allow_pickle=False) as hand_asset:
                mid_ids = np.concatenate([hand_asset["left_vert_ids"], hand_asset["right_vert_ids"]])
            low_to_mid = self.layer.nv_lod_mid_to_low.detach().cpu().numpy()
            soma_ids = np.flatnonzero(np.isin(low_to_mid, mid_ids))
            self.hand_vertex_ids = (original_ids, soma_ids)
            self.hand_faces = tuple(
                self._crop_faces(faces, vertex_ids)
                for faces, vertex_ids in zip((self.original_faces, self.faces), self.hand_vertex_ids)
            )

    @staticmethod
    def _crop_faces(faces: np.ndarray, vertex_ids: np.ndarray) -> np.ndarray:
        """Keep triangles wholly inside the selected vertices and remap their indices."""
        selected = faces[np.isin(faces, vertex_ids).all(axis=1)]
        if len(selected) == 0:
            raise ValueError("Hand selection contains no mesh faces")
        return np.searchsorted(vertex_ids, selected)

    def display_geometry(
        self, arrays: tuple[np.ndarray, np.ndarray, np.ndarray], hands_only: bool = False
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Select display meshes without changing their shared coordinates [m]."""
        if self.kind == "GRAB" and hands_only:
            original, soma, obj = arrays
            original_ids, soma_ids = self.hand_vertex_ids
            return (original[original_ids], soma[soma_ids], obj), self.hand_faces
        return arrays, (self.original_faces, self.faces)

    def tensor(self, data):
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def frame(self, index):
        m = self.motion
        if self.kind == "GraspXL":
            hand = self.source["right_hand"]
            source = self.original(
                global_orient=self.tensor(hand["rot"][index : index + 1]),
                hand_pose=self.tensor(hand["pose"][index : index + 1]),
                betas=torch.zeros(1, 10, device=self.device),
            ).vertices
            source = source + self.tensor(hand["trans"][index : index + 1])[:, None]
            fitted = self.layer.pose(
                self.tensor(m["poses"][index : index + 1]),
                pose2rot=m["poses"].ndim == 3,
                absolute_pose=bool(m["absolute_pose"]),
                global_translation=self.tensor(m["transl"][index : index + 1]),
            )["vertices"]
        else:
            params = self.source["body"].item()["params"]
            keys = [
                "global_orient",
                "body_pose",
                "left_hand_pose",
                "right_hand_pose",
                "jaw_pose",
                "leye_pose",
                "reye_pose",
                "expression",
                "transl",
            ]
            source = self.original(
                **{k: self.tensor(params[k][index : index + 1]) for k in keys},
                betas=torch.zeros(1, 10, device=self.device),
            ).vertices
            fitted = replay(self.layer, m, np.array([index]))["vertices"]
        rot = _object_rotation(self.kind, m["object_rot"][index].reshape(3))
        obj = np.asarray(self.mesh.vertices) @ rot.T + m["object_trans"][index].reshape(3)
        return source[0].cpu().numpy(), fitted[0].cpu().numpy(), obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ensure_chumpy_compat()
    # The paired GraspXL export is optional; GRAB alone is enough to run the viewer.
    pairs_file = MANO_ROOT / "metadata/pairs.json"
    pairs = json.loads(pairs_file.read_text()) if pairs_file.exists() else []
    if not pairs:
        print(f"GraspXL pairs not found at {pairs_file}; showing GRAB only.", flush=True)
    grasp = {f"{p['grab_object']} / {Path(p['mano_path']).parent.name} / {Path(p['mano_path']).stem}": p for p in pairs}
    grab = {
        str(p.relative_to(GRAB_ROOT)): str(p.relative_to(GRAB_ROOT))
        for p in sorted((GRAB_ROOT / "motions").glob("*/*.npz"))
    }
    if args.check:
        checks = [("GraspXL", pairs[0])] if pairs else []
        if grab:
            checks.append(("GRAB", next(iter(grab.values()))))
        if not checks:
            raise SystemExit("No GraspXL pairs or GRAB motions were found next to this project.")
        for kind, entry in checks:
            sequence = Sequence(kind, entry, args.device)
            for i in [0, sequence.count // 2, sequence.count - 1]:
                arrays = sequence.frame(i)
                assert all(np.isfinite(a).all() for a in arrays)
                print(kind, i, [a.shape for a in arrays], flush=True)
                for hands_only in (False, True):
                    displayed, faces = sequence.display_geometry(arrays, hands_only)
                    assert np.array_equal(displayed[2], arrays[2])
                    for vertices, triangles in zip(displayed[:2], faces):
                        assert np.isfinite(vertices).all()
                        assert len(triangles) > 0 and triangles.min() >= 0 and triangles.max() < len(vertices)
                    if kind == "GRAB" and hands_only:
                        for full, cropped, ids in zip(arrays[:2], displayed[:2], sequence.hand_vertex_ids):
                            assert np.array_equal(cropped, full[ids])
                            assert len(cropped) < len(full)
                        print("  Hands only:", [a.shape for a in displayed], flush=True)
        return
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    server.scene.set_up_direction("+z")
    server.gui.add_markdown("## MANO / SOMA comparison\nBlue: original · Orange: SOMA. Object scale is preserved.")
    available = tuple(name for name, entries in (("GraspXL", grasp), ("GRAB", grab)) if entries)
    if not available:
        raise SystemExit("No GraspXL pairs or GRAB motions were found next to this project.")
    dataset = server.gui.add_dropdown("Dataset", available)
    options = list(grasp if available[0] == "GraspXL" else grab)
    select = server.gui.add_dropdown(
        "Sequence", options, initial_value=next((k for k in options if k.startswith("mug")), options[0])
    )
    load = server.gui.add_button("Load sequence")
    frame = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Play", False)
    speed = server.gui.add_slider("Playback frames/s", min=1, max=120, step=1, initial_value=30)
    mode = server.gui.add_dropdown("View", ("Side by side", "Overlay"))
    body_view = server.gui.add_dropdown("GRAB display", ("Full body", "Hands only"), visible=False)
    status = server.gui.add_markdown("Loading…")
    state = {"reload": True, "camera": ((0.0, 0.0, 0.0), (0.65, -0.85, 0.65))}

    @dataset.on_update
    def _(_event):
        select.options = list(grasp if dataset.value == "GraspXL" else grab)
        select.value = select.options[0]
        playing.value = False

    @load.on_click
    def _(_event):
        state["reload"] = True
        playing.value = False

    @server.on_client_connect
    def _(client):
        client.camera.look_at, client.camera.position = state["camera"]

    sequence = None
    handles = []
    previous = None
    cached_index = None
    cached_frame = None
    tick = time.monotonic()
    while True:
        try:
            if state["reload"]:
                state["reload"] = False
                status.content = "Reconstructing selected sequence…"
                sequence = Sequence(
                    dataset.value, (grasp if dataset.value == "GraspXL" else grab)[select.value], args.device
                )
                frame.max = sequence.count - 1
                frame.value = 0
                previous = None
                cached_index = None
                body_view.visible = sequence.kind == "GRAB"
                label = "MANO" if sequence.kind == "GraspXL" else "SMPL-X (original GRAB)"
                note = (
                    "Source FPS unknown; playback speed is adjustable."
                    if sequence.kind == "GraspXL"
                    else "Source: 120 FPS. Playback may be slower during reconstruction."
                )
                status.content = f"**{label} ↔ SOMA** · {sequence.count} frames\n\n{note}"
                print(f"Loaded {sequence.kind}: {select.value}", flush=True)
            now = time.monotonic()
            if sequence is not None and playing.value and now - tick >= 1 / speed.value:
                frame.value = (frame.value + 1) % sequence.count
                tick = now
            hands_only = sequence is not None and sequence.kind == "GRAB" and body_view.value == "Hands only"
            key = (frame.value, mode.value, hands_only)
            if sequence is not None and key != previous:
                if cached_index != frame.value:
                    cached_index = frame.value
                    cached_frame = sequence.frame(cached_index)
                arrays, faces = sequence.display_geometry(cached_frame, hands_only)
                original, soma, obj = arrays
                layout_changed = previous is None or key[1:] != previous[1:]
                if layout_changed:
                    center = obj.mean(axis=0)
                    points = np.concatenate(arrays)
                    low, high = points.min(axis=0), points.max(axis=0)
                    span = max((high - low).max(), 0.2)
                    shift = span * 1.15
                offset = np.array([shift if mode.value == "Side by side" else 0, 0, 0])
                with server.atomic():
                    vertices = [original - center, obj - center, soma - center + offset, obj - center + offset]
                    if layout_changed:
                        for h in handles:
                            h.remove()
                        handles = []
                        for side, color, body_vertices, obj_vertices, triangles in [
                            ("original", (65, 145, 245), vertices[0], vertices[1], faces[0]),
                            ("soma", (245, 150, 55), vertices[2], vertices[3], faces[1]),
                        ]:
                            handles.append(
                                server.scene.add_mesh_simple(
                                    f"/{side}/body", body_vertices.astype(np.float32), triangles, color=color
                                )
                            )
                            handles.append(
                                server.scene.add_mesh_simple(
                                    f"/{side}/object",
                                    obj_vertices.astype(np.float32),
                                    sequence.mesh.faces,
                                    color=(155, 175, 165),
                                )
                            )
                    else:
                        for h, verts in zip(handles, vertices):
                            h.vertices = verts.astype(np.float32)
                if layout_changed:
                    focus = (low + high) / 2 - center + offset / 2
                    position = focus + np.array([span, -span * 2, span])
                    state["camera"] = (tuple(focus), tuple(position))
                    for client in server.get_clients().values():
                        client.camera.look_at, client.camera.position = state["camera"]
                previous = key
            time.sleep(0.005)
        except Exception as exc:
            traceback.print_exc()
            status.content = f"Error: {exc}"
            playing.value = False
            sequence = None
            time.sleep(0.2)


if __name__ == "__main__":
    main()
