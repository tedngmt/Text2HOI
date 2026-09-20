# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render recorded GRAB versus selected and latest physical validation rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / "grab_and_graspxl_mano_and_soma_mug_comparisons"))
from camera import _camera_spec, _grab_detail_specs
from render_gl import Renderer, _normals
from video_geometry import HandVisuals

WIDTH, HEIGHT, FPS = 1920, 1080, 30
PW, PH = 640, 420
COLORS = [(0.22, 0.52, 0.88), (0.12, 0.62, 0.52), (0.94, 0.48, 0.19)]


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


class SceneRenderer(Renderer):
    """Use the existing native-triangle EGL renderer for three independent worlds."""

    def __init__(self):
        super().__init__(width=PW, height=PH)
        self.hand = HandVisuals()
        with (LOCAL.parent / "grab_mug_refinement/prepared/object.npz").open("rb") as stream, np.load(stream) as obj:
            self.object_vertices = obj["object_vertices_canonical"].copy()
            self.object_faces = obj["object_faces"].copy()
        with Path("/home/nmt/manotorch/assets/mano/models/MANO_RIGHT.pkl").open("rb") as stream:
            self.recorded_faces = np.asarray(pickle.load(stream, encoding="latin1")["f"])
        self._upload_mesh(
            "mug", self.object_vertices, _normals(self.object_vertices, self.object_faces), self.object_faces
        )
        self.pixels = np.empty((PH, PW, 4), dtype=np.uint8)

    def begin_window(self, arrays: dict, original: np.ndarray) -> tuple[dict, list[list[dict]]]:
        table = trimesh.creation.box(extents=arrays["table_dimensions"])
        self._upload_mesh("table", table.vertices, table.vertex_normals, table.faces)
        self.table_matrix = pose_matrix(arrays["table_pose"])
        self.table_world = table.vertices @ self.table_matrix[:3, :3].T + self.table_matrix[:3, 3]
        self._upload_mesh("original", original[0], _normals(original[0], self.recorded_faces), self.recorded_faces)
        for key in ("selected", "latest"):
            vertices, normals = self.hand.vertices_and_normals(arrays[key + "_state"][0, :51])
            self._upload_mesh(key, vertices, normals, self.hand.faces)
        object_poses = [
            arrays["reference_object_pose"],
            arrays["selected_state"][:, 102:109],
            arrays["latest_state"][:, 102:109],
        ]
        minimum, maximum = self.object_vertices.min(0), self.object_vertices.max(0)
        corners = np.array(
            [
                [x, y, z]
                for x in (minimum[0], maximum[0])
                for y in (minimum[1], maximum[1])
                for z in (minimum[2], maximum[2])
            ]
        )
        points = [original.reshape(-1, 3), self.table_world]
        for key in ("selected", "latest"):
            points.append(arrays[key + "_state"][:, 115:178].reshape(-1, 3))
            points.append(arrays[key + "_state"][:, :3])
        detail = []
        for poses in object_poses:
            rotations = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
            points.append((corners @ rotations.transpose(0, 2, 1) + poses[:, None, :3]).reshape(-1, 3))
            specs, _ = _grab_detail_specs(rotations, poses[:, :3], minimum, maximum, FPS, PW / PH)
            detail.append(specs)
        # Fit a single fixed world camera to all three results, including dropped mugs.
        direction = np.array([0.65, -1.0, 0.65])
        direction /= np.linalg.norm(direction)
        right = np.cross([0, 0, 1], direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        basis = np.stack([right, up, direction], axis=1)
        projected = np.concatenate(points) @ basis
        low, high = projected.min(0) - 0.065, projected.max(0) + 0.065
        target = ((low + high) * 0.5) @ basis.T
        span = high - low
        overview = _camera_spec(target, direction, max(span[1], span[0] / (PW / PH)), PW / PH, span[2])
        return overview, detail

    def panels(self, arrays: dict, original: np.ndarray, frame: int, overview: dict, detail: list) -> np.ndarray:
        self._upload_mesh("original", original[frame], _normals(original[frame], self.recorded_faces))
        for key in ("selected", "latest"):
            vertices, normals = self.hand.vertices_and_normals(arrays[key + "_state"][frame, :51])
            self._upload_mesh(key, vertices, normals)
        poses = [
            arrays["reference_object_pose"][frame],
            arrays["selected_state"][frame, 102:109],
            arrays["latest_state"][frame, 102:109],
        ]
        output = np.empty((6, PH, PW, 3), dtype=np.uint8)
        for column, key in enumerate(("original", "selected", "latest")):
            for row in range(2):
                camera = overview if row == 0 else detail[column][frame]
                self._matrix("vp", self._view_projection(camera))
                self.gl.glUniform3f(self.locations["light_direction"], *camera["view_direction"])
                self.gl.glClear(0x00004000 | 0x00000100)
                self._draw("table", (0.76, 0.72, 0.65), self.table_matrix)
                self._draw("mug", (0.51, 0.60, 0.63), pose_matrix(poses[column]))
                self._draw(key, COLORS[column], np.eye(4))
                self.gl.glReadPixels(0, 0, PW, PH, 0x1908, 0x1401, self.pixels.ctypes.data)
                output[row * 3 + column] = self.pixels[::-1, :, :3]
        error = self.gl.glGetError()
        if error:
            raise RuntimeError(f"OpenGL error {error:x}")
        return output


def caption(image: np.ndarray, text: str, point: tuple[int, int], size: float = 0.65, color=(239, 243, 246)):
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_DUPLEX, size, color, 1, cv2.LINE_AA)


def compose(panels: np.ndarray, record: dict, arrays: dict, frame: int, selected_epoch: int, latest_epoch: int):
    canvas = np.full((HEIGHT, WIDTH, 3), (32, 28, 23), np.uint8)
    labels = ["GRAB RECORDING", f"SELECTED CONTROLLER | epoch {selected_epoch}", f"TRAINED PPO | epoch {latest_epoch}"]
    subtitles = [
        "Personalized MANO | recorded mug motion",
        "Baseline retained by validation",
        "Not selected: validation safeguards failed",
    ]
    for column in range(3):
        x = column * PW
        rgb = tuple(int(value * 255) for value in COLORS[column])
        caption(canvas, labels[column], (x + 18, 34), 0.73, rgb[::-1])
        caption(canvas, subtitles[column], (x + 18, 64), 0.52)
        caption(canvas, "WORLD VIEW | fixed shared camera", (x + 18, 89), 0.49, (180, 189, 194))
        for row, y in enumerate((100, 552)):
            canvas[y : y + PH, x : x + PW] = panels[row * 3 + column][..., ::-1]
        if column and arrays[("selected" if column == 1 else "latest") + "_failed"][frame]:
            for y in (100, 552):
                cv2.rectangle(canvas, (x + 10, y + 12), (x + PW - 10, y + 48), (48, 42, 155), -1)
                caption(canvas, "FAILED | last valid state held", (x + 24, y + 37), 0.61)
    caption(canvas, "HANDLE CLOSE-UP | camera follows each mug; use world view above to compare drift", (18, 542), 0.62)
    for x in (PW, 2 * PW):
        cv2.line(canvas, (x, 0), (x, 520), (95, 100, 105), 2)
        cv2.line(canvas, (x, 552), (x, 972), (95, 100, 105), 2)
    label = record["clip_id"].replace("__", " / ")
    caption(
        canvas,
        f"{label} | validation window {record['window_index'] + 1}/18"
        f" | frame {frame + 1}/{len(arrays['source_frames'])}",
        (20, 1001),
        0.68,
    )
    caption(
        canvas,
        f"30 fps | source time {int(arrays['source_frames'][frame]) / 120:.2f} s"
        " | right hand only | physics resets between windows",
        (20, 1032),
        0.57,
    )
    caption(
        canvas,
        "2-epoch PPO test only; 100-epoch run has not started. Physics columns show the GraspXL visual hand.",
        (20, 1062),
        0.57,
        (171, 197, 222),
    )
    return canvas


def title_card(record: dict | None = None) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), (32, 28, 23), np.uint8)
    caption(canvas, "GRAB MUG | TEXT2HOI + GRASPXL", (105, 260), 1.30)
    if record is None:
        lines = [
            "GRAB recording  /  Selected baseline (epoch 0)  /  Trained PPO (epoch 2)",
            "18 validation windows from 8 clips. Failed attempts are included.",
            "Separate simulation windows; physics resets between them.",
            "The trained candidate did not pass validation. Baseline was retained.",
            "Short test only. The 100-epoch run has not started.",
        ]
    else:
        lines = [
            record["clip_id"].replace("__", " / "),
            f"Validation window {record['window_index'] + 1}/18 | episode {record['episode_index'] + 1}",
            "PHYSICS RESET | all columns begin from this window's reference state",
        ]
    for index, text in enumerate(lines):
        caption(canvas, text, (110, 360 + index * 66), 0.85 if record else 0.75)
    return canvas


class Encoder:
    def __init__(self, path: Path):
        self.path = path
        self.partial = path.with_name(path.stem + ".partial.mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{WIDTH}x{HEIGHT}",
                "-r",
                str(FPS),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-threads",
                "4",
                str(self.partial),
            ],
            stdin=subprocess.PIPE,
        )
        self.count = 0

    def write(self, frame: np.ndarray):
        self.process.stdin.write(frame.tobytes())
        self.count += 1

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("Video encoding failed")
        self.partial.replace(self.path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=LOCAL / "runs/policy_final_pilot/video_export/rollouts/manifest.json"
    )
    parser.add_argument("--output_dir", type=Path, default=LOCAL / "runs/policy_final_pilot/video_export/videos")
    parser.add_argument("--preview_only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest["windows"]
    selected_epoch = manifest["checkpoint_provenance"]["selected"]["epoch"]
    latest_epoch = manifest["checkpoint_provenance"]["latest"]["epoch"]
    if (selected_epoch, latest_epoch, len(records), manifest["split"]) != (0, 2, 18, "validation"):
        raise ValueError("These captions describe the specific two-epoch validation pilot; update them for another run")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    renderer = SceneRenderer()
    print(json.dumps(renderer.renderer_info), flush=True)
    started = time.monotonic()
    reference_root = Path(manifest["prepared"]) / "clips"
    original_root = LOCAL.parent / "grab_mug_refinement/prepared/clips"
    compilation = None if args.preview_only else Encoder(output / "GRAB_Mug_Combined_Pipeline_Pilot.mp4")
    if compilation:
        for _ in range(240):
            compilation.write(title_card())
    clip_encoder, previous_clip = None, None
    details = []
    for record in records:
        path = Path(record["npz_path"])
        if not path.is_absolute():
            path = args.manifest.parent / path
        with np.load(path) as data:
            arrays = {key: data[key].copy() for key in data.files}
        with np.load(reference_root / (record["clip_id"] + ".npz")) as reference:
            rotation, shift = reference["world_to_training_rotation"], reference["world_to_training_translation"]
        with np.load(original_root / (record["clip_id"] + ".npz")) as data:
            original = data["rhand_vertices"][record["start"] : record["end"]] @ rotation.T + shift
        overview, detail = renderer.begin_window(arrays, original)
        count = len(original)
        frames = sorted({0, count // 2, count - 1}) if args.preview_only else range(count)
        if compilation:
            if previous_clip != record["clip_id"]:
                if clip_encoder:
                    clip_encoder.close()
                clip_encoder = Encoder(output / "clips" / (record["clip_id"] + ".mp4"))
                previous_clip = record["clip_id"]
            for _ in range(15):
                card = title_card(record)
                compilation.write(card)
                clip_encoder.write(card)
        for frame in frames:
            panels = renderer.panels(arrays, original, frame, overview, detail)
            image = compose(panels, record, arrays, frame, selected_epoch, latest_epoch)
            if frame in {0, count // 2, count - 1}:
                preview = output / "previews" / f"window_{record['window_index']:02d}_frame_{frame:03d}.jpg"
                preview.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(preview), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if compilation:
                compilation.write(image)
                clip_encoder.write(image)
        details.append(
            {
                "window_index": record["window_index"],
                "clip_id": record["clip_id"],
                "frames": count,
                "overview_camera": overview,
            }
        )
        print(
            json.dumps({"rendered_window": record["window_index"], "clip": record["clip_id"], "frames": count}),
            flush=True,
        )
        if args.preview_only:
            break
    if compilation:
        clip_encoder.close()
        compilation.close()
    report = {
        "complete": not args.preview_only,
        "fps": FPS,
        "resolution": [WIDTH, HEIGHT],
        "renderer": renderer.renderer_info,
        "windows": details,
        "selected_epoch": selected_epoch,
        "latest_epoch": latest_epoch,
        "rollout_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "renderer_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "full_100_epoch_training_completed": False,
        "failure_display": "Hold last valid state with red FAILED label; never show automatic reset as motion",
        "overview": "Same fixed camera and scale across all columns within each window",
        "detail": "Each column follows its own mug with world-up yaw camera; actual mug tilt is retained",
        "elapsed_seconds": time.monotonic() - started,
        "compilation_frames": compilation.count if compilation else None,
    }
    (output / ("preview_report.json" if args.preview_only else "render_report.json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
