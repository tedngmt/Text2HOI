# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Render identical cameras for recorded and refined MANO geometry."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("LP_NUM_THREADS", "4")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "render_helpers"))
from camera import _camera_spec, _grab_detail_specs
from render_gl import GLRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--footer", default="Recorded-motion cleanup")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()
    with np.load(args.geometry) as archive:
        data = {k: archive[k] for k in archive.files}
    count = len(data["original_vertices"])
    obj = data["object_vertices_canonical"]
    rotation, translation = data["object_rotation"], data["object_translation"]
    corners = np.array(np.meshgrid(*zip(obj.min(0), obj.max(0)))).T.reshape(-1, 3)
    moving = corners @ rotation.transpose(0, 2, 1) + translation[:, None]
    all_points = np.concatenate(
        [data["original_vertices"].reshape(-1, 3), data["soma_vertices"].reshape(-1, 3), moving.reshape(-1, 3)]
    )
    low, high = all_points.min(0), all_points.max(0)
    overview = _camera_spec((low + high) / 2, [0.7, -1, 0.65], max(np.linalg.norm(high - low) * 1.1, 0.4), 2)
    center = (obj.min(0) + obj.max(0)) / 2
    detail_height = max(np.linalg.norm(obj.max(0) - obj.min(0)) * 1.5, 0.4)
    details = [
        _camera_spec(center @ rot.T + trans, [0.7, -1, 0.65], detail_height, 2)
        for rot, trans in zip(rotation, translation)
    ]
    if "_mug_" in args.label:
        details, _ = _grab_detail_specs(rotation, translation, obj.min(0), obj.max(0), 30, 2)
    # The shared renderer uses this historical name for any world-follow camera.
    detail = dict(details[0], coordinate_frame="world_follow_handle")
    specs = dict(canonical_to_z_up=np.eye(3).tolist(), overview=overview, detail=detail, detail_frames=details)
    renderer = GLRenderer(width=640, height=320)
    temporary = args.output.with_suffix(".partial.mp4")
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1280x800",
            "-r",
            "30",
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
            str(temporary),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for frame in range(count):
            panels = renderer.render_clip(data, specs, frame, frame + 1)[0]
            canvas = np.full((800, 1280, 3), [21, 26, 35], dtype=np.uint8)
            canvas[90:410, :640], canvas[90:410, 640:] = panels[0], panels[1]
            canvas[430:750, :640], canvas[430:750, 640:] = panels[2], panels[3]
            for column, key in ((0, "metrics_before"), (640, "metrics_after")):
                if key not in data:
                    continue
                inside, depth, touching, acceleration, total = data[key][frame]
                cv2.rectangle(canvas, (column + 8, 96), (column + 480, 150), (21, 26, 35), -1)
                color = (255, 120, 110) if inside > 0 else (130, 225, 150)
                cv2.putText(
                    canvas,
                    f"Penetrating verts: {int(inside)} / {int(total)}   deepest {depth:.1f} mm",
                    (column + 16, 117),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"Touching verts (<5 mm): {int(touching)}   accel {acceleration:.2f} m/s2",
                    (column + 16, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (235, 239, 245),
                    1,
                    cv2.LINE_AA,
                )
            if args.summary:
                cv2.putText(
                    canvas, args.summary, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 214, 120), 1, cv2.LINE_AA
                )
            for text, xy, scale in [
                (args.label, (20, 27), 0.65),
                ("Recorded GRAB / standard MANO", (20, 68), 0.65),
                ("Text2HOI refiner applied", (660, 68), 0.65),
                ("Overview", (20, 425), 0.45),
                ("Object-follow view | Object motion unchanged", (680, 425), 0.45),
                (f"{args.footer} | {frame + 1}/{count} | 30 FPS", (20, 782), 0.6),
            ]:
                cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 239, 245), 1, cv2.LINE_AA)
            process.stdin.write(canvas.tobytes())
            if frame in {0, count // 2, count - 1}:
                cv2.imwrite(
                    str(args.output.with_name(args.output.stem + f"_{frame:05d}.png")),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                )
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,width,height",
                "-of",
                "json",
                str(temporary),
            ]
        )
    )
    stream = info["streams"][0]
    if int(stream["nb_read_frames"]) != count or stream["width"] != 1280 or stream["height"] != 800:
        raise ValueError("Encoded video dimensions or frame count do not match")
    temporary.replace(args.output)
    print("Rendered " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
