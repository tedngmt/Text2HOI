# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate a completed mug demo and extract unmodified frames for inspection."""

import json
from pathlib import Path

import cv2
import numpy as np


def main():
    local = Path(__file__).resolve().parent
    run = json.loads((local / "last_run.json").read_text())
    if not run["completed"]:
        raise ValueError("The last run did not complete.")
    output = Path(run["output"])
    video = output / "motion/batch0_0_sample0_refined.mp4"
    data = np.load(output / "generated_motion.npz", allow_pickle=False)
    nframes = run["frames"]
    assert all(np.isfinite(data[key]).all() for key in data.files)
    assert data["right_hand_vertices"].shape == (nframes, 778, 3)
    assert data["object_vertices"].shape[0] == nframes
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened(), "Video export cannot be decoded."
    fps = capture.get(cv2.CAP_PROP_FPS)
    selected = {0, nframes // 2, nframes - 1}
    count = 0
    contrasts = []
    while True:
        okay, frame = capture.read()
        if not okay:
            break
        assert frame.shape == (512, 512, 3)
        contrasts.append(float(frame.std()))
        if count in selected:
            assert cv2.imwrite(str(output / ("frame_%03d.png" % count)), frame)
        count += 1
    capture.release()
    assert count == nframes
    assert abs(fps - run["fps"]) < 0.01
    assert min(contrasts) > 1.0, "At least one frame appears blank."
    object_exports = list((output / "obj_file/batch0_0_sample0/object").glob("*.obj"))
    right_exports = list((output / "obj_file/batch0_0_sample0/rhand").glob("*.obj"))
    assert len(object_exports) == nframes and len(right_exports) == nframes
    report = {
        "video": str(video),
        "decoded_frames": count,
        "fps": fps,
        "seconds": count / fps,
        "resolution": [512, 512],
        "object_mesh_exports": len(object_exports),
        "right_hand_mesh_exports": len(right_exports),
        "minimum_frame_standard_deviation": min(contrasts),
        "all_generated_arrays_finite": True,
        "scope": "Output-integrity checks only; no claim of grasp accuracy or dynamic stability.",
    }
    (output / "output_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
