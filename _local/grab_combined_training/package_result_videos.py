# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify and package the existing GRAB mug pilot videos for local sharing."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
INTRO_FRAMES = 240
RESET_CARD_FRAMES = 15
FPS = 30


def sha256(path: Path) -> str:
    """Hash a file without loading an entire video into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Atomically write a package report."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_video(path: Path, expected_frames: int, ffprobe: str) -> dict:
    """Decode-count frames and verify the expected 1080p H.264 stream at 30 Hz."""
    if not path.is_file():
        raise FileNotFoundError(f"Rendering is incomplete: {path}")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stderr.strip():
        raise RuntimeError(f"Video decode reported errors for {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"Expected one selected video stream in {path}")
    stream = streams[0]
    expected = {"codec_name": "h264", "width": 1920, "height": 1080, "pix_fmt": "yuv420p"}
    if any(stream.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Wrong video format in {path}: {stream}")
    if Fraction(stream["avg_frame_rate"]) != FPS or Fraction(stream["r_frame_rate"]) != FPS:
        raise ValueError(f"Wrong frame rate in {path}: {stream}")
    actual_frames = int(stream["nb_read_frames"])
    if actual_frames != expected_frames:
        raise ValueError(f"Wrong frame count in {path}: expected {expected_frames}, decoded {actual_frames}")
    duration = float(stream["duration"])
    if abs(duration - expected_frames / FPS) > 0.001:
        raise ValueError(f"Wrong duration in {path}: {duration}")
    return {
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "width": stream["width"],
        "height": stream["height"],
        "fps": FPS,
        "frames": actual_frames,
        "duration_seconds": duration,
        "decoded_frame_count_verified": True,
    }


def copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy one checked artifact, checking that its content stayed unchanged."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != expected_sha256:
        temporary.unlink()
        raise RuntimeError(f"File changed during packaging: {source}")
    temporary.replace(destination)


def make_zip(output: Path, relative_paths: list[Path]) -> dict:
    """Create a stable-named archive and verify its CRCs and member list."""
    target = output / "GRAB_Mug_Pilot_Videos.zip"
    temporary = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for relative in sorted(relative_paths):
            # Videos are already compressed. Fixed metadata makes repeated packaging reproducible.
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if relative.suffix == ".mp4" else zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with (output / relative).open("rb") as source, archive.open(info, "w", force_zip64=True) as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Package archive CRC verification failed")
        if set(archive.namelist()) != {path.as_posix() for path in relative_paths}:
            raise RuntimeError("Package archive member list does not match the requested artifacts")
    temporary.replace(target)
    return {"path": str(target), "sha256": sha256(target), "size_bytes": target.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=LOCAL / "runs/policy_final_pilot/video_export/videos")
    parser.add_argument("--output_dir", type=Path, default=LOCAL / "share")
    args = parser.parse_args()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    report_path = source / "render_report.json"
    if not report_path.is_file():
        raise FileNotFoundError("Render report does not exist yet; wait until all videos finish rendering")
    report = json.loads(report_path.read_text())
    if report.get("complete") is not True:
        raise ValueError("Refusing to package an incomplete render")
    rollout_path = source.parent / "rollouts" / "manifest.json"
    rollout = json.loads(rollout_path.read_text())
    if sha256(rollout_path) != report["rollout_manifest_sha256"]:
        raise ValueError("Rollout manifest changed after rendering")
    if (report["selected_epoch"], report["latest_epoch"], rollout["split"], rollout["window_count"]) != (
        0,
        2,
        "validation",
        18,
    ):
        raise ValueError("This package describes the specific two-epoch pilot; its captions do not fit this input")
    if report["full_100_epoch_training_completed"] or rollout["test_split_used"]:
        raise ValueError("Package description would be incorrect for this experiment")
    clip_frames = defaultdict(int)
    for window, rendered in zip(rollout["windows"], report["windows"]):
        length = window["end"] - window["start"]
        if (window["window_index"], window["clip_id"], length) != (
            rendered["window_index"],
            rendered["clip_id"],
            rendered["frames"],
        ):
            raise ValueError("Rendered windows do not match the exported rollout timeline")
        clip_frames[window["clip_id"]] += length + RESET_CARD_FRAMES
    if len(report["windows"]) != len(rollout["windows"]) or len(clip_frames) != 8:
        raise ValueError("Expected all 18 validation windows and all eight supported validation clips")
    compilation_frames = sum(clip_frames.values()) + INTRO_FRAMES
    if report["compilation_frames"] != compilation_frames:
        raise ValueError("Render report has an unexpected compilation frame count")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("ffprobe is required to verify actual encoded video streams")
    jobs = [
        (
            source / "GRAB_Mug_Combined_Pipeline_Pilot.mp4",
            Path("GRAB_Mug_Pilot_Comparison.mp4"),
            compilation_frames,
        )
    ]
    jobs.extend(
        (source / "clips" / f"{clip}.mp4", Path("clips") / f"{clip}.mp4", frames)
        for clip, frames in sorted(clip_frames.items())
    )
    # Verify every source before creating any deliverable copies.
    with ThreadPoolExecutor(max_workers=2) as pool:
        checks = list(pool.map(lambda job: verify_video(job[0], job[2], ffprobe), jobs))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for the complete compilation decode check")
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-threads", "2", "-i", str(jobs[0][0]), "-f", "null", "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    if decoded.stderr.strip():
        raise RuntimeError(f"Compilation decode reported errors: {decoded.stderr.strip()}")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for (input_path, relative, _), check in zip(jobs, checks):
        copy_verified(input_path, output / relative, check["sha256"])
        artifacts.append(dict(check, path=relative.as_posix()))
    readme = """GRAB MUG: TEXT2HOI + GRASPXL PILOT RESULTS

Open GRAB_Mug_Pilot_Comparison.mp4 for the complete comparison (56.53 seconds).
The clips/ folder contains all eight supported validation recordings separately.
All videos are 1920 x 1080, H.264, 30 fps. No audio is included.

The three columns show:
1. Original GRAB recording: personalized MANO hand and recorded mug movement.
2. Validation-selected controller: epoch 0 reference-PD baseline.
3. Latest trained PPO controller: epoch 2, rejected by validation safeguards.

The top row uses a shared fixed world camera within each window. The bottom row
follows each mug's handle, so use the top row to compare object drift.
The physics columns use native GraspXL visual hand meshes. Their hand shape and
segmented appearance differ from the personalized MANO recording.

Coverage: all 18 validation windows from eight eligible right-hand mug clips.
Physics resets at each labeled window boundary; these are separate rollouts.
Some source intervals overlap. A red FAILED label means the last valid physical
state is frozen after terminal failure; automatic simulator resets are hidden.
The mug moves under gravity and hand/table contact in the physics columns.

This is a short two-epoch PPO test. The full 100-epoch training has NOT run.
The trained candidate did not improve validation sufficiently and was rejected;
the videos demonstrate the current result, including failed grasps and drops.
Validation subjects were excluded from this local fine-tuning, but exposure
during released Text2HOI pretraining cannot be excluded. No test split was used.

GRAB_Mug_Pilot_Manifest.json records video hashes, dimensions, decoded frame
counts, source provenance, checkpoint hashes, and the validation metrics.
"""
    readme_path = output / "README.txt"
    readme_path.write_text(readme)
    package = {
        "schema_version": 1,
        "scope": "Two-epoch Text2HOI/GraspXL mug-controller pilot; all supported validation videos",
        "full_100_epoch_training_completed": False,
        "test_split_used": False,
        "validation_clips": len(clip_frames),
        "validation_windows": len(rollout["windows"]),
        "compilation_frames": compilation_frames,
        "compilation_duration_seconds": compilation_frames / FPS,
        "window_reset_card_frames": RESET_CARD_FRAMES,
        "intro_frames": INTRO_FRAMES,
        "source_render_report_sha256": sha256(report_path),
        "source_rollout_manifest_sha256": sha256(rollout_path),
        "source_data_manifest_sha256": rollout["manifest_sha256"],
        "environment_sha256": rollout["environment_sha256"],
        "checkpoint_provenance": rollout["checkpoint_provenance"],
        "validation_summaries": {key: value["summary"] for key, value in rollout["summaries"].items()},
        "readme": {"path": "README.txt", "sha256": sha256(readme_path)},
        "videos": artifacts,
    }
    manifest_path = Path("GRAB_Mug_Pilot_Manifest.json")
    write_json(output / manifest_path, package)
    members = [Path(item["path"]) for item in artifacts] + [Path("README.txt"), manifest_path]
    archive = make_zip(output, members)
    verification = {
        "passed": True,
        "video_count": len(artifacts),
        "all_streams_decoded_and_frame_counts_verified": True,
        "compilation_full_ffmpeg_decode_passed": True,
        "all_copy_hashes_verified": True,
        "archive_crc_and_members_verified": True,
        "artifact_manifest_sha256": sha256(output / manifest_path),
        "zip": archive,
    }
    write_json(output / "package_verification.json", verification)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
