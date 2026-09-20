# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Verify released GRAB row identities against raw recordings before splitting."""

import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

LOCAL = Path(__file__).resolve().parent
PROJECTS = LOCAL.parents[2]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    data_path = PROJECTS / "Text2HOI/data/grab/data.npz"
    with np.load(data_path, allow_pickle=True) as data:
        objects, lengths = data["obj_name"], data["nframes"]
        trajectories = {key: data[key] for key in ("x_obj", "x_lhand", "x_rhand")}
    inventory = json.loads((LOCAL.parent / "refiner_tools/inventory.json").read_text())
    records = sorted(inventory["clips"], key=lambda row: row["source_member"])
    assert len(records) == len(objects) == 1335
    archive = None
    archive_path = None
    maximum_error = 0.0
    fingerprints = {}
    for index, row in enumerate(records):
        if row["source_archive"] != archive_path:
            if archive is not None:
                archive.close()
            archive_path = row["source_archive"]
            archive = zipfile.ZipFile(archive_path)
        with np.load(io.BytesIO(archive.read(row["source_member"])), allow_pickle=True) as source:
            assert str(source["obj_name"].item()) == str(objects[index]), (index, row["clip_id"])
            contact = np.nonzero(np.sum(source["contact"].item()["object"][::4], axis=1))[0]
            start, end = int(contact[0]), int(contact[-1])
            assert end - start == int(lengths[index]), (index, "length mismatch")
            count = end - start
            for key, raw_key in (("x_obj", "object"), ("x_lhand", "lhand"), ("x_rhand", "rhand")):
                params = source[raw_key].item()["params"]
                translation = params["transl"][::4][start:end]
                orientation = params["global_orient"][::4][start:end]
                if raw_key != "object":
                    orientation = np.concatenate([orientation, params["fullpose"][::4][start:end]], axis=1)
                rotation = Rotation.from_rotvec(orientation.reshape(-1, 3)).as_matrix()[:, :, :2]
                expected = np.concatenate([translation, rotation.reshape(count, -1)], axis=1)
                error = float(np.max(np.abs(expected - trajectories[key][index, :count])))
                if error > 2e-5:
                    raise ValueError(f"Row identity verification failed: {row['clip_id']} {key}: {error}")
                maximum_error = max(maximum_error, error)
        subject = row["subject"]
        split = "test" if subject == "s10" else "validation" if subject == "s9" else "train"
        row.update(index=index, split=split, retained_frames=count, contact_start_30fps=start, contact_end_30fps=end)
        fingerprint = hashlib.sha256()
        for key in trajectories:
            fingerprint.update(np.asarray(trajectories[key][index, :count], dtype=np.float32).tobytes())
        value = fingerprint.hexdigest()
        if value in fingerprints and fingerprints[value] != split:
            raise ValueError("Exact duplicate motion crosses splits")
        fingerprints[value] = split
        row["motion_sha256"] = value
        if (index + 1) % 100 == 0:
            print(f"Verified {index + 1}/1335 native-to-preprocessed clip identities", flush=True)
    archive.close()
    summary = {}
    for split in ("train", "validation", "test"):
        subset = [row for row in records if row["split"] == split]
        summary[split] = dict(
            clips=len(subset),
            subjects=sorted({r["subject"] for r in subset}),
            objects=dict(Counter(r["object_name"] for r in subset)),
            retained_frames=sum(r["retained_frames"] for r in subset),
        )
    manifest = dict(
        version=1,
        seed=42,
        data_path=str(data_path),
        data_sha256=sha256(data_path),
        verified_maximum_parameter_error=maximum_error,
        splits=summary,
        clips=records,
        scope="Subject-held-out refiner experiment, not an unseen end-to-end Text2HOI benchmark",
        pretrained_exposure="Frozen generator and point encoder may have seen all GRAB subjects",
    )
    path = LOCAL / "split_manifest.json"
    contents = json.dumps(manifest, indent=2) + "\n"
    if path.exists() and path.read_text() != contents:
        raise FileExistsError("Existing immutable split differs; do not silently replace an experiment split")
    path.write_text(contents)
    print(json.dumps({k: {a: b for a, b in v.items() if a != "objects"} for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
