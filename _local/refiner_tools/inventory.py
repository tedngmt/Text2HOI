# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inventory all native GRAB recordings and their exact object geometry."""

from __future__ import annotations

import io
import json
import pickle
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh

LOCAL = Path(__file__).resolve().parent
PROJECTS = LOCAL.parents[2]


def main():
    root = PROJECTS / "Grab Dataset"
    records = []
    for path in sorted(root.glob("grab__s*.zip")):
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith(".npz") or name.startswith("__MACOSX"):
                    continue
                with np.load(io.BytesIO(archive.read(name)), allow_pickle=True) as source:
                    subject = str(source["sbj_id"].item()) if "sbj_id" in source else Path(name).parts[0]
                    records.append(
                        {
                            "clip_id": subject + "__" + Path(name).stem,
                            "subject": subject,
                            "object_name": str(source["obj_name"].item()),
                            "gender": str(source["gender"].item()),
                            "action": str(source["motion_intent"].item()),
                            "source_frames": int(source["n_frames"]),
                            "frames": len(range(0, int(source["n_frames"]), 4)),
                            "source_fps": float(source["framerate"]),
                            "source_archive": str(path),
                            "source_member": name,
                        }
                    )
        print(json.dumps({"archive": path.name, "clips_so_far": len(records)}), flush=True)
    with (PROJECTS / "Text2HOI/data/grab/obj.pkl").open("rb") as stream:
        pretrained = pickle.load(stream)
    objects = {}
    with zipfile.ZipFile(root / "tools__object_meshes__contact_meshes.zip") as archive:
        for name in sorted({row["object_name"] for row in records}):
            mesh = trimesh.load(io.BytesIO(archive.read(f"contact_meshes/{name}.ply")), file_type="ply", process=False)
            objects[name] = {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
                "volume_m3": float(mesh.volume),
                "extent_m": mesh.extents.tolist(),
                "has_pretrained_points": name in pretrained["obj_pcs"],
                "clips": sum(r["object_name"] == name for r in records),
            }
    report = {
        "clips": records,
        "objects": objects,
        "clip_count": len(records),
        "object_count": len(objects),
        "frames_30fps": sum(r["frames"] for r in records),
        "subject_counts": dict(Counter(r["subject"] for r in records)),
        "invalid_source_rates": [r["clip_id"] for r in records if r["source_fps"] != 120],
    }
    (LOCAL / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("clips", "objects")}))
    print(
        json.dumps(
            {
                "geometry_issues": {
                    key: obj
                    for key, obj in objects.items()
                    if not (
                        obj["watertight"]
                        and obj["winding_consistent"]
                        and obj["volume_m3"] > 0
                        and obj["has_pretrained_points"]
                    )
                }
            }
        )
    )


if __name__ == "__main__":
    main()
