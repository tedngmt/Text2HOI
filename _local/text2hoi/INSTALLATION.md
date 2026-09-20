<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Local Text2HOI setup

Prepared 2026-09-17 using the user's clone, now at `/home/nmt/Projects/Text2HOI`, revision
`9643c209f08e28d7ed870095cea78c7244e4fd42`. The upstream tracked files are unchanged.
The private helpers in this folder use that clone directly.

## Verified first result

The seed-42 run in `outputs/20260917T011303Z_seed_42/` completed successfully:
147 frames at 30 FPS (4.9 seconds), 512 x 512 video, and 147 hand/object mesh
pairs. The generated arrays are finite, every video frame decoded, and the first,
middle, and last frames were visually inspected. Model loading, generation,
refinement, rendering, and export took 43.76 seconds with cached text encoders.
Peak PyTorch allocated GPU memory was 1,355 MiB; reserved memory was 1,468 MiB.
These figures are for one sample and exclude the initial downloads.

[Open the generated video](outputs/20260917T011303Z_seed_42/motion/batch0_0_sample0_refined.mp4).
See the run's `run_report.json` and `output_validation.json` for measured details.

## Run the mug demo

From Windows, double-click [run_mug.bat](run_mug.bat).

From WSL Ubuntu:

```bash
bash /home/nmt/Projects/Text2HOI/_local/text2hoi/run_mug.sh
```

To generate another sample:

```bash
bash /home/nmt/Projects/Text2HOI/_local/text2hoi/run_mug.sh --seed 43
```

The default prompt is **"Lift a mug with the right hand."** The original
CLIP/MPNet selectors are retained. The launcher checks that they select `mug`
and the right hand, then runs the original generation and refinement pipeline.
The shorter wording "Lift mug with right hand." selected the left hand in the
local check, so the launcher deliberately rejects that result. Other prompts
are accepted with `--prompt`, but must still select mug and right hand.

Rendering runs in batches of eight frames for the 8 GB GPU. This changes render
batching only, not the model weights or sampling steps. Each run gets its own
timestamped directory under `outputs/`; prior results are preserved.

Outputs include:

- `motion/batch0_0_sample0_refined.mp4`: rendered hand/object video.
- `obj_file/batch0_0_sample0/`: a hand mesh and object mesh for each frame.
- `generated_motion.npz`: generated parameters and vertices before the demo's
  visualization recentering, with canonical object geometry/faces.
- `config.yaml` and `run_report.json`: configuration, selected object/hand,
  source revision, seed, elapsed time, and PyTorch peak GPU memory.
- `last_run.json` in this folder points to the latest attempted run and states
  whether it completed successfully.

The generated hand parameters use the upstream 99-value representation:
translation plus 16 joint rotations in 6D form. Object parameters have nine
values. GRAB uses `flat_hand_mean=True`, zero MANO shape coefficients, meters,
and the upstream object rotation convention. These are not directly equivalent
to the GraspXL MANO source's axis-angle/mean-pose convention.

## Environment and assets

The isolated WSL Conda environment is `/home/nmt/miniconda3/envs/text2hoi`:
Python 3.8.20, PyTorch 1.13.0+cu116, torchvision 0.14.0+cu116, PyTorch3D 0.7.2,
and NumPy 1.23.5. See [requirements_runtime.txt](requirements_runtime.txt) for
the compatible dependency pins and [environment_freeze.txt](environment_freeze.txt)
for the installed versions, including the CLIP source revision.

IsaacLab and SOMA environments were not changed. Rendering uses PyTorch3D CUDA;
this demo does not require Vulkan/OpenGL changes.
The launcher adds `/usr/lib/wsl/lib` to its own `LD_LIBRARY_PATH`, because this
older cuDNN build loads the unversioned `libcuda.so` dynamically. No system driver
or global shell configuration was changed.

The unused system CUDA 11.8 toolkit was removed on 2026-09-17 at the user's
request: 54 toolkit/library/profiling packages and the old CUDA apt source/key.
No `/usr/local/cuda*` installation or system `nvcc` remains. The package-manager
consistency checks passed. The removal plan, configuration backups, and purge log
are recorded in [cuda_cleanup_result.json](cuda_cleanup_result.json).

Text2HOI loads its CUDA 11.6 runtime, cuBLAS, and cuDNN from its own `torch/lib`
folder; the shared GPU driver comes from `/usr/lib/wsl/`. Both `env_isaaclab` and
`soma-x` retain PyTorch 2.10.0+cu128 / CUDA 12.8 libraries inside their environments.
GPU matrix multiplication and convolution passed in both environments after the
cleanup, and Text2HOI's compiled PyTorch3D CUDA nearest-neighbor operation passed.
The complete mug demo also passed after removal in
`outputs/20260917T013423Z_seed_42/`: 147 decoded video frames, 147 hand/object mesh
pairs, and finite generated arrays, with 43.36 seconds measured pipeline time.
The Conda environments, Windows GPU driver, WSL driver libraries, and graphics
settings were preserved. A system toolkit is optional unless a future project
needs to compile CUDA code; it is distinct from a Linux GPU driver.

See [system_cuda_after_cleanup.json](system_cuda_after_cleanup.json),
[cuda_cleanup_env_validation.json](cuda_cleanup_env_validation.json), and
[conda_cuda_audit.json](conda_cuda_audit.json). The earlier
[system_cuda_audit.json](system_cuda_audit.json) is retained as the pre-removal
inventory. Windows CUDA paths inherited by WSL were outside this cleanup's scope.

Assets are linked from the user's existing downloads:

| Repository location | Source |
|---|---|
| `checkpoints/grab/` | `/home/nmt/Projects/Text2HOI_Download/Checkpoints/grab` |
| `data/grab/` data files | `/home/nmt/Projects/Text2HOI_Download/Preprocessing/grab` |
| `data/mano/mano_v1_2/models/` | Licensed files in `/home/nmt/Projects/SOMA-X/assets/MANO` |

These are WSL symlinks. Keep the source directories in place. Only the GRAB mug
mesh was prepared; other demo objects need their corresponding meshes.
The 3.97 GB preprocessed training archive is linked but is not loaded by the demo.

The mug came from the existing GRAB contact-mesh ZIP and was simplified with
the upstream algorithm using PyMeshLab 2021.10 to 4,000 vertices / 8,000 faces.
Its sampled vertices match the downloaded `obj.pkl` to a maximum error of
`2.98e-9 m`. See [mesh_preparation.json](mesh_preparation.json).

All five checkpoint state dictionaries loaded with `weights_only=True`, and all
384 tensors were finite. Both object-pickle copies are identical. All downloaded
asset hashes are recorded in [asset_validation.json](asset_validation.json).
CLIP ViT-B/32 and `sentence-transformers/all-mpnet-base-v2` were downloaded into
the normal WSL model caches.

## Migration to Linux storage

The project, downloads, and MANO assets were moved to `/home/nmt/Projects` on
2026-09-17. Python helpers now find sibling projects relative to their own file
location. The Windows launcher starts the Linux copy; its outputs are accessible
at `\\wsl.localhost\Ubuntu\home\nmt\Projects\Text2HOI\_local\text2hoi\outputs`.

The copy left eight former symlinks as empty regular files. These placeholders
were backed up and replaced with links to the migrated downloads and MANO assets.
Nonempty data files and historical run/audit reports were preserved. The helper
and placeholder backups, migrated asset validation, and link targets are recorded
under `migration_backup/`. Windows `:Zone.Identifier` sidecars are not linked into
the model's data directory.

All 12 downloaded asset hashes matched their pre-migration records. The migrated
demo passed in `outputs/20260917T025035Z_seed_42/`: 147 decoded video frames at
30 FPS, 147 hand/object mesh pairs, and finite generated arrays. Its report
confirms `/home/nmt/Projects/Text2HOI` as the source repository. The measured
pipeline time was 44.49 seconds with 1,355 MiB peak allocated GPU memory; this
single demo is an integrity check, not a storage-performance benchmark.

## Scope

A separate GraspXL mug fine-tuning pilot is now available in
[the GraspXL pilot folder](../graspxl_mug/README.md). It has its own data adapter,
checkpoint, generated motion, and IsaacLab replay; the launcher documented here
continues to run the original GRAB demo.

This setup runs the released **GRAB** checkpoint on GRAB's mug. It does not yet
adapt the GraspXL mug, train on GraspXL/SOMA, reproduce paper benchmark scores,
or establish physical grasp stability. GraspXL requires its actual mesh, matching
point data, explicit hand-pose conversion, and the saved evaluation protocol.
