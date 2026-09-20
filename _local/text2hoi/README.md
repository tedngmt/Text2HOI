<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Text2HOI mug feasibility check

**Current setup:** see [INSTALLATION.md](INSTALLATION.md) for the isolated WSL
environment, asset links, and mug launcher prepared on 2026-09-17. The notes
below record the earlier source/download feasibility check, before installation.

Checked 2026-09-16 against the [official repository](https://github.com/JunukCha/Text2HOI).
This is a source/download check, not completed installation or inference.

## Verified

- The official GRAB checkpoint folder lists `texthom.pth`, `refiner.pth`,
  `seq_cvae.pth`, `pointfeat.pth`, and `contact_estimator.pth`.
- `seq_cvae.pth` was downloaded to `checkpoints/grab/` (2,502,297 bytes).
  CPU loading with `torch.load(weights_only=True)` passed; its `model` state
  contains eight finite tensors. See `checkpoint_download_check.json`.
  The other four model files have not been downloaded or loaded.
- The official GRAB object-data folder lists `obj.pkl`.
- The official PyTorch3D wheel index responds and lists the requested
  `pytorch3d-0.7.2-cp38-cp38-linux_x86_64.whl`. The wheel was not installed.
- Source code supports the `mug` object and right-hand MANO generation from
  text plus object geometry, without supplied future object motion or hand frames.

## Official assets for the first GRAB mug run

- [Five GRAB checkpoints](https://drive.google.com/drive/folders/1GFZkjzL37jO2BgujVGtJ-ezY97PNO4FU)
- [GRAB object pickle](https://drive.google.com/drive/folders/1G9KZMcxV8EMKjMquQTsm0gWv8s8E9AVm)
- The matching processed mug mesh; the existing original GRAB object archive
  can supply the source mesh for preprocessing. Verify sampled-point/mesh
  correspondence against `obj.pkl`.
- Existing licensed `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` in
  `C:/Linux/SOMA-X/assets/MANO/`; the demo initializes both hands.
- CLIP `ViT-B/32` and `sentence-transformers/all-mpnet-base-v2`, downloaded by
  the model-loading code when required.

## Setup and test boundaries

Use an isolated Conda environment. The official installer targets Python 3.8,
PyTorch 1.13.0/CUDA 11.6, PyTorch3D 0.7.2, and NumPy 1.23.5. Other requirements
are unpinned and include both `hydra-core` and the unrelated `hydra` package;
review compatibility instead of executing the installation script unchanged.
The user's IsaacLab and SOMA environments have not been modified.

After installation and asset preparation, run one GRAB prompt instead of the
supplied shell script, which launches examples for three datasets:

```bash
python demo/demo.py dataset=grab \
  '+test_text=[Lift a mug with the right hand.]' \
  +nsamples=1 save_obj=True \
  hydra.output_subdir=null \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled
```

This is the proposed test command, not a command already run successfully.
The demo selects object/hand by text similarity; verify that it selects `mug`
and the right hand because `coffeemug` and `cup` also appear in its candidates.
Measure peak memory and rendering requirements on the 8 GB GPU during this run.

GraspXL support still requires the actual GraspXL mug mesh and aligned point
cloud in the input format. Preserve metric scale, object transforms, and the
source MANO mean-pose convention when comparing generated/reference geometry.
No model retraining is inherently required just to attempt a new-object input.

Training code is released, but the complete paper evaluation pipeline has not
been verified; an [evaluation-code request](https://github.com/JunukCha/Text2HOI/issues/18)
exists. Build and validate the common contact/penetration/motion metrics before
claiming benchmark reproduction or a performance comparison with other models.
