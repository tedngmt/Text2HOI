<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# OpenHOI and Text2HOI: GraspXL performance comparison

Prepared 2026-09-16. OpenHOI is the primary candidate; Text2HOI is the baseline.
The goal is to compare generated hand-object motion quality on the local GraspXL
data. This document and the split manifest are preparation artifacts: no model
installation, training, or model evaluation was performed in preparing them.

Update 2026-09-17: Text2HOI was installed in its own WSL environment, and one
released-checkpoint GRAB mug/right-hand demo completed successfully (147 frames).
See [the local setup and result](../text2hoi/INSTALLATION.md). OpenHOI checkpoint
availability remains unresolved. GraspXL adaptation and benchmark evaluation
below are still pending; the GRAB demo is not a GraspXL performance result.

Later on 2026-09-17, a separate [GraspXL mug pilot](../graspxl_mug/README.md)
validated the MANO/object adapter and completed 20 Text2HOI fine-tuning updates.
It uses 10 mug sequences for training and five for sequence-level validation.
The saved object-level benchmark manifest remains unchanged. Because its mug UID
belongs to validation, this pilot checkpoint is ineligible for an untrained-on-mug
evaluation under that protocol. No benchmark performance claim follows from the
pilot's denoising loss or kinematic IsaacLab replay.

## Task and experimental order

1. Start with MANO: object geometry and a simple grasp instruction produce a
   complete right-hand and object motion sequence. Both models receive the same
   available information. Evaluate released checkpoints separately from models
   fine-tuned on GraspXL; record all external pretraining and frozen components.
2. Adapt and evaluate OpenHOI first, then Text2HOI on the saved split. Establish
   common preprocessing and metrics before comparing their results. Freeze the
   protocol using training/validation data; do not tune on the test set.
3. Consider SOMA adaptation as a separate experiment after the MANO comparison.
   Converting a generated MANO motion to SOMA is not native SOMA training. Use
   corresponding reconstructed geometry for comparisons across skeletons.
4. Keep GRAB as a later, separate generalization experiment. Its matched object
   names do not establish identical meshes or equivalent action labels.

OpenHOI is the quality-first hypothesis, not an established winner on this data.
Its authors report better joint error, motion-distribution, and contact-related
metrics than Text2HOI in their evaluations; those scores cannot replace a common
local experiment. See the [OpenHOI paper, Tables 2 and A2](https://arxiv.org/html/2505.18947v2),
[OpenHOI code](https://github.com/Zhenhao-Zhang/OpenHOI), and
[Text2HOI code](https://github.com/JunukCha/Text2HOI).

## Saved dataset split

Use [graspxl_openhoi_text2hoi_v1.json](graspxl_openhoi_text2hoi_v1.json).
Paths in its records are relative to their respective sibling dataset packages.

| Split | Source mesh IDs | Object-size entries | Motions | Frames |
|---|---:|---:|---:|---:|
| Training | 32 | 35 | 488 | 75,640 |
| Validation | 8 | 8 | 120 | 18,600 |
| Test | 8 | 8 | 141 | 21,855 |
| Total | 48 | 51 | 749 | 116,095 |

The deterministic assignment sorts unique source UIDs by the lowercase SHA256
of UTF-8 `graspxl-openhoi-text2hoi-v1:<uid>`. The first 32 UIDs are training,
the next 8 validation, and the last 8 test. It was selected before model results.
Keep every size variant, sequence window, augmentation, and paired MANO/SOMA
representation of a UID in the same split. Preserve this manifest across both
models; any future alternative split must have a new experiment identity.

Preparation checks covered counts, unique pair paths, source-manifest agreement,
existence of all referenced MANO/SOMA motions and meshes, and absence of
cross-split duplicate motion/mesh hashes recorded in source metadata. Source
manifest and pair-list hashes are saved. Raw motion hashes were not recomputed.
Near-duplicate geometry and external pretraining overlap remain unaudited.
Describe the test objects as **held out from GraspXL training/fine-tuning** until
that audit supports a stronger claim. This is not a held-out-category benchmark.

## Common data and conditioning rules

- Retain native object scale and shared hand/object coordinates. The source uses
  meters, the right MANO hand, `use_pca=False`, `flat_hand_mean=False`, and zero
  shape coefficients. Verify each adapter reproduces the source geometry before
  training; explicitly handle the absent left hand.
- Both models must use the same frame selection, sequence horizon, and temporal
  resampling policy. Source FPS is unknown. A selected playback/training rate is
  an assumption, not recovered timing. Do not report physical speed or jerk
  without defining that assumption.
- Use truthful source-object grasp labels, including proxy objects; do not infer
  drinking, cutting, or other GRAB activities from matching object names. Freeze
  the same prompt mapping for both models using metadata and training data.
- Future hand/object trajectories, target grasp poses, and target contact maps
  must not enter inference for this generation task. Ground-truth contact labels
  can supervise training; test-time contact/affordance conditions must be
  predicted from permitted inputs. Evaluate any oracle-guided task separately.
- Derive normalization and learned preprocessing from training data only. Use
  equal hyperparameter-search budgets and the same validation selection rule;
  model-specific learning rates are allowed. Record data exposure and compute.

## Evaluation and reporting

| Aspect | Common measurement and interpretation |
|---|---|
| Reference accuracy | Hand-joint/wrist position error and object translation/rotation error, with units and coordinates stated. Report scene/object-relative placement separately from wrist-relative articulation; root alignment can hide misplaced grasps. |
| Contact quality | Contact distances, penetration depth/rate, and contact sliding during detected contact. Freeze mesh handling, thresholds, and contact-phase detection on validation data. Report invalid/non-watertight mesh coverage; unsigned distance alone does not measure penetration. |
| Motion quality | Discontinuities and frame-based differences with the timing policy stated. Use one shared feature extractor, normalization, and sample-count protocol for any motion FID; paper-specific FID numbers are not directly comparable. |
| Diversity | Multiple outputs per object/prompt, alongside contact validity. A plausible alternative grasp can differ substantially from the single recorded trajectory. |
| Physical stability | Separate dynamic lift/hold evaluation with identical mass, friction, collisions, and hand control. Kinematically moving the object along a predicted path does not establish grasp success. |
| Resource cost | Peak VRAM, generation time including refinement, training time, parameter count, and exact hardware/configuration. |

Start with five generated samples per test condition and the same sampling
budget for both models. Report averages; label best-of-five accuracy separately.
Aggregate within each source UID first, then average across UIDs so objects with
more sequences do not dominate. Estimate uncertainty at the object level,
acknowledging only eight test UIDs. Aim for three training seeds if resources
allow; label a single-seed pilot accordingly. Include representative failures
and synchronized viewer comparisons, not only selected successful examples.

Before final evaluation, freeze the contact thresholds, feature extractor,
checkpoint-selection rule, prompt mapping, output horizon, and any physics
protocol. Prefer contact validity and motion quality jointly when selecting a
model; do not declare a winner from reference joint error alone. These synthetic
motions provide a GraspXL reference, not independent proof of human realism.

## Execution status and next steps

- Done: local package inventory and paired split preparation.
- Next: isolate OpenHOI dependencies, record its source revision and checkpoint
  provenance, validate a GraspXL adapter, then run a small training/validation
  smoke check before a full experiment. Repeat for Text2HOI using the same split.
- Pending: source FPS resolution or declared timing policy, near-duplicate and
  pretraining-overlap audits, metric implementation/calibration, and all scores.
- Hardware: the local GPU has 8 GB VRAM. OpenHOI's README reports A100 80 GB
  training hardware; a full local training recipe has not been verified. Record
  any reduced configuration explicitly. Larger compute, if available later,
  should use the same dataset/evaluation protocol.

The dataset packages and existing viewer are read-only inputs to this benchmark.
