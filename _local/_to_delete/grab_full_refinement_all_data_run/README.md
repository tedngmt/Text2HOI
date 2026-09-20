# All-GRAB Text2HOI refiner run

This run follows the published Text2HOI **refiner training objective and schedule**.
It does not train GraspXL, RaiSim, Isaac Lab, or a reinforcement-learning policy.

## Protocol

- All 1,335 released GRAB training sequences; 51 objects.
- Upstream `iteration=3000` and GRAB `data_num=1335`, `text_num=276` produce
  `ceil((3000 / (1335 / 276)) / 50) * 50 = 650` epochs.
- AdamW, learning rate 0.0001, default weight decay 0.01; FP32 training.
- Effective batch 64, balanced sampling with replacement, drop incomplete batch:
  20 optimizer updates per epoch, 13,000 updates total.
- Eight microbatches of eight sequences per update fit the 8 GB GPU.
  Reconstruction/contact losses use their original batch averages. Penetration
  numerator gradients are accumulated separately for each hand and divided by
  the entire effective batch's interior-point count. Averaging microbatch
  penetration means would change the objective; this implementation avoids that.
- 150-frame windows, official standard zero-beta MANO, no data augmentation.
- Official loss weights: reconstruction 1, penetration 1, L1 contact 5.
- Refiner initialized according to upstream `zero_initialized=True`.
  Released diffusion generator, CLIP, and point encoder remain frozen.
- Checkpoint selected by lowest **training** loss, as upstream. There is no
  held-out evaluation or demonstrated protection against overfitting.

Hardware/operational adaptations: zero data-loader workers, gradient accumulation,
local logs instead of W&B, resumable optimizer/RNG checkpoints each epoch, and
video export after training instead of periodic generated-text demo rendering.
Microbatching changes stochastic RNG ordering and floating-point reduction order;
this is not a bitwise reproduction of a physical batch of 64.

Official sources:
- https://github.com/JunukCha/Text2HOI/blob/main/train/train_refiner.py
- https://github.com/JunukCha/Text2HOI/blob/main/configs/refiner/refiner.yaml
- https://github.com/JunukCha/Text2HOI/blob/main/configs/dataset/grab.yaml

## Important distinction from the earlier mug experiment

The earlier mug experiment fine-tuned released weights with a custom native-mesh
SDF cleanup loss and personalized MANO geometry. This run uses the published
coarse-diffusion-to-GRAB training objective and standard MANO shape. It is not an
extension of those custom losses.

Videos apply the resulting refiner to recorded motions, with the original object
trajectory unchanged, so both columns can be compared directly. This application
differs from the coarse generated inputs used during official training. The videos
are in-sample cleanup visualizations, not an official text-generation benchmark
or proof that motion/contact accuracy improved.

The released preprocessing trims each sequence to its object-contact interval.
Exports cover all released sequences at 30 FPS, including their entire retained
interval, but not the discarded lead-in and trailing frames of the raw archives.
Both video columns use the same standard MANO shape; they are not SOMA comparisons.

The renderer's required camera/EGL helpers are bundled in `render_helpers/`.
They were recovered from commit `20a903478^` after the older comparison folder
was removed, so export no longer depends on that older experiment's directory.

## Running and monitoring

### Lower sustained load in a warm room

```bash
cd /home/nmt/Projects/Text2HOI/_local/grab_full_refinement
/home/nmt/miniconda3/envs/text2hoi/bin/python control.py cool  # 50% active time
/home/nmt/miniconda3/envs/text2hoi/bin/python control.py eco   # 25% active time
/home/nmt/miniconda3/envs/text2hoi/bin/python control.py full  # no extra idle time
```

Choose one mode. Settings persist in `cooling_settings.json` and are reread after
every training microbatch, so changes work while the updated trainer is running.
They do not start a stopped run: use `control.py resume` when ready.
Cool inserts roughly one second of idle time per second of computation; Eco
inserts three. Expect approximately 2× or 4× the remaining training time,
respectively, although hardware throttling and other work affect actual timings.

The GPU is synchronized before sleeping. Model weights, optimizer, sampling,
batch size, learning rate, and checkpoint resume protocol are unchanged.
This lowers sustained load; it is not an instantaneous utilization, power, or
temperature cap, and memory remains allocated while idling. This setting applies
to training, not the later video-rendering stage.

From Ubuntu, use these commands (no Conda activation needed):

```bash
cd /home/nmt/Projects/Text2HOI/_local/grab_full_refinement
/home/nmt/miniconda3/envs/text2hoi/bin/python control.py status
/home/nmt/miniconda3/envs/text2hoi/bin/python control.py pause
/home/nmt/miniconda3/envs/text2hoi/bin/python control.py resume
```

`pause` stops the pipeline and releases GPU memory. It keeps the checkpoint from
the latest completed epoch; the unfinished epoch is repeated on resume. It does
not save a new mid-epoch checkpoint. Confirm `status` says NOT RUNNING before
shutting down. After restarting Ubuntu, `resume` launches the job in the background.
The epoch numbers in logs are one-based; checkpoint epoch fields are zero-based.
During video export, completed videos are verified and skipped on resume.

The pipeline runs training first, then automatically exports per-object videos.
It resumes from the latest complete epoch when restarted. Only one pipeline
process can hold its lock.

```bash
/home/nmt/miniconda3/envs/text2hoi/bin/python \
  /home/nmt/Projects/Text2HOI/_local/grab_full_refinement/run_pipeline.py
```

- `pipeline_status.json`: running stage / failure / completion.
- `training.log`, `run_official/steps.jsonl`: training progress and timings.
- `run_official/latest.pth`: model, optimizer, and RNG state for resume.
- `run_official/best.pth`: selected model for export.
- `run_official/training_complete.json`: written only after all 650 epochs.
- `share/<object>/*.mp4`: recorded-versus-refined comparisons, overview and
  object-follow views, 1280 × 800, 30 FPS. Mug detail views follow the handle.
- `share/manifest.json`: per-clip provenance, checksums, and completion flag.
- `export_smoke*`: exporter tests using **released weights**, not this new run.

Do not run another heavy GPU workload concurrently when comparing timing.

## Measurements before the full run

- Physical batch 64: 12.91 GiB peak allocation, 62.3 seconds for its first update;
  this spilled beyond dedicated GPU memory.
- Microbatch 8, effective batch 64: 2.82 GiB peak allocation, 8.69 seconds for
  the second update; approximately 31.4 training hours if sustained.
- Microbatch 16, effective batch 64: 4.64 GiB peak allocation, 11.54 seconds for
  the second update. Chose eight for the full run.
- Refiner input preparation plus forward pass on one 150-frame airplane window:
  20.2 ms average over 30 repetitions after five warmups, approximately 7,427
  motion frames/second. This excludes diffusion generation, disk access, final
  mesh reconstruction, and video rendering. It is batched sequence throughput,
  not an interactive application's guaranteed FPS.

`check_accumulation.py` verified loss and SGD updates against an unsplit batch
with unequal interior-point counts and empty groups. Short GPU training probes
completed; the export smoke test was encoded and checked with ffprobe.
