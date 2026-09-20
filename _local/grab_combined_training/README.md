# GRAB mug: Text2HOI refinement and physical tracking

This pipeline trains two separate models: the pretrained Text2HOI **refiner**
adapts both hands in recorded GRAB motions, then a new PPO policy learns to track
the right-hand references with the GraspXL articulated hand. Before PPO training,
a separate fitting step puts finger rotations within the simulated hand's joint
limits. The Text2HOI text-to-motion generator stays frozen. The stages exchange
saved motion files; gradients do not pass between them.

The simulation stage uses **RaiSim through GraspXL**, running headlessly in the
`graspxl` Conda environment. Its files now live in the Text2HOI repository (moved from the IsaacLab workspace on 2026-09-20), and this
stage does not use IsaacLab's physics engine. The PPO policy starts from a zero
residual over the reference PD controller. It does not load or transfer the
released GraspXL demonstration policy.

The full 100-epoch run has **not started**. Short development checks are described
below; they do not establish grasp success or policy quality.

From Ubuntu, launch the complete experiment with:

```bash
cd /home/nmt/Projects/Text2HOI
source /home/nmt/miniconda3/etc/profile.d/conda.sh
conda activate graspxl
python _local/grab_combined_training/run_pipeline.py --epochs 100
```

The launcher automatically runs refinement in `text2hoi` and preparation,
joint-limit fitting, and policy training in `graspxl`. It sets the required
library paths, checks GPU access and the compiled tracking environment, and
writes subprocess output to stage logs.
Its default run directory is `_local/grab_combined_training/runs/grab_mug_100`.
Use `--output_dir /absolute/path/to/new_run` for another experiment.

`--epochs 100` means **100 dataset passes for each trainable stage**: 100 refiner
epochs followed by 100 PPO reference-window epochs. Each policy epoch visits all
supported training windows. The separate `ppo_passes: 4` setting permits up to
four optimization passes over each collected rollout; these are not the 100
dataset epochs. A KL limit can stop rollout optimization before all four passes.

Settings are in [config.json](config.json). Current defaults use 96-frame windows,
refiner learning rate `1e-5`, policy learning rate `1e-4`, eight parallel physics
environments, and validation every five epochs, plus the first and final epoch.
Both stages complete the requested epoch budget and preserve the best acceptable
validation checkpoint. **100 epochs cannot guarantee absence of overfitting.**
The selected checkpoint may come from an earlier epoch or remain at epoch 0.

All 44 GRAB mug recordings are included in preparation and reference export.
They contain 13,171 frames at 30 FPS, obtained by retaining every fourth frame
from the original 120 FPS recordings. Whole subjects are assigned before making
windows:

| Split | Subjects | Refiner clips | PPO clips | PPO episodes |
|---|---|---:|---:|---:|
| Training | s1, s2, s3, s4, s6, s7 | 28 | 25 | 31 |
| Validation | s5, s8 | 9 | 8 | 9 |
| Test | s9, s10 | 7 | 7 | 7 |

The right-hand physics stage therefore uses 40 clips and 47 supported episodes.
At the configured 96-frame window size, training has 57 windows covering 4,205
planned transitions; validation has 18 windows covering 1,168 planned transitions.
Failures can terminate a rollout before all planned transitions are completed.

Physics resets at the beginning of every window, which contains at most 96
transitions. Later windows can start mid-grasp or with the mug already airborne,
even though the enclosing episode starts from a supported pickup. Consequently,
the lift-completion metric measures a reference-height threshold within each
window; it does not measure uninterrupted full-recording grasp-and-lift success.

Only the training group supplies gradients. PPO uses right-hand-only contact
phases with a preceding reach and a supported initial mug position. Four clips
have no eligible episodes: `s1__mug_offhand_1`, `s2__mug_pass_1`,
`s3__mug_offhand_1`, and `s8__mug_offhand_1`. These exclusions follow recorded
contact and initial-height rules, independent of policy performance. Their full
motions remain in the refiner dataset. The physics scene contains the right hand,
mug, and table; drink/pass phases do not include a mouth, second hand, or recipient.

These are **holdouts from local adaptation**. All 44 local mug trajectories match
the released Text2HOI preprocessing, as recorded in
[pretraining_overlap.json](pretraining_overlap.json). Exposure during released
checkpoint pretraining cannot be excluded. The older `all44_v2` refiner, trained
on every recording locally, is not used to initialize this experiment.

The refiner keeps the recorded object trajectory and personalized MANO anatomy.
Its losses penalize mug penetration, changes to source contacts, displacement,
and abrupt corrections. Translation corrections are bounded to 15 mm and
rotation-6D component corrections to 0.15. Current defaults give the contact loss
weight 1.0. A candidate checkpoint must improve the validation objective and
penetration while retaining at least **baseline contact retention minus 0.02**
on the 0–1 scale: at most two percentage points of loss. Every rejected candidate
records its reason. The initial pretrained model remains the fallback when no
trained candidate qualifies. Full-run test metrics are computed only after
checkpoint selection.

PPO controls a dynamic hand and mug; it does not prescribe the mug's future
trajectory. Reference joint velocities supply feedforward targets to reduce lag
during fast recorded motions. The learned corrections are bounded to 3 cm of
wrist translation and 0.2 rad of root/finger rotation. This controller makes
local corrections around the supplied motion; it does not plan a new reach
around arbitrary obstacles. Training randomizes mug mass/inertia by 0.8–1.2
times nominal, friction over 0.6–1.0, and PD gains by 0.9–1.1. Whole reference
motions receive up to ±2.5 cm of table-plane translation and ±0.15 rad of yaw. Validation/test use fixed
settings. Observation running means and variances update **only from training
rollouts** and are frozen for evaluation. The policy uses weight decay, gradient
clipping, a KL limit, and learning-rate reduction on validation plateaus. Its
selection score includes failures and table penetration alongside reward. A
candidate must also avoid additional failed validation windows, a contact-recall
drop greater than two percentage points, a maximum table- or object-penetration
increase greater than 2 mm, or lower lift completion relative to the zero-residual
reference-PD baseline. Failed windows retain their full planned contact and lift
targets in the metric denominators, so early failure cannot hide missed targets.
The baseline policy at epoch 0 remains available if no trained policy qualifies.
Tracking, contact, penetration, and lift metrics remain available for interpretation.

Every clip uses the same native GRAB mug geometry. Its collision approximation
has **128 CoACD convex parts**, preserving the cup cavity and handle opening more
closely than a single hull. Collision probes and opening checks are saved under
`prepared/assets/mug/`; the approximation can still differ from the native mesh.
Nominal mug mass is an experiment setting of 0.25 kg, not a measured GRAB value.
The simulated tabletop now matches the captured dimensions:
**0.450015 m × 0.540018 m × 0.005481 m**, with the recorded per-clip center and
yaw, an upper surface at z = 0.5 m, and no legs. It is a static box approximation;
the table does not replay capture-frame motion.

The GraspXL simulation hand has generic geometry. Initial transfer of MANO
rotations differs from personalized hand joints by approximately 4–11 mm in
per-clip means, with an individual discrepancy up to 17.8 mm. The subsequent
`retarget.py` stage fits finger joint positions using 100 projected Adam steps
within the original URDF limits. Wrist motion, object motion, and source-frame
indices remain unchanged; subject-specific anatomy is not fitted.

On the prepared references, the fitting stage's mean joint error relative to
the **unbounded generic-hand targets** is approximately 0.267 mm, compared with
1.382 mm after simply clipping the angles. Rare individual errors reach 36.6 mm.
These fitting errors have a different reference from the personalized-hand
discrepancies above. The fitted motions satisfy finger limits but are not
guaranteed to preserve contacts or avoid collisions. The `prepared_feasible`
folder name refers to these joint constraints, not proven physical grasp success.

The launcher writes the following artifacts within the selected run directory:

| Location | Contents |
|---|---|
| `pipeline_status.json`, `*.log` | Stage status, exact subprocess commands, and logs |
| `refiner/best.pth`, `refiner/last.pth` | Selected and most recent refinement checkpoints |
| `refiner/history.json`, `evaluation_*.json` | Training curves, validation metrics, and selection decisions |
| `refiner/references/`, `refiner/reference_manifest.json` | All 44 selected-model motion exports and provenance |
| `prepared_refined/` | Decoded selected-refiner references before joint-limit fitting |
| `prepared_feasible/`, `prepared_feasible/retarget_report.json` | Fitted references and captured tabletop poses used by PPO |
| `policy/best.pt`, `policy/last.pt` | Selected and most recent policy checkpoints |
| `policy/report.json`, `policy/best_validation.json`, `policy/test_final.json` | Policy status and evaluation records |

Shared source preparation is in `_local/grab_combined_training/prepared_source/`.
The cached mug collision assets remain in `prepared/assets/mug/`. Each run then
creates its own `prepared_refined/` and `prepared_feasible/` outputs.

Resume the same run after an interruption with:

```bash
python _local/grab_combined_training/run_pipeline.py --epochs 100 --resume
```

Supply the same `--output_dir` when using a custom location. Keep the original
configuration and epoch target. Completed stages are skipped. Checkpoints restore
model parameters, optimizer moments, scheduler state, saved random states, and
the policy's observation-normalization moments. Refiner checkpoints are saved
at validation epochs, normally 1, 5, 10, and so on, plus the final epoch; policy
checkpoints are saved every epoch. A refiner interruption can therefore require
repeating up to four completed epochs plus the interrupted epoch. Resume restores
the saved checkpoint's log history and archives later or partially written log
rows under `refiner/resume_archives/`. If refinement already reached the target
epoch but export was interrupted, resume reruns export without further training.

Resume occurs at saved epoch boundaries; it does not restore a paused simulator
timestep. A run lock prevents two pipeline processes from writing the same
directory. For stage-level development checks, the refiner's `--skip_test` flag
still exports all 44 motions while omitting test loss calculations and metric
reports; the complete pipeline's final evaluation uses its selected checkpoints.

Development validation so far covers the refiner's finite updates, checkpoint
resume, full training-frame coverage, all-clip export, and preservation of object
trajectories/timing. A three-epoch contact-guard check rejected every trained
candidate because contact retention fell below its threshold, correctly retaining
epoch 0. See `runs/smoke_refiner_contact_guard/smoke_verification.json` and
[contact_guard_check.json](contact_guard_check.json).

Simulator mechanics checks cover table collision, dynamic state behavior, and
reference-pose conversion; [simulator_verification.json](simulator_verification.json)
records their scope. The final integrated pilot used the contact-guard-selected
refiner exports, fitted references, and captured tabletop. It completed two PPO
epochs, resuming from epoch 1 to epoch 2, with 162 optimizer updates and 8,030
training transitions. Each epoch visited all 57 training windows and evaluated
all 18 validation windows. Model parameters, optimizer counters, and observation
moments continued correctly after resume; all policy parameters remained finite.
Test evaluation was disabled. See
[runs/policy_final_pilot/verification.json](runs/policy_final_pilot/verification.json)
and [setup_verification.json](setup_verification.json).

Both short-run PPO candidates failed the physical validation safeguards, so the
selected policy remains the epoch-0 reference controller. Its validation contact
recall was 53.4%, and 4 of 11 planned lift windows met the lift threshold without
a terminal failure. These are window metrics, not full-recording grasp success.
This check establishes that training, selection, and recovery work; improved
grasp quality has not been established. Earlier `policy_pilot` and smoke logs
describe superseded references or evaluation protocols.

The refiner also successfully resumed an already-completed epoch-3 checkpoint
to export all 44 motions without further training or test scoring. Its exported
object trajectories and timing were unchanged; all 13,171 fitted frames satisfied
finger limits. Failed-window metric regressions and refiner-recovery regressions
passed, as did targeted Ruff and Python compilation checks. The repository-wide
`./isaaclab.sh -f` attempt could not run its hooks: `pre_commit` was absent and its
automatic installation failed because sandbox network access was unavailable.
Details are in `precommit_check.log`.

The full 100-epoch run has not started. Neither these short checks nor the planned
training budget establish reliable grasp performance.

Video exports of this pilot are in [share/](share/). The combined
`GRAB_Mug_Pilot_Comparison.mp4` shows original recorded GRAB motion, the selected
epoch-0 reference controller, and the latest epoch-2 PPO controller side by side.
Individual videos cover eight validation recordings and all 18 evaluated windows.
The top row uses a shared fixed world camera with the table visible; the lower
row follows each mug's handle while preserving its actual tilt. Failed rollouts
hold the last valid state with a red label, and title cards mark physics resets
between windows. These are recorded simulator states, not mesh playback presented
as successful physics. The original column uses personalized MANO geometry;
the two physics columns use the actual GraspXL hand's segmented visual meshes.

`export_rollouts.py` reproduces both checkpoint evaluations exactly, with no
training or test-set evaluation. `render_results.py` renders those saved states,
and `package_result_videos.py` checks video metadata and creates the shareable
MP4s and ZIP. The captions deliberately identify this two-epoch pilot and must
be updated before using the renderer for a later experiment.
