#!/usr/bin/env bash
# Train Text2HOI stages on GRAB from the released preprocessed data.
#
#   bash scripts/train/train_grab.sh <stage> [hydra overrides...]
#
# Stages, in the order the paper's pipeline depends on them:
#   length    text -> sequence length CVAE            (train/train_seq_cvae.py)
#   contact   contact estimator + point encoder       (train/train_contact_estimator.py)
#   texthom   text-to-motion diffusion model          (train/train_texthom.py)
#   refiner   hand refinement network                 (train/train_refiner.py)
#
# Everything a stage writes goes under $RUN (default outputs/grab). Nothing is
# written into checkpoints/grab, which links to the downloaded released weights.
# The upstream length and contact trainers save to their weight_path, so running
# them unmodified would overwrite those released files; this launcher redirects them.
#
# Frozen prerequisite models load from checkpoints/grab (released) by default.
# To chain your own results instead, pass overrides, for example:
#   bash scripts/train/train_grab.sh refiner texthom.weight_path=outputs/grab/texthom/model/texthom_best.pth
#
# Environment knobs: RUN, WANDB_MODE (offline|online|disabled), TEXT2HOI_ENV.
set -euo pipefail

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
[ $# -ge 1 ] || usage 1
stage="$1"; shift
case "$stage" in -h|--help|help) usage 0 ;; esac

source "$(dirname -- "${BASH_SOURCE[0]}")/grab_env.sh"
RUN="${RUN:-outputs/grab}"

# Refuse user overrides that would point a trainable model's save path at the released weights.
for arg in "$@"; do
  case "$arg" in
    seq_cvae.weight_path=checkpoints/*|contact.weight_path=checkpoints/*|pointfeat.weight_path=checkpoints/*)
      if [ "$stage" = length ] || [ "$stage" = contact ]; then
        echo "Refusing '$arg': the $stage trainer saves to that path and would overwrite released weights." >&2
        exit 2
      fi ;;
  esac
done

for required in data/grab/data.npz data/grab/obj.pkl data/grab/text.json data/grab/balance_weights.pkl \
                data/mano/mano_v1_2/models/MANO_RIGHT.pkl data/mano/mano_v1_2/models/MANO_LEFT.pkl; do
  [ -e "$required" ] || { echo "Missing $required. See TRAINING_GRAB.md for the asset links." >&2; exit 3; }
done

case "$stage" in
  length)
    script=train/train_seq_cvae.py
    out="$RUN/seq_cvae"
    stage_args=(dataset.augm=True "seq_cvae.save_root=$out" "seq_cvae.weight_path=$out/seq_cvae.pth") ;;
  contact)
    script=train/train_contact_estimator.py
    out="$RUN/contact_estimator"
    stage_args=(dataset.augm=True
                "contact.save_root=$out" "contact.weight_path=$out/contact_estimator.pth"
                "pointfeat.save_root=$out" "pointfeat.weight_path=$out/pointfeat.pth") ;;
  texthom)
    script=train/train_texthom.py
    out="$RUN/texthom"
    stage_args=("texthom.save_root=$out") ;;
  refiner)
    script=train/train_refiner.py
    out="$RUN/refiner"
    stage_args=("refiner.save_root=$out" 'text=[Drink cup with right hand.]') ;;
  *) echo "Unknown stage '$stage'." >&2; usage 1 ;;
esac

mkdir -p "$out"
log="$out/train_$(date +%Y%m%dT%H%M%S).log"
echo "Stage: $stage | outputs: $out | log: $log | WANDB_MODE=$WANDB_MODE"
python -u "$script" dataset=grab "hydra.run.dir=$out/hydra" "${stage_args[@]}" "$@" 2>&1 | tee "$log"
