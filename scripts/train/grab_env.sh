# Source this file; do not execute it.
# Prepares the local WSL shell for Text2HOI GRAB training and demos.
#   source scripts/train/grab_env.sh

_t2h_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Conda's activate scripts read unset variables, so relax "set -u" around them.
_t2h_restore_u=0
case $- in *u*) _t2h_restore_u=1; set +u ;; esac
source "${CONDA_SH:-/home/nmt/miniconda3/etc/profile.d/conda.sh}"
conda activate "${TEXT2HOI_ENV:-text2hoi}"
[ "$_t2h_restore_u" = 1 ] && set -u

# cuDNN 8 in this PyTorch 1.13 build loads the unversioned libcuda.so, which WSL
# keeps outside the default loader path.
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# The trainers call wandb.login(). "offline" keeps loss curves on disk under wandb/
# without an account; set WANDB_MODE=online after "wandb login" to upload them.
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$_t2h_repo"
unset _t2h_restore_u
