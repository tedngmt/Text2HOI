#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
source /home/nmt/miniconda3/etc/profile.d/conda.sh
conda activate text2hoi
# cuDNN 8 dynamically loads libcuda.so, whose unversioned name is not in ldconfig.
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python -u "$script_dir/run_mug.py" "$@"
