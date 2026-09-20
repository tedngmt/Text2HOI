# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Verify live cooling settings and idle scheduling without running training."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from cooling import CoolingThrottle, idle_seconds, read_duty, set_duty

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "cooling.json"
    throttle = CoolingThrottle(path)
    assert read_duty(path) == 100
    assert idle_seconds(2, 100) == 0
    with patch("cooling.time.sleep") as sleep:
        set_duty(50, path)
        assert throttle.wait(2) == 2
        sleep.assert_called_once_with(2)
        set_duty(25, path)
        assert throttle.wait(2) == 6
        path.write_text("invalid")
        assert throttle.wait(2) == 6
        set_duty(100, path)
        assert throttle.wait(2) == 0
        assert sleep.call_count == 3
    try:
        set_duty(0, path)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid duty accepted")
print("PASS: live modes, idle timing, invalid-setting fallback; no training or checkpoint changes")
