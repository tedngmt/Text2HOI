# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Control sustained training load with idle time, without changing updates."""

import json
import time
from contextlib import suppress
from pathlib import Path

SETTINGS = Path(__file__).resolve().with_name("cooling_settings.json")
ALLOWED_DUTIES = (25, 50, 75, 100)


def read_duty(path=SETTINGS):
    if not path.exists():
        return 100
    duty = json.loads(path.read_text())["active_percent"]
    if duty not in ALLOWED_DUTIES:
        raise ValueError("active_percent must be 25, 50, 75, or 100")
    return duty


def set_duty(duty, path=SETTINGS):
    if duty not in ALLOWED_DUTIES:
        raise ValueError("active_percent must be 25, 50, 75, or 100")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"active_percent": duty}, indent=2) + "\n")
    temporary.replace(path)


def idle_seconds(active_seconds, duty):
    if duty not in ALLOWED_DUTIES:
        raise ValueError("Invalid active percentage")
    return max(0.0, active_seconds) * (100 / duty - 1)


class CoolingThrottle:
    def __init__(self, path=SETTINGS):
        self.path = path
        self.duty = 100

    def wait(self, active_seconds):
        # Retain the last valid setting if a manually edited file is invalid.
        with suppress(OSError, ValueError, KeyError):
            self.duty = read_duty(self.path)
        delay = idle_seconds(active_seconds, self.duty)
        if delay:
            time.sleep(delay)
        return delay
