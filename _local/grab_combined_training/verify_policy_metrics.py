# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regress failed-window accounting and physical checkpoint-selection safeguards.

The mock reproduces the vector wrapper's terminal reset: its state is already a
reset state when ``done`` is returned. Planned reference contacts and lifts must
therefore remain in the denominators even when no target frame was completed.
Run with GraspXL's Python; RaiSim and CUDA are not needed for these checks.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


class FailureImmediately:
    """Reproduce an environment that fails and automatically resets on its first step."""

    num_envs = 1

    def __init__(self, observation_size: int):
        self.observation = np.zeros((1, observation_size), np.float32)
        self.state = np.zeros((1, 199), np.float32)
        self.reward = np.array([-10.0], np.float32)
        self.done = np.array([True])

    def reset(self, sequences: list[dict], seed: int, dt: float) -> None:
        pass

    def targets(self, sequences: list[dict], frame: int) -> None:
        pass

    def step(self, actions: np.ndarray) -> None:
        pass


def suite(module) -> unittest.TestSuite:
    class PolicyMetricRegression(unittest.TestCase):
        def test_early_failure_preserves_future_contact_and_lift_targets(self):
            object_pose = np.zeros((6, 7), np.float32)
            object_pose[:, 2] = [0.5, 0.5, 0.5, 0.6, 0.8, 0.9]
            object_pose[:, 3] = 1
            reference = {
                "hand_qpos": np.zeros((6, 51), np.float32),
                "object_pose": object_pose,
                "joints": np.zeros((6, 21, 3), np.float32),
                "contact": np.array([False, False, False, True, True, True]),
                "table_dimensions": np.array([0.45, 0.54, 0.00548], np.float32),
                "table_pose": np.array([1, 0, 0.49726, 1, 0, 0, 0], np.float32),
            }
            rollout, rows = module.collect(
                FailureImmediately(module.OBSERVATION_SIZE),
                [{"clip_id": "failure_before_grasp", "start": 0, "end": 6}],
                {"failure_before_grasp": reference},
                None,
                module.ObservationMoments(module.OBSERVATION_SIZE),
                np.random.default_rng(42),
                torch.device("cpu"),
                {"environment": {"control_dt": 1 / 30}, "gamma": 0.99, "gae_lambda": 0.95},
                False,
            )
            self.assertTrue(rows[0]["failed"])
            self.assertEqual(rows[0]["completed_frames"], 0)
            self.assertEqual(rows[0]["expected_contact_frames"], 3)
            self.assertAlmostEqual(rows[0]["reference_lift_max_m"], 0.4, places=6)
            summary = module.summarize(rows)
            self.assertEqual(summary["planned_frames"], 5)
            self.assertEqual(summary["lift_windows"], 1)
            self.assertEqual(summary["lift_completion_fraction"], 0)
            self.assertEqual(summary["contact_recall"], 0)
            self.assertEqual(summary["failure_rate"], 1)
            np.testing.assert_allclose(rollout["returns"], [[-10]])

        @staticmethod
        def baseline() -> dict:
            return {
                "failure_rate": 0.1,
                "contact_recall": 0.8,
                "max_table_penetration_m": 0.001,
                "max_object_penetration_m": 0.003,
                "lift_completion_fraction": 0.7,
            }

        def test_rejects_more_object_penetration_despite_unchanged_other_metrics(self):
            baseline = self.baseline()
            candidate = dict(baseline, max_object_penetration_m=0.006)
            self.assertIn(
                "maximum_hand_object_penetration_worsened_more_than_two_mm",
                module.selection_guard(candidate, baseline),
            )

        def test_rejects_fewer_completed_lifts_despite_unchanged_other_metrics(self):
            baseline = self.baseline()
            candidate = dict(baseline, lift_completion_fraction=0.6)
            self.assertIn(
                "fewer_planned_lifts_completed_than_reference_controller",
                module.selection_guard(candidate, baseline),
            )

        def test_equally_physical_candidate_remains_eligible(self):
            baseline = self.baseline()
            self.assertEqual(module.selection_guard(baseline.copy(), baseline), [])

    return unittest.defaultTestLoader.loadTestsFromTestCase(PolicyMetricRegression)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, default=Path(__file__).with_name("train_policy.py"))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("policy_metrics_under_test", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite(module))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
