# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check resume/export boundaries against a saved checkpoint without GPU training."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from train_refiner import RefinerExperiment


class RefinerResumeChecks(unittest.TestCase):
    """Exercise actual checkpoint restore and run control using inert model slots."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.local = Path(__file__).resolve().parent
        cls.source = cls.local / "runs" / "smoke_refiner_contact_guard" / "last.pth"
        cls.saved = torch.load(cls.source, map_location="cpu")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="refiner_resume_check_")
        self.addCleanup(self.temporary.cleanup)
        experiment = RefinerExperiment.__new__(RefinerExperiment)
        experiment.args = SimpleNamespace(epochs=int(self.saved["epoch"]), skip_export=False, skip_test=True)
        experiment.output = Path(self.temporary.name)
        experiment.report = dict(self.saved["config"])
        experiment.model = Mock()
        experiment.optimizer = Mock()
        experiment.scheduler = Mock()
        experiment.rng = np.random.default_rng(19)
        experiment.digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        experiment.write_json = lambda path, value: path.write_text(json.dumps(value, indent=2) + "\n")
        experiment.write_report = Mock()
        experiment.started = time.monotonic()
        experiment.source_weights = Path(experiment.report["source_checkpoint"])
        experiment.source_hash = experiment.report["source_checkpoint_sha256"]
        experiment.generator_weights = Path(experiment.report["generator_checkpoint"])
        experiment.generator_hash = experiment.report["generator_checkpoint_sha256"]
        experiment.prepared = self.local.parent / "grab_mug_refinement" / "prepared"
        experiment.split_path = Path(experiment.report["split_path"])
        experiment.checkpoint = Mock()
        experiment.evaluate = Mock(return_value={"evaluation": {"aggregate": {}}, "exports": []})
        self.experiment = experiment
        # The actual checkpoint, metadata validation, log recovery, and run
        # control execute. Only model slots and device RNG restoration are inert.
        self.cuda_patch = patch("torch.cuda.set_rng_state_all")
        self.cuda_patch.start()
        self.addCleanup(self.cuda_patch.stop)

    def test_equal_target_resumes_export_without_training(self) -> None:
        experiment = self.experiment
        experiment.restore(self.source)
        self.assertEqual(experiment.epoch, experiment.args.epochs)
        experiment.optimizer.load_state_dict.assert_called_once()
        experiment.scheduler.load_state_dict.assert_called_once()
        experiment.run()
        experiment.optimizer.step.assert_not_called()
        experiment.model.train.assert_not_called()
        experiment.checkpoint.assert_not_called()
        self.assertEqual([call.args[0] for call in experiment.evaluate.call_args_list], ["train", "validation", "test"])
        self.assertFalse(experiment.evaluate.call_args_list[-1].kwargs["measure"])
        self.assertTrue((experiment.output / "reference_manifest.json").exists())
        self.assertEqual(experiment.report["phase"], "complete")
        self.assertNotIn("selected_test", experiment.report)

    def test_target_below_checkpoint_is_rejected(self) -> None:
        self.experiment.args.epochs -= 1
        with self.assertRaisesRegex(ValueError, "epoch"):
            self.experiment.restore(self.source)

    def test_trailing_training_rows_are_archived_and_removed(self) -> None:
        experiment = self.experiment
        experiment.args.epochs += 1
        epoch, step = int(self.saved["epoch"]), int(self.saved["step"])
        rows = [
            {"epoch": epoch, "step": step - 1},
            {"epoch": epoch, "step": step},
            {"epoch": epoch + 1, "step": step + 1},
            {"epoch": epoch + 1, "step": step + 2},
        ]
        original = "".join(json.dumps(row) + "\n" for row in rows) + '{"epoch":'
        log = experiment.output / "training.jsonl"
        log.write_text(original)
        experiment.restore(self.source)
        retained = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(retained, rows[:2])
        archives = list((experiment.output / "resume_archives").glob("*.jsonl"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_text(), original)
        self.assertEqual(experiment.report["training_log_recovery"]["discarded_rows"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
