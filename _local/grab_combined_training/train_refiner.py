# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a validated Text2HOI refiner and export GRAB references for IsaacLab.

This stage adapts recorded MANO trajectories. The text-to-motion generator is
frozen. Validation is held out from this local adaptation, although released
GRAB pretraining may already have exposed the model to these sequences.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from hydra import compose, initialize_config_dir


def load_previous_helpers(local: Path):
    """Reuse the independently audited native-mug SDF implementation."""
    path = local.parent / "grab_mug_refinement" / "train_refinement.py"
    specification = importlib.util.spec_from_file_location("grab_mug_geometry_helpers", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class RefinerExperiment:
    """Own one subject-separated refinement experiment and its artifacts."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.local = Path(__file__).resolve().parent
        helpers = load_previous_helpers(self.local)
        self.write_json = helpers.write_json
        self.digest = helpers.digest
        self.output = args.output_dir.resolve()
        self.prepared = args.prepared.resolve()
        self.split_path = args.splits.resolve()
        self.repo = self.local.parents[2] / "Text2HOI"
        self.started = time.monotonic()
        self.step = 0
        self.epoch = 0
        self.best_epoch = 0
        self.best_score = float("inf")
        self.rng = np.random.default_rng(args.seed)
        self.history = []
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "report.json").exists() and not args.resume:
            raise ValueError("Use a new output directory, or explicitly resume its checkpoint.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use the text2hoi environment with GPU access.")
        torch.set_num_threads(2)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        os.chdir(self.repo)
        sys.path.insert(0, str(self.repo))
        os.environ.setdefault("WANDB_MODE", "disabled")

        from lib.models.mano import build_mano_aa
        from lib.utils.model_utils import build_refiner
        from lib.utils.proc import proc_refiner_input
        from lib.utils.proc_output import get_hand_joints_w_tip, get_hand_verts

        self.get_hand_verts = get_hand_verts
        self.get_hand_joints = get_hand_joints_w_tip
        self.proc_refiner_input = proc_refiner_input
        self.manifest = json.loads((self.prepared / "manifest.json").read_text())
        split_document = json.loads(self.split_path.read_text())
        self.splits = split_document.get("splits", split_document)
        entries = self.manifest["clips"]
        self.clip_split = {}
        used_subjects = set()
        for split in ("train", "validation", "test"):
            group = self.splits[split]
            subjects = set(group["subjects"])
            if used_subjects & subjects:
                raise ValueError("Subject leakage between local training/validation/test groups.")
            used_subjects |= subjects
            for clip in group["clip_ids"]:
                if clip in self.clip_split:
                    raise ValueError("Clip occurs in more than one split: " + clip)
                self.clip_split[clip] = split
        if set(self.clip_split) != {entry["clip_id"] for entry in entries}:
            raise ValueError("The split must include every prepared GRAB mug clip exactly once.")
        for entry in entries:
            if entry["subject"] not in self.splits[self.clip_split[entry["clip_id"]]]["subjects"]:
                raise ValueError("Clip subject does not agree with split subject metadata.")

        with np.load(self.prepared / "object.npz", allow_pickle=False) as archive:
            self.objects = dict(archive)
        self.points = self.tensor(self.objects["object_points"])[None]
        self.normals = self.tensor(self.objects["object_normals"])[None]
        self.sdf = helpers.MugDistance(self.prepared / "sdf.npz")
        self.layers = {}
        for subject in sorted(used_subjects):
            with np.load(self.prepared / "subjects" / (subject + ".npz"), allow_pickle=False) as archive:
                assets = dict(archive)
            pair = []
            for side in ("lhand", "rhand"):
                layer = build_mano_aa(is_rhand=side == "rhand", flat_hand=True).cuda()
                layer.v_template.copy_(self.tensor(assets[side + "_v_template"]))
                layer.requires_grad_(False)
                pair.append(layer)
            self.layers[subject] = pair
        with initialize_config_dir(version_base=None, config_dir=str(self.repo / "configs")):
            config = compose(config_name="config", overrides=["dataset=grab"])
        self.model = build_refiner(config, test=True)
        self.model.requires_grad_(True)
        self.source_weights = self.repo / config.refiner.weight_path
        self.source_hash = self.digest(self.source_weights)
        self.generator_weights = self.repo / config.texthom.weight_path
        self.generator_hash = self.digest(self.generator_weights)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=args.lr_patience,
            threshold=args.min_delta,
            threshold_mode="abs",
            min_lr=args.min_learning_rate,
        )
        self.limits = self.tensor([0.015] * 3 + [0.15] * 96)
        self.report = {
            "experiment": "grab_mug_combined_validated_refiner_v1",
            "scope": "Recorded GRAB motion -> Text2HOI refiner -> reference tracking; generator is frozen.",
            "pretraining_exposure": (
                "Validation/test are excluded from local adaptation only. The released upstream preprocessing "
                "contains all 44 mug clips, and checkpoint training exposure cannot be excluded."
            ),
            "generator_trained": False,
            "refiner_trained": True,
            "physics_simulation_in_this_stage": False,
            "source_checkpoint": str(self.source_weights),
            "source_checkpoint_sha256": self.source_hash,
            "generator_checkpoint": str(self.generator_weights),
            "generator_checkpoint_sha256": self.generator_hash,
            "prepared_manifest_sha256": self.digest(self.prepared / "manifest.json"),
            "split_path": str(self.split_path),
            "split_sha256": self.digest(self.split_path),
            "sdf_sha256": self.digest(self.prepared / "sdf.npz"),
            "splits": self.splits,
            "epochs_requested": args.epochs,
            "test_evaluation_enabled": not args.skip_test and not args.skip_export,
            "validation_every": args.validation_every,
            "selection": (
                "Minimum validation objective among candidates with lower baseline penetration and "
                "contact retention within the allowed absolute drop; epoch 0 is the fallback."
            ),
            "contact_retention_max_drop": args.contact_retention_max_drop,
            "early_stopping": False,
            "overfitting_policy": (
                "Complete the requested epoch budget, reduce LR on validation plateaus, "
                "retain the best validation checkpoint, and report the train/validation curves. "
                "These controls cannot guarantee absence of overfitting."
            ),
            "seed": args.seed,
            "window_frames": args.window,
            "batch_size": 1,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": "Upstream 0.1 active during training; disabled during validation/export",
            "gradient_clip_norm": 10.0,
            "checkpoint_residual_wrapper": "0.1 scale; tanh cap 0.015 m translation / 0.15 rotation-6D components",
            "loss": (
                f"SDF penetration + {args.contact_weight} contact + .03 displacement "
                "+ .1 correction velocity + .05 acceleration"
            ),
            "contact_weight": args.contact_weight,
            "units": "Positions are metres; loss displacement terms use millimetres and source 30 FPS frame intervals.",
            "trainable_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "phase": "preparing_features",
            "output": str(self.output),
        }
        self.write_report()
        self.data = self.prepare_features(entries)
        self.windows = self.make_windows("train")
        self.report["windows_per_epoch"] = len(self.windows)
        self.report["clips_per_split"] = {
            split: sum(item["split"] == split for item in self.data) for split in ("train", "validation", "test")
        }
        self.report["frames_per_split"] = {
            split: sum(item["nframes"] for item in self.data if item["split"] == split)
            for split in ("train", "validation", "test")
        }
        if args.resume:
            self.restore(args.resume.resolve())

    @staticmethod
    def tensor(value) -> torch.Tensor:
        """Create a CUDA float32 tensor for model calculations."""
        return torch.as_tensor(value, dtype=torch.float32, device="cuda")

    def write_report(self) -> None:
        """Persist progress without exposing partially written JSON."""
        self.report.update(
            {
                "epoch_completed": self.epoch,
                "steps_completed": self.step,
                "elapsed_seconds": time.monotonic() - self.started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            }
        )
        self.write_json(self.output / "report.json", self.report)

    def prepare_features(self, entries: list[dict]) -> list[dict]:
        """Cache source-only conditioning without fitting across held-out clips."""
        result = []
        for entry in entries:
            with np.load(self.prepared / entry["clip_path"], allow_pickle=False) as archive:
                arrays = dict(archive)
            hands = [self.tensor(arrays["x_" + side])[None] for side in ("lhand", "rhand")]
            obj = self.tensor(arrays["x_obj"])[None]
            length = obj.shape[1]
            item = {
                "entry": entry,
                "clip_id": entry["clip_id"],
                "subject": entry["subject"],
                "split": self.clip_split[entry["clip_id"]],
                "nframes": length,
                "arrays": arrays,
                "translation": self.tensor(arrays["object_translation"]),
                "rotation": self.tensor(arrays["object_rotation"]),
            }
            coverage = np.zeros(1024, dtype=np.float32)
            for side in ("lhand", "rhand"):
                selected = arrays[side + "_contact_point_indices"][arrays[side + "_contact_mask"]]
                coverage[selected] = 1
            feature_parts = [[], []]
            before_parts = []
            with torch.no_grad():
                for start in range(0, length, 128):
                    end = min(length, start + 128)
                    mask = torch.ones((1, end - start), dtype=torch.bool, device="cuda")
                    values = self.proc_refiner_input(
                        hands[0][:, start:end],
                        hands[1][:, start:end],
                        obj[:, start:end],
                        *self.layers[entry["subject"]],
                        self.points,
                        self.normals,
                        mask,
                        mask,
                        mask,
                        self.tensor(coverage)[None],
                        "grab",
                    )
                    for side in range(2):
                        feature_parts[side].append(values[side][0])
                    vertices, _ = self.local_vertices(
                        item, (hands[0][:, start:end], hands[1][:, start:end]), start, end
                    )
                    before_parts.append(vertices)
                item["inputs"] = [torch.cat(parts) for parts in feature_parts]
                item["before"] = torch.cat(before_parts)
                local_vertices = torch.einsum(
                    "tvi,tij->tvj", item["before"] - item["translation"][:, None], item["rotation"]
                )
                item["before_sdf"] = self.sdf(local_vertices)
                item["near"] = item["before_sdf"].abs() < 0.012
                item["contact"] = item["before_sdf"].abs() < 0.003
            result.append(item)
            print(json.dumps({"features": item["clip_id"], "split": item["split"], "frames": length}), flush=True)
        return result

    def make_windows(self, split: str) -> list[tuple[int, int, int]]:
        """Cover every source frame of the requested split with bounded windows."""
        windows = []
        for index, item in enumerate(self.data):
            if item["split"] != split:
                continue
            length = item["nframes"]
            windows.extend(
                (index, start, min(length, start + self.args.window))
                for start in range(0, length, self.args.window)
                if min(length, start + self.args.window) - start >= 8
            )
            tail = (index, max(0, length - self.args.window), length)
            if tail not in windows:
                windows.append(tail)
        return windows

    def predict(self, item: dict, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the same bounded correction during training and inference."""
        inputs = [value[None, start:end] for value in item["inputs"]]
        mask = torch.ones((1, end - start), dtype=torch.bool, device="cuda")
        predictions = self.model(*inputs, valid_mask_lhand=mask, valid_mask_rhand=mask)
        return tuple(
            value[..., :99] + self.limits * torch.tanh(0.1 * (prediction - value[..., :99]) / self.limits)
            for value, prediction in zip(inputs, predictions)
        )

    def local_vertices(
        self, item: dict, parameters: tuple[torch.Tensor, torch.Tensor], start: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return world and object-local personalized MANO vertices [m]."""
        vertices = torch.cat(
            [self.get_hand_verts(params, layer)[0] for params, layer in zip(parameters, self.layers[item["subject"]])],
            dim=1,
        )
        local_vertices = torch.einsum(
            "tvi,tij->tvj", vertices - item["translation"][start:end, None], item["rotation"][start:end]
        )
        return vertices, local_vertices

    def loss_terms(self, item: dict, vertices: torch.Tensor, distances: torch.Tensor, start: int, end: int) -> dict:
        """Measure geometry corrections using millimetre-scaled penalties."""
        source = item["before"][start:end]
        original_distances = item["before_sdf"][start:end]
        near = item["near"][start:end]
        contacts = item["contact"][start:end]
        correction = (vertices - source) * 1000
        penetration = functional.relu(0.0002 - distances) * 1000
        penetration_loss = penetration.square().sum() / near.sum().clamp_min(1)
        target_distance = original_distances.clamp_min(0.0002)
        contact_loss = (((distances - target_distance) * 1000).square() * contacts).sum() / contacts.sum().clamp_min(1)
        displacement = correction.square().mean()
        velocity = correction.diff(dim=0).square().mean()
        acceleration = correction.diff(dim=0, n=2).square().mean()
        total = (
            penetration_loss
            + self.args.contact_weight * contact_loss
            + 0.03 * displacement
            + 0.1 * velocity
            + 0.05 * acceleration
        )
        return {
            "total": total,
            "penetration": penetration_loss,
            "contact": contact_loss,
            "displacement": displacement,
            "velocity": velocity,
            "acceleration": acceleration,
        }

    def infer(self, item: dict) -> tuple:
        """Blend overlapping predictions and reconstruct the entire source clip."""
        length = item["nframes"]
        accumulators = [torch.zeros((length, 99), device="cuda") for _ in range(2)]
        denominator = torch.zeros((length, 1), device="cuda")
        starts = list(range(0, max(length - self.args.window + 1, 1), self.args.window // 2))
        starts = sorted(set(starts + [max(0, length - self.args.window)]))
        for start in starts:
            end = min(length, start + self.args.window)
            predictions = self.predict(item, start, end)
            weights = torch.hann_window(end - start, periodic=False, device="cuda").clamp_min(0.05)[:, None]
            for side in range(2):
                accumulators[side][start:end] += predictions[side][0] * weights
            denominator[start:end] += weights
        parameters = tuple((value / denominator)[None] for value in accumulators)
        vertices_parts, distance_parts, joints_parts = [], [], [[], []]
        for start in range(0, length, 128):
            end = min(length, start + 128)
            chunks = tuple(value[:, start:end] for value in parameters)
            vertices, local_vertices = self.local_vertices(item, chunks, start, end)
            vertices_parts.append(vertices)
            distance_parts.append(self.sdf(local_vertices))
            for side in range(2):
                joints_parts[side].append(self.get_hand_joints(chunks[side], self.layers[item["subject"]][side])[0])
        return (
            parameters,
            torch.cat(vertices_parts),
            torch.cat(distance_parts),
            tuple(torch.cat(x) for x in joints_parts),
        )

    def evaluate(self, split: str, stage: str, export: bool = False, measure: bool = True) -> dict:
        """Evaluate complete clips without gradients; optionally export references."""
        if not measure and not export:
            raise ValueError("Disabling measurements is only supported while exporting references.")
        self.model.eval()
        results, exports = [], []
        if export:
            (self.output / "references").mkdir(exist_ok=True)
        with torch.no_grad():
            for item in self.data:
                if item["split"] != split:
                    continue
                parameters, vertices, distances, joints = self.infer(item)
                metrics = None
                if measure:
                    loss = self.loss_terms(item, vertices, distances, 0, item["nframes"])
                    terms = {key: float(value) for key, value in loss.items()}
                    if not all(np.isfinite(value) for value in terms.values()):
                        raise RuntimeError("Non-finite evaluation: " + item["clip_id"])
                    depth = (-distances).clamp_min(0) * 1000
                    movement = (vertices - item["before"]).norm(dim=-1) * 1000
                    selected = item["contact"]
                    metrics = {
                        "clip_id": item["clip_id"],
                        "frames": item["nframes"],
                        "loss": terms,
                        "vertices_deeper_than_2_mm": int((depth > 2).sum()),
                        "vertices_deeper_than_5_mm": int((depth > 5).sum()),
                        "maximum_penetration_mm": float(depth.max()),
                        "mean_vertex_displacement_mm": float(movement.mean()),
                        "maximum_vertex_displacement_mm": float(movement.max()),
                        "source_contact_vertices": int(selected.sum()),
                        "retained_source_contact_vertices": int((distances[selected].abs() < 0.003).sum()),
                    }
                    results.append(metrics)
                if export:
                    arrays = item["arrays"]
                    path = self.output / "references" / (item["clip_id"] + ".npz")
                    np.savez_compressed(
                        path,
                        x_lhand=parameters[0][0].cpu().numpy(),
                        x_rhand=parameters[1][0].cpu().numpy(),
                        x_obj=arrays["x_obj"],
                        object_rotation=arrays["object_rotation"],
                        object_translation=arrays["object_translation"],
                        source_frame_indices=arrays["source_frame_indices"],
                        lhand_joints=joints[0].cpu().numpy(),
                        rhand_joints=joints[1].cpu().numpy(),
                        lhand_vertices=vertices[:, :778].cpu().numpy(),
                        rhand_vertices=vertices[:, 778:].cpu().numpy(),
                        valid_mask_lhand=arrays["valid_mask_lhand"],
                        valid_mask_rhand=arrays["valid_mask_rhand"],
                        valid_mask_obj=arrays["valid_mask_obj"],
                        lhand_source_contact=arrays["lhand_source_contact"],
                        rhand_source_contact=arrays["rhand_source_contact"],
                        source_fps=np.asarray(120.0, dtype=np.float32),
                        playback_fps=np.asarray(30.0, dtype=np.float32),
                    )
                    exports.append(
                        {
                            **item["entry"],
                            "split": split,
                            "reference_path": str(path),
                            "reference_sha256": self.digest(path),
                            "metrics": metrics,
                        }
                    )
        if not measure:
            return {"evaluation": None, "exports": exports}
        frames = sum(value["frames"] for value in results)
        aggregate = {
            "loss": {
                key: sum(value["loss"][key] * value["frames"] for value in results) / frames
                for key in ("total", "penetration", "contact", "displacement", "velocity", "acceleration")
            },
            "vertices_deeper_than_2_mm": sum(value["vertices_deeper_than_2_mm"] for value in results),
            "vertices_deeper_than_5_mm": sum(value["vertices_deeper_than_5_mm"] for value in results),
            "maximum_penetration_mm": max(value["maximum_penetration_mm"] for value in results),
            "mean_vertex_displacement_mm": sum(
                value["mean_vertex_displacement_mm"] * value["frames"] for value in results
            )
            / frames,
            "source_contact_retention": sum(value["retained_source_contact_vertices"] for value in results)
            / max(1, sum(value["source_contact_vertices"] for value in results)),
        }
        evaluation = {
            "split": split,
            "stage": stage,
            "epoch": self.epoch,
            "frames": frames,
            "per_clip": results,
            "aggregate": aggregate,
        }
        self.write_json(self.output / ("evaluation_" + stage + "_" + split + ".json"), evaluation)
        print(
            "EVALUATION " + json.dumps({"split": split, "stage": stage, "epoch": self.epoch, **aggregate}), flush=True
        )
        if export:
            return {"evaluation": evaluation, "exports": exports}
        return evaluation

    def checkpoint(self, filename: str) -> None:
        """Atomically save model, optimizer, scheduler, and random-generator state."""
        path = self.output / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "step": self.step,
                "best_epoch": self.best_epoch,
                "best_score": self.best_score,
                "history": self.history,
                "config": self.report,
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "generator": self.rng.bit_generator.state,
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                },
            },
            temporary,
        )
        temporary.replace(path)

    def restore(self, checkpoint_path: Path) -> None:
        """Resume this experiment while rejecting data or loss-setting changes."""
        saved = torch.load(checkpoint_path, map_location="cpu")
        previous = saved["config"]
        if "validation_initial" in previous:
            self.report["validation_initial"] = previous["validation_initial"]
        for key in (
            "experiment",
            "prepared_manifest_sha256",
            "split_sha256",
            "sdf_sha256",
            "source_checkpoint_sha256",
            "window_frames",
            "checkpoint_residual_wrapper",
            "loss",
            "contact_retention_max_drop",
        ):
            if previous[key] != self.report[key]:
                raise ValueError("Resume metadata mismatch: " + key)
        self.model.load_state_dict(saved["model"], strict=True)
        self.optimizer.load_state_dict(saved["optimizer"])
        self.scheduler.load_state_dict(saved["scheduler"])
        self.epoch, self.step = int(saved["epoch"]), int(saved["step"])
        self.best_epoch, self.best_score = int(saved["best_epoch"]), float(saved["best_score"])
        self.history = saved["history"]
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        self.rng.bit_generator.state = saved["rng"]["generator"]
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        best_source = checkpoint_path.parent / "best.pth"
        if not best_source.is_file():
            raise ValueError("Resume requires the associated best.pth checkpoint.")
        if best_source.resolve() != (self.output / "best.pth").resolve():
            shutil.copy2(best_source, self.output / "best.pth")
        self.report["resume_checkpoint"] = str(checkpoint_path)
        self.report["resume_checkpoint_sha256"] = self.digest(checkpoint_path)
        self.report["resumed_from_epoch"] = self.epoch
        if self.epoch > self.args.epochs:
            raise ValueError("epochs is a total target and cannot be below the resumed epoch.")
        self.restore_training_log()

    def restore_training_log(self) -> None:
        """Archive discarded log history and retain only saved-checkpoint updates."""
        path = self.output / "training.jsonl"
        if not path.exists():
            return
        lines = path.read_text().splitlines(keepends=True)
        retained = []
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
                step, epoch = int(row["step"]), int(row["epoch"])
            except (ValueError, KeyError, TypeError):
                if index != len(lines) - 1:
                    raise ValueError(f"Malformed training log before its final line: {path}:{index + 1}") from None
                # A process can stop partway through its final JSON write.
                continue
            if step <= self.step and epoch <= self.epoch:
                retained.append(line if line.endswith("\n") else line + "\n")
        discarded = len(lines) - len(retained)
        if not discarded:
            return
        archive_dir = self.output / "resume_archives"
        archive_dir.mkdir(exist_ok=True)
        archive = archive_dir / f"training_before_step_{self.step}_{time.time_ns()}.jsonl"
        shutil.copy2(path, archive)
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(retained))
        temporary.replace(path)
        self.report["training_log_recovery"] = {
            "archive": str(archive),
            "retained_rows": len(retained),
            "discarded_rows": discarded,
            "checkpoint_epoch": self.epoch,
            "checkpoint_step": self.step,
        }

    def selection_guard(self, candidate: dict) -> dict:
        """Reject lower-loss models that sacrifice source contact retention."""
        baseline = self.report["validation_initial"]
        minimum_contact = max(0.0, baseline["source_contact_retention"] - self.args.contact_retention_max_drop)
        contact_ok = candidate["source_contact_retention"] >= minimum_contact
        penetration_ok = candidate["loss"]["penetration"] < baseline["loss"]["penetration"]
        objective_ok = candidate["loss"]["total"] < self.best_score - self.args.min_delta
        reasons = []
        if not contact_ok:
            reasons.append("contact_retention_below_baseline_minus_allowed_drop")
        if not penetration_ok:
            reasons.append("penetration_not_better_than_baseline")
        if not objective_ok:
            reasons.append("objective_not_better_than_accepted_best")
        return {
            "eligible": contact_ok and penetration_ok and objective_ok,
            "baseline_contact_retention": baseline["source_contact_retention"],
            "minimum_contact_retention": minimum_contact,
            "candidate_contact_retention": candidate["source_contact_retention"],
            "baseline_penetration": baseline["loss"]["penetration"],
            "candidate_penetration": candidate["loss"]["penetration"],
            "rejection_reasons": reasons,
        }

    def run(self) -> None:
        """Complete the requested budget and select only using validation metrics."""
        if self.epoch == 0:
            self.report["phase"] = "initial_validation"
            initial = self.evaluate("validation", "initial")
            self.best_score = initial["aggregate"]["loss"]["total"]
            self.report["validation_initial"] = initial["aggregate"]
            self.checkpoint("best.pth")
        self.report["phase"] = "training"
        self.write_report()
        with (self.output / "training.jsonl").open("a") as stream:
            for epoch in range(self.epoch + 1, self.args.epochs + 1):
                self.model.train()
                totals = []
                epoch_started = time.monotonic()
                for index in self.rng.permutation(len(self.windows)):
                    item_index, start, end = self.windows[index]
                    item = self.data[item_index]
                    self.optimizer.zero_grad(set_to_none=True)
                    parameters = self.predict(item, start, end)
                    vertices, local_vertices = self.local_vertices(item, parameters, start, end)
                    losses = self.loss_terms(item, vertices, self.sdf(local_vertices), start, end)
                    if not torch.isfinite(losses["total"]):
                        raise RuntimeError("Non-finite training loss: " + item["clip_id"])
                    losses["total"].backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    if not torch.isfinite(norm):
                        raise RuntimeError("Non-finite training gradients")
                    self.optimizer.step()
                    self.step += 1
                    record = {
                        "epoch": epoch,
                        "step": self.step,
                        "clip_id": item["clip_id"],
                        "start": start,
                        "end": end,
                        "gradient_norm": float(norm),
                        **{key: float(value) for key, value in losses.items()},
                    }
                    totals.append(record)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                self.epoch = epoch
                summary = {
                    "epoch": epoch,
                    "step": self.step,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "training_loss": float(np.mean([value["total"] for value in totals])),
                    "epoch_training_seconds": time.monotonic() - epoch_started,
                }
                validate = epoch == 1 or epoch % self.args.validation_every == 0 or epoch == self.args.epochs
                if validate:
                    # Both curves use complete-clip inference with dropout disabled,
                    # so the reported generalization gap is comparable.
                    train_eval = self.evaluate("train", f"epoch_{epoch:03d}")
                    validation = self.evaluate("validation", f"epoch_{epoch:03d}")
                    score = validation["aggregate"]["loss"]["total"]
                    summary["training_evaluation"] = train_eval["aggregate"]
                    summary["validation"] = validation["aggregate"]
                    summary["validation_minus_train_loss"] = score - train_eval["aggregate"]["loss"]["total"]
                    self.scheduler.step(score)
                    guard = self.selection_guard(validation["aggregate"])
                    summary["selection_guard"] = guard
                    improved = guard["eligible"]
                    if improved:
                        self.best_score, self.best_epoch = score, epoch
                    self.history.append(summary)
                    if improved:
                        self.checkpoint("best.pth")
                    self.checkpoint("last.pth")
                else:
                    self.history.append(summary)
                self.report.update(
                    {"best_epoch": self.best_epoch, "best_validation_loss": self.best_score, "latest_epoch": summary}
                )
                self.write_json(self.output / "history.json", {"epochs": self.history})
                self.write_report()
                print("EPOCH " + json.dumps(summary, allow_nan=False), flush=True)

        self.report["phase"] = "selecting_best"
        selected = torch.load(self.output / "best.pth", map_location="cpu")
        self.model.load_state_dict(selected["model"], strict=True)
        del selected
        self.report["selected_checkpoint"] = str(self.output / "best.pth")
        self.report["selected_checkpoint_sha256"] = self.digest(self.output / "best.pth")
        self.report["last_checkpoint"] = str(self.output / "last.pth")
        self.report["best_epoch"] = self.best_epoch
        self.report["best_validation_loss"] = self.best_score
        if not self.args.skip_export:
            self.report["phase"] = "exporting_selected_model"
            self.write_report()
            exported = []
            for split in ("train", "validation", "test"):
                measure = not (self.args.skip_test and split == "test")
                result = self.evaluate(split, "selected_best", export=True, measure=measure)
                if measure:
                    self.report["selected_" + split] = result["evaluation"]["aggregate"]
                exported.extend(result["exports"])
            self.write_json(
                self.output / "reference_manifest.json",
                {
                    "schema_version": 1,
                    "source": "Recorded GRAB clips refined by validation-selected Text2HOI refiner",
                    "checkpoint": str(self.output / "best.pth"),
                    "checkpoint_sha256": self.report["selected_checkpoint_sha256"],
                    "best_epoch": self.best_epoch,
                    "prepared_root": str(self.prepared),
                    "split_path": str(self.split_path),
                    "split_sha256": self.report["split_sha256"],
                    "clips": exported,
                    "object_path": str(self.prepared / "object.npz"),
                    "source_fps": 120,
                    "playback_fps": 30,
                    "joint_order": "MANO joints 0..15, then index/middle/pinky/ring/thumb tips at 16..20",
                    "pretraining_exposure": self.report["pretraining_exposure"],
                    "generator_trained": False,
                    "test_evaluation_enabled": not self.args.skip_test,
                },
            )
        self.report["original_refiner_unchanged"] = self.digest(self.source_weights) == self.source_hash
        self.report["original_generator_unchanged"] = self.digest(self.generator_weights) == self.generator_hash
        if not self.report["original_refiner_unchanged"] or not self.report["original_generator_unchanged"]:
            raise RuntimeError("A released checkpoint changed unexpectedly.")
        self.report["phase"] = "complete"
        self.report["all_losses_and_gradients_finite"] = True
        self.write_report()
        print("COMPLETE " + str(self.output), flush=True)


def main() -> None:
    local = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=local.parent / "grab_mug_refinement" / "prepared")
    parser.add_argument("--splits", type=Path, default=local / "prepared" / "splits.json")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--contact_weight", type=float, default=1.0)
    parser.add_argument("--contact_retention_max_drop", type=float, default=0.02)
    parser.add_argument("--validation_every", type=int, default=5)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--min_learning_rate", type=float, default=1e-7)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--skip_export", action="store_true", help="Smoke checks only: omit final test evaluation and export."
    )
    parser.add_argument(
        "--skip_test", action="store_true", help="Export all references but omit test losses, metrics, and reports."
    )
    args = parser.parse_args()
    if args.epochs < 1 or not 8 <= args.window <= 150 or args.validation_every < 1:
        parser.error("epochs and validation_every must be positive, and window must be 8..150")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.min_delta < 0 or args.lr_patience < 0:
        parser.error("Invalid optimizer or scheduler settings")
    if args.contact_weight <= 0 or not 0 <= args.contact_retention_max_drop <= 1:
        parser.error("contact_weight must be positive and contact_retention_max_drop must be on the 0..1 scale")
    args.prepared = args.prepared.resolve()
    args.splits = args.splits.resolve()
    args.output_dir = args.output_dir.resolve()
    args.resume = args.resume.resolve() if args.resume else None
    experiment = RefinerExperiment(args)
    try:
        experiment.run()
    except Exception as error:
        experiment.report.update({"phase": "failed", "error": repr(error)})
        experiment.write_report()
        raise


if __name__ == "__main__":
    main()
