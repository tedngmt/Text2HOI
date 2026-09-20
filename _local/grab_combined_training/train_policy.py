# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a residual PPO controller in the dedicated GraspXL GRAB tracking task.

An epoch visits every supported training-reference window once. PPO optimization
passes over a rollout are a separate setting. Validation subjects never supply
policy gradients or observation-normalization updates. All distances are in m.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn
from torch.distributions import Normal

OBSERVATION_SIZE = 383
EVALUATION_PROTOCOL = "planned_contact_and_lift_targets_v2"


def write_json(path: Path, value: dict) -> None:
    """Atomically write a report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    """Return the file digest used for experiment provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ObservationMoments:
    """Track training observation moments without updating on validation data."""

    def __init__(self, size: int):
        self.mean = np.zeros(size, dtype=np.float64)
        self.variance = np.ones(size, dtype=np.float64)
        self.count = 1e-4

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.mean) / np.sqrt(self.variance + 1e-6), -10, 10).astype(np.float32)

    def update(self, values: np.ndarray) -> None:
        count = len(values)
        if not count:
            return
        mean = values.mean(axis=0, dtype=np.float64)
        variance = values.var(axis=0, dtype=np.float64)
        delta = mean - self.mean
        total = self.count + count
        self.variance = (self.variance * self.count + variance * count + delta**2 * self.count * count / total) / total
        self.mean += delta * count / total
        self.count = total

    def state_dict(self) -> dict:
        return {"mean": self.mean, "variance": self.variance, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"].copy()
        self.variance = state["variance"].copy()
        self.count = float(state["count"])


class TrackingPolicy(nn.Module):
    """Gaussian residual policy; tanh bounds each action to [-1, 1]."""

    def __init__(self, observation_size: int, action_size: int = 51):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(observation_size, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh())
        self.mean = nn.Linear(128, action_size)
        self.critic = nn.Sequential(
            nn.Linear(observation_size, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        # Begin at the reference-PD baseline, with small exploratory residuals.
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((action_size,), -2.0))

    def distribution(self, observation: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        distribution = Normal(self.mean(self.actor(observation)), self.log_std.clamp(-5, -0.5).exp())
        return distribution, self.critic(observation).squeeze(-1)

    @staticmethod
    def log_probability(distribution: Normal, latent: torch.Tensor) -> torch.Tensor:
        # Stable log(1 - tanh(z)^2), including the change of variables.
        jacobian = 2 * (math.log(2) - latent - torch.nn.functional.softplus(-2 * latent))
        return (distribution.log_prob(latent) - jacobian).sum(-1)


def load_references(prepared: Path) -> tuple[dict, dict[str, dict], dict[str, list[dict]]]:
    """Read whole-recording splits and build nonoverlapping reference windows."""
    manifest = json.loads((prepared / "manifest.json").read_text())
    references = {}
    episodes = {"train": [], "validation": [], "test": []}
    subjects = {key: set() for key in episodes}
    for clip in manifest["clips"]:
        split = clip["split"]
        subjects[split].add(clip["subject"])
        path = Path(clip["path"])
        if not path.is_absolute():
            path = prepared / path
        if clip.get("sha256") and sha256(path) != clip["sha256"]:
            raise ValueError(f"Reference content changed: {path}")
        with np.load(path) as archive:
            item = {key: archive[key].copy() for key in archive.files}
        for key in ("hand_qpos", "object_pose", "joints", "contact"):
            if not np.isfinite(item[key]).all():
                raise ValueError(f"Nonfinite reference {clip['clip_id']}/{key}")
        if item["hand_qpos"].shape != (clip["nframes"], 51):
            raise ValueError(f"Wrong hand shape for {clip['clip_id']}")
        references[clip["clip_id"]] = item
        for index, episode in enumerate(clip["episodes"]):
            if episode.get("supported", True) and episode["end"] - episode["start"] >= 2:
                episodes[split].append(dict(episode, clip_id=clip["clip_id"], episode_index=index))
    if any(subjects[a] & subjects[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Subjects overlap across splits")
    if any(not episodes[split] for split in episodes):
        raise ValueError("Every split needs at least one supported episode")
    return manifest, references, episodes


def make_windows(episodes: list[dict], frames: int, references: dict | None = None) -> list[dict]:
    """Cover all transitions with windows that fit the hand's translation range [m]."""
    windows = []
    for episode in episodes:
        start = episode["start"]
        while start < episode["end"] - 1:
            end = min(start + frames + 1, episode["end"])
            if references is not None:
                positions = references[episode["clip_id"]]["hand_qpos"][:, :3]
                maximum_range = np.array([1.30, 1.30, 1.45])  # Includes margin for the yaw augmentation and residuals.
                while end - start > 2 and (np.ptp(positions[start:end], axis=0) > maximum_range).any():
                    end -= 1
                if (np.ptp(positions[start:end], axis=0) > maximum_range).any():
                    raise ValueError("A single reference transition exceeds the hand's workspace")
            windows.append(dict(episode, start=start, end=end))
            start = end - 1
    return windows


def transform_reference(item: dict, rng: np.random.Generator, augment: bool) -> dict:
    """Apply one rigid table-plane transform to an entire trajectory [m, rad]."""
    result = {key: np.asarray(item[key]).copy() for key in ("hand_qpos", "object_pose", "joints", "contact")}
    result["table_dimensions"] = np.asarray(item["table_dimensions"]).copy()
    result["table_pose"] = np.asarray(item["table_pose"]).copy()
    if not augment:
        return result
    yaw = rng.uniform(-0.15, 0.15)
    rotation = Rotation.from_rotvec([0, 0, yaw])
    matrix = rotation.as_matrix()
    center = result["object_pose"][0, :3].copy()
    shift = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), 0.0])
    for points in (result["hand_qpos"][:, :3], result["object_pose"][:, :3], result["joints"]):
        points[:] = (points - center) @ matrix.T + center + shift
    orientations = Rotation.from_euler("XYZ", result["hand_qpos"][:, 3:6])
    result["hand_qpos"][:, 3:6] = (rotation * orientations).as_euler("XYZ")
    quaternion = result["object_pose"][:, [4, 5, 6, 3]]
    quaternion = (rotation * Rotation.from_quat(quaternion)).as_quat()
    result["object_pose"][:, 3:7] = quaternion[:, [3, 0, 1, 2]]
    table = result["table_pose"]
    table[:3] = (table[:3] - center) @ matrix.T + center + shift
    table_quaternion = (rotation * Rotation.from_quat(table[[4, 5, 6, 3]])).as_quat()
    table[3:7] = table_quaternion[[3, 0, 1, 2]]
    return result


def policy_observation(raw: np.ndarray) -> np.ndarray:
    """Use relative hand/joint positions and tracking errors [m, rad]."""
    features = raw.copy()
    object_position = raw[:, 102:105]
    features[:, :3] -= object_position
    features[:, 102:105] -= raw[:, 373:376]
    features[:, 115:178] = (raw[:, 115:178].reshape(-1, 21, 3) - object_position[:, None]).reshape(-1, 63)
    features[:, 199:250] -= raw[:, :51]
    features[:, 202:250] = np.arctan2(np.sin(features[:, 202:250]), np.cos(features[:, 202:250]))
    features[:, 250:253] -= object_position
    features[:, 257:320] -= raw[:, 115:178]
    features[:, 373:376] -= object_position
    return features


class TrackingBatch:
    """Manage the raw GraspXL vector environment without its demo-only wrapper."""

    def __init__(self, manifest: dict, config: dict, augment: bool):
        import yaml
        from raisimGymTorch.env.bin import ours_grab_tracking

        self.num_envs = int(config["num_envs"])
        environment = dict(config["environment"], num_envs=self.num_envs, domain_randomization=augment)
        self.raw = ours_grab_tracking.RaisimGymEnv(config["resource_root"], yaml.safe_dump(environment))
        self.binary_sha256 = sha256(Path(ours_grab_tracking.__file__))
        self.control_dt = float(config["environment"]["control_dt"])
        self.raw.load_multi_articulated([manifest["object_urdf"]] * self.num_envs)
        self.observation = np.zeros((self.num_envs, self.raw.getRightObDim()), dtype=np.float32)
        self.left_observation = np.zeros((self.num_envs, self.raw.getLeftObDim()), dtype=np.float32)
        self.state = np.zeros((self.num_envs, self.raw.getGSDim()), dtype=np.float32)
        self.zero_actions = np.zeros((self.num_envs, 51), dtype=np.float32)
        self.reward = np.zeros(self.num_envs, dtype=np.float32)
        self.left_reward = np.zeros(self.num_envs, dtype=np.float32)
        self.done = np.zeros(self.num_envs, dtype=np.bool_)
        if self.observation.shape[1] != OBSERVATION_SIZE or self.state.shape[1] != 199:
            raise ValueError(f"Unexpected tracking interface: obs {self.observation.shape}, state {self.state.shape}")

    def close(self) -> None:
        self.raw.close()

    def reset(self, sequences: list[dict], seed: int, dt: float) -> None:
        self.raw.setSeed(seed)
        table_dimensions = np.stack([item["table_dimensions"] for item in sequences]).astype(np.float32)
        table_pose = np.stack([item["table_pose"] for item in sequences]).astype(np.float32)
        self.raw.add_stage(table_dimensions, table_pose)
        hand = np.stack([item["hand_qpos"][0] for item in sequences]).astype(np.float32)
        origins = np.stack(
            [(item["hand_qpos"][:, :3].min(0) + item["hand_qpos"][:, :3].max(0)) * 0.5 for item in sequences]
        ).astype(np.float32)
        hand = np.concatenate((hand, origins), axis=1)
        hand_velocity = np.stack([(item["hand_qpos"][1] - item["hand_qpos"][0]) / dt for item in sequences])
        # Avoid Euler branch-cut spikes; initialization follows the first frame.
        hand_velocity[:, 3:] = np.arctan2(np.sin(hand_velocity[:, 3:] * dt), np.cos(hand_velocity[:, 3:] * dt)) / dt
        hand_velocity = np.clip(hand_velocity, -10, 10).astype(np.float32)
        object_state = np.zeros((self.num_envs, 13), dtype=np.float32)
        for i, item in enumerate(sequences):
            object_state[i, :7] = item["object_pose"][0]
            object_state[i, 7:10] = (item["object_pose"][1, :3] - item["object_pose"][0, :3]) / dt
            rotations = Rotation.from_quat(item["object_pose"][:2, [4, 5, 6, 3]])
            object_state[i, 10:13] = (rotations[1] * rotations[0].inv()).as_rotvec() / dt
        self.raw.reset_state(hand, self.zero_actions, hand_velocity, self.zero_actions, object_state)

    def targets(self, sequences: list[dict], frame: int) -> None:
        indices = [min(frame, len(item["hand_qpos"]) - 1) for item in sequences]
        hands = np.stack([item["hand_qpos"][i] for item, i in zip(sequences, indices)]).astype(np.float32)
        objects = np.stack([item["object_pose"][i] for item, i in zip(sequences, indices)]).astype(np.float32)
        joints = np.stack([item["joints"][i].reshape(63) for item, i in zip(sequences, indices)]).astype(np.float32)
        extra = np.array(
            [[i / (len(item["hand_qpos"]) - 1), float(item["contact"][i])] for item, i in zip(sequences, indices)],
            dtype=np.float32,
        )
        velocity = np.stack(
            [(item["hand_qpos"][i] - item["hand_qpos"][max(0, i - 1)]) for item, i in zip(sequences, indices)]
        )
        velocity[:, 3:] = np.arctan2(np.sin(velocity[:, 3:]), np.cos(velocity[:, 3:]))
        velocity /= self.control_dt
        velocity[:, :3] = np.clip(velocity[:, :3], -5.0, 5.0)
        velocity[:, 3:] = np.clip(velocity[:, 3:], -20.0, 20.0)
        extra = np.concatenate((extra, velocity), axis=1).astype(np.float32)
        self.raw.set_goals_r(objects, joints, hands, extra)
        self.raw.observe(self.observation, self.left_observation)
        if not np.isfinite(self.observation).all():
            raise RuntimeError("Simulator produced nonfinite observations")

    def step(self, actions: np.ndarray) -> None:
        self.raw.step(
            np.ascontiguousarray(actions, dtype=np.float32), self.zero_actions, self.reward, self.left_reward, self.done
        )
        self.raw.get_global_state(self.state)
        if not np.isfinite(self.reward).all() or not np.isfinite(self.state).all():
            raise RuntimeError("Simulator produced nonfinite rewards/state")


def collect(
    batch: TrackingBatch,
    windows: list[dict],
    references: dict,
    policy: TrackingPolicy | None,
    moments: ObservationMoments,
    rng: np.random.Generator,
    device: torch.device,
    config: dict,
    train: bool,
) -> tuple[dict, list[dict]]:
    """Collect one padded vector batch; mask padding and stop at physical failures."""
    count = len(windows)
    padded = windows + [windows[-1]] * (batch.num_envs - count)
    sequences = []
    for window in padded:
        source = references[window["clip_id"]]
        item = {
            key: source[key][window["start"] : window["end"]]
            for key in ("hand_qpos", "object_pose", "joints", "contact")
        }
        item["table_dimensions"] = source["table_dimensions"]
        item["table_pose"] = source["table_pose"]
        sequences.append(transform_reference(item, rng, train))
    lengths = np.array([len(item["hand_qpos"]) - 1 for item in sequences])
    batch.reset(sequences, int(rng.integers(1, 2**30)), float(config["environment"]["control_dt"]))
    alive = np.arange(batch.num_envs) < count
    records = {key: [] for key in ("obs", "raw_obs", "latent", "logp", "value", "reward", "terminal", "valid")}
    metrics = [
        {
            "clip_id": w["clip_id"],
            "frames": int(lengths[i]),
            "completed_frames": 0,
            "failed": False,
            "reward": 0.0,
            "joint_error_sum_m": 0.0,
            "object_error_sum_m": 0.0,
            "table_depth_max_m": 0.0,
            "object_depth_max_m": 0.0,
            "contact_frames": 0,
            "expected_contact_frames": int(np.count_nonzero(sequences[i]["contact"][1:])),
            "matched_contact_frames": 0,
            "achieved_lift_max_m": 0.0,
            "reference_lift_max_m": float(
                max(0.0, np.max(sequences[i]["object_pose"][1:, 2] - sequences[i]["object_pose"][0, 2]))
            ),
        }
        for i, w in enumerate(windows)
    ]
    for frame in range(1, int(lengths.max()) + 1):
        active = alive & (frame <= lengths)
        batch.targets(sequences, frame)
        features = policy_observation(batch.observation)
        normalized = moments.normalize(features)
        observation = torch.from_numpy(normalized).to(device)
        with torch.no_grad():
            if policy is None:
                latent = torch.zeros((batch.num_envs, 51), device=device)
                values = torch.zeros(batch.num_envs, device=device)
                logp = torch.zeros_like(values)
            else:
                distribution, values = policy.distribution(observation)
                latent = distribution.sample() if train else distribution.mean
                logp = policy.log_probability(distribution, latent)
            actions = latent.tanh().cpu().numpy()
        records["obs"].append(normalized)
        records["raw_obs"].append(features)
        records["latent"].append(latent.cpu().numpy())
        records["logp"].append(logp.cpu().numpy())
        records["value"].append(values.cpu().numpy())
        batch.step(actions)
        terminal = batch.done | (frame == lengths)
        records["reward"].append(batch.reward.copy())
        records["terminal"].append(terminal.copy())
        records["valid"].append(active.copy())
        for i in range(count):
            if not active[i]:
                continue
            row = metrics[i]
            row["reward"] += float(batch.reward[i])
            row["failed"] = bool(row["failed"] or batch.done[i])
            if batch.done[i]:
                continue  # Stock vector wrapper has already reset the terminal world.
            row["completed_frames"] += 1
            state = batch.state[i]
            expected = sequences[i]
            row["joint_error_sum_m"] += float(
                np.linalg.norm(state[115:178].reshape(21, 3) - expected["joints"][frame], axis=-1).mean()
            )
            row["object_error_sum_m"] += float(np.linalg.norm(state[102:105] - expected["object_pose"][frame, :3]))
            row["table_depth_max_m"] = max(row["table_depth_max_m"], float(state[195]))
            row["object_depth_max_m"] = max(row["object_depth_max_m"], float(state[196]))
            contact = bool(state[178:194].any())
            expected_contact = bool(expected["contact"][frame])
            row["contact_frames"] += int(contact)
            row["matched_contact_frames"] += int(contact and expected_contact)
            row["achieved_lift_max_m"] = max(
                row["achieved_lift_max_m"], float(state[104] - expected["object_pose"][0, 2])
            )
        alive &= ~batch.done
        if not (alive & (frame < lengths)).any():
            break
    rollout = {key: np.stack(value) for key, value in records.items()}
    advantage = np.zeros_like(rollout["reward"])
    accumulator = np.zeros(batch.num_envs, dtype=np.float32)
    for t in reversed(range(len(advantage))):
        continuation = (~rollout["terminal"][t]).astype(np.float32)
        next_value = rollout["value"][t + 1] if t + 1 < len(advantage) else np.zeros(batch.num_envs)
        delta = rollout["reward"][t] + config["gamma"] * next_value * continuation - rollout["value"][t]
        accumulator = (delta + config["gamma"] * config["gae_lambda"] * continuation * accumulator) * rollout["valid"][
            t
        ]
        advantage[t] = accumulator
    rollout["advantage"] = advantage
    rollout["returns"] = advantage + rollout["value"]
    return rollout, metrics


def summarize(rows: list[dict]) -> dict:
    """Aggregate tracking and collision metrics; failed windows remain in the score."""
    planned = sum(row["frames"] for row in rows)
    completed = sum(row["completed_frames"] for row in rows)
    failed = sum(row["failed"] for row in rows)
    rewards = sum(row["reward"] for row in rows)
    expected = sum(row["expected_contact_frames"] for row in rows)
    lift_rows = [row for row in rows if row["reference_lift_max_m"] >= 0.03]
    result = {
        "windows": len(rows),
        "planned_frames": planned,
        "completed_frames": completed,
        "failure_rate": failed / max(1, len(rows)),
        "mean_reward_per_planned_frame": rewards / max(1, planned),
        "mean_joint_error_m_on_completed_frames": sum(row["joint_error_sum_m"] for row in rows) / max(1, completed),
        "mean_object_error_m_on_completed_frames": sum(row["object_error_sum_m"] for row in rows) / max(1, completed),
        "max_table_penetration_m": max((row["table_depth_max_m"] for row in rows), default=0),
        "max_object_penetration_m": max((row["object_depth_max_m"] for row in rows), default=0),
        "contact_recall": sum(row["matched_contact_frames"] for row in rows) / max(1, expected),
        "lift_windows": len(lift_rows),
        "lift_completion_fraction": sum(
            not row["failed"] and row["achieved_lift_max_m"] >= 0.8 * row["reference_lift_max_m"] for row in lift_rows
        )
        / max(1, len(lift_rows)),
    }
    # The collision/failure terms prevent selecting a high-reward unsafe rollout.
    result["selection_score"] = (
        result["mean_reward_per_planned_frame"] - 2 * result["failure_rate"] - 20 * result["max_table_penetration_m"]
    )
    return result


def selection_guard(candidate: dict, baseline: dict) -> list[str]:
    """Reject candidates that sacrifice physical interactions for a higher total reward."""
    reasons = []
    if candidate["failure_rate"] > baseline["failure_rate"]:
        reasons.append("more_failed_windows_than_reference_controller")
    if candidate["contact_recall"] < baseline["contact_recall"] - 0.02:
        reasons.append("contact_recall_dropped_more_than_two_percentage_points")
    if candidate["max_table_penetration_m"] > baseline["max_table_penetration_m"] + 0.002:
        reasons.append("maximum_table_penetration_worsened_more_than_two_mm")
    if candidate["max_object_penetration_m"] > baseline["max_object_penetration_m"] + 0.002:
        reasons.append("maximum_hand_object_penetration_worsened_more_than_two_mm")
    if candidate["lift_completion_fraction"] < baseline["lift_completion_fraction"]:
        reasons.append("fewer_planned_lifts_completed_than_reference_controller")
    return reasons


def optimize(
    policy: TrackingPolicy, optimizer: torch.optim.Optimizer, rollout: dict, config: dict, device: torch.device
) -> dict:
    mask = rollout["valid"].reshape(-1)
    tensors = {
        key: torch.as_tensor(value.reshape((-1,) + value.shape[2:])[mask], device=device, dtype=torch.float32)
        for key, value in rollout.items()
        if key in {"obs", "latent", "logp", "value", "advantage", "returns"}
    }
    size = len(tensors["obs"])
    if size < 2:
        return {"samples": size, "updates": 0}
    tensors["advantage"] = (tensors["advantage"] - tensors["advantage"].mean()) / (
        tensors["advantage"].std(unbiased=False) + 1e-8
    )
    updates = 0
    divergences = []
    for _ in range(config["ppo_passes"]):
        order = torch.randperm(size, device=device)
        stop = False
        for indices in order.split(config["minibatch_size"]):
            distribution, value = policy.distribution(tensors["obs"][indices])
            logp = policy.log_probability(distribution, tensors["latent"][indices])
            logratio = logp - tensors["logp"][indices]
            ratio = logratio.exp()
            advantage = tensors["advantage"][indices]
            actor_loss = torch.maximum(
                -advantage * ratio, -advantage * ratio.clamp(1 - config["clip_ratio"], 1 + config["clip_ratio"])
            ).mean()
            value_loss = 0.5 * (value - tensors["returns"][indices]).square().mean()
            loss = (
                actor_loss
                + config["value_coefficient"] * value_loss
                - config["entropy_coefficient"] * distribution.entropy().sum(-1).mean()
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
            optimizer.step()
            updates += 1
            divergence = float(((ratio - 1) - logratio).mean().detach())
            divergences.append(divergence)
            if divergence > config["target_kl"]:
                stop = True
                break
        if stop:
            break
    return {"samples": size, "updates": updates, "mean_kl": float(np.mean(divergences))}


def evaluate(batch, windows, references, policy, moments, config, device) -> tuple[dict, list[dict]]:
    rows = []
    rng = np.random.default_rng(config["seed"] + 1_000_000)
    for start in range(0, len(windows), batch.num_envs):
        _, current = collect(
            batch, windows[start : start + batch.num_envs], references, policy, moments, rng, device, config, False
        )
        rows.extend(current)
    return summarize(rows), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true", help="One batch per split, two epochs; excludes final test")
    parser.add_argument("--skip_test", action="store_true", help="Development checks only: leave the test split unused")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())["policy"]
    if args.smoke:
        config.update(epochs=2, num_envs=2, window_frames=24, validation_interval=1, ppo_passes=2)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if config["epochs"] < 1:
        parser.error("epochs must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest, references, episodes = load_references(args.prepared.resolve())
    dimensions = next(iter(references.values()))["table_dimensions"]
    if not all(np.allclose(item["table_dimensions"], dimensions, atol=1e-6) for item in references.values()):
        raise ValueError("This vectorized task requires the same physical table dimensions in every recording")
    config["environment"].update(
        table_length=float(dimensions[0]), table_width=float(dimensions[1]), table_thickness=float(dimensions[2])
    )
    windows = {split: make_windows(items, config["window_frames"], references) for split, items in episodes.items()}
    if args.smoke:
        windows = {split: items[: config["num_envs"]] for split, items in windows.items()}
    environment = TrackingBatch(manifest, config, True)
    validation_env = TrackingBatch(manifest, config, False)
    policy = TrackingPolicy(OBSERVATION_SIZE).to(device)
    moments = ObservationMoments(OBSERVATION_SIZE)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    rng = np.random.default_rng(config["seed"])
    epoch_start, best_score, steps = 1, -float("inf"), 0
    data_hash = sha256(args.prepared / "manifest.json")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("evaluation_protocol") != EVALUATION_PROTOCOL:
            raise ValueError("Evaluation metric definitions changed; use a fresh run for the corrected metrics")
        if checkpoint["manifest_sha256"] != data_hash:
            raise ValueError("Resume data manifest changed")
        if checkpoint.get("environment_sha256") != environment.binary_sha256:
            raise ValueError("Simulator binary changed; use a fresh run after environment changes")
        previous_config = {key: value for key, value in checkpoint["config"].items() if key != "epochs"}
        if previous_config != {key: value for key, value in config.items() if key != "epochs"}:
            raise ValueError("Resume configuration changed beyond the epoch target")
        policy.load_state_dict(checkpoint["policy"])
        moments.load_state_dict(checkpoint["moments"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        epoch_start = checkpoint["epoch"] + 1
        best_score, steps = checkpoint["best_score"], checkpoint["steps"]
        previous_best = args.resume.parent / "best.pt"
        if not previous_best.exists():
            raise FileNotFoundError("Resume requires the prior validation-selected best.pt beside last.pt")
        if previous_best.resolve() != (output / "best.pt").resolve():
            shutil.copy2(previous_best, output / "best.pt")
    elif (output / "last.pt").exists():
        raise FileExistsError("Output already contains a run; use --resume or a fresh output_dir")
    report = {
        "scope": "Text2HOI-refined GRAB references -> new residual PPO GraspXL tracking controller",
        "epoch_definition": "One pass through all supported training reference windows; ppo_passes is separate",
        "pretrained_graspxl_policy_loaded": False,
        "config": config,
        "prepared": str(args.prepared.resolve()),
        "manifest_sha256": data_hash,
        "environment_sha256": environment.binary_sha256,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "split_windows": {key: len(value) for key, value in windows.items()},
        "local_validation_only": "Released Text2HOI weights may already have seen these GRAB recordings",
        "smoke": args.smoke,
        "test_evaluation_enabled": not (args.smoke or args.skip_test),
        "device": str(device),
        "phase": "baseline",
        "epochs_completed": epoch_start - 1,
    }
    write_json(output / "report.json", report)
    baseline_path = output / "validation_baseline.json"
    if not baseline_path.exists():
        baseline, rows = evaluate(validation_env, windows["validation"], references, None, moments, config, device)
        write_json(baseline_path, {"summary": baseline, "windows": rows})
    else:
        baseline = json.loads(baseline_path.read_text())["summary"]

    def checkpoint_state(epoch: int) -> dict:
        return {
            "epoch": epoch,
            "steps": steps,
            "best_score": best_score,
            "config": config,
            "policy": policy.state_dict(),
            "moments": moments.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "manifest_sha256": data_hash,
            "environment_sha256": environment.binary_sha256,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }

    if not args.resume:
        best_score = baseline["selection_score"]
        torch.save(checkpoint_state(0), output / "best.pt")
        write_json(output / "best_validation.json", {"epoch": 0, "summary": baseline, "baseline": True})
    started = time.monotonic()
    try:
        for epoch in range(epoch_start, config["epochs"] + 1):
            order = rng.permutation(len(windows["train"]))
            training_rows, updates = [], 0
            for offset in range(0, len(order), environment.num_envs):
                current = [windows["train"][i] for i in order[offset : offset + environment.num_envs]]
                rollout, rows = collect(environment, current, references, policy, moments, rng, device, config, True)
                statistics = optimize(policy, optimizer, rollout, config, device)
                moments.update(rollout["raw_obs"][rollout["valid"]])
                training_rows.extend(rows)
                updates += statistics["updates"]
                steps += statistics["samples"]
            entry = {
                "epoch": epoch,
                "train": summarize(training_rows),
                "updates": updates,
                "steps": steps,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.monotonic() - started,
            }
            improved = False
            if epoch == 1 or epoch % config["validation_interval"] == 0 or epoch == config["epochs"]:
                validation, rows = evaluate(
                    validation_env, windows["validation"], references, policy, moments, config, device
                )
                entry["validation"] = validation
                scheduler.step(validation["selection_score"])
                guard_reasons = selection_guard(validation, baseline)
                entry["checkpoint_guard"] = {"eligible": not guard_reasons, "rejection_reasons": guard_reasons}
                if not guard_reasons and validation["selection_score"] > best_score:
                    best_score, improved = validation["selection_score"], True
                    write_json(
                        output / "best_validation.json", {"epoch": epoch, "summary": validation, "windows": rows}
                    )
            checkpoint = checkpoint_state(epoch)
            temporary = output / "last.tmp.pt"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "last.pt")
            if improved:
                torch.save(checkpoint, output / "best.pt")
            with (output / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
            report.update(
                phase="training",
                epochs_completed=epoch,
                best_validation_score=best_score,
                environment_steps=steps,
                elapsed_seconds=time.monotonic() - started,
            )
            write_json(output / "report.json", report)
            print(json.dumps(entry, allow_nan=False), flush=True)
        if not args.smoke and not args.skip_test:
            checkpoint = torch.load(output / "best.pt", map_location=device)
            policy.load_state_dict(checkpoint["policy"])
            moments.load_state_dict(checkpoint["moments"])
            final_test, rows = evaluate(validation_env, windows["test"], references, policy, moments, config, device)
            write_json(
                output / "test_final.json",
                {"selected_epoch": checkpoint["epoch"], "summary": final_test, "windows": rows},
            )
        report.update(phase="complete", elapsed_seconds=time.monotonic() - started)
        write_json(output / "report.json", report)
    finally:
        environment.close()
        validation_env.close()


if __name__ == "__main__":
    main()
