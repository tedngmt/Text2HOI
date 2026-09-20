# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Upstream refiner objective with deterministic held-out evaluation windows."""

import random

import numpy as np
import torch
from accumulation import capture_penetration_parts
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from lib.datasets.datasets import get_dataset
from lib.models.mano import build_mano_aa
from lib.networks.clip import encoded_text, load_and_freeze_clip
from lib.utils.loss import get_joint_contact_loss, get_l2_loss, get_penetration_loss
from lib.utils.model_utils import build_model_and_diffusion, build_pointnetfeat, build_refiner
from lib.utils.proc import get_contact_map, pc_normalize, proc_obj_feat_final_train, proc_refiner_input
from lib.utils.proc_grab import process_text
from lib.utils.proc_output import get_hand_joints_w_tip
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Windows(Dataset):
    def __init__(self, data, indices):
        self.data, self.indices = data, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        absolute = self.indices[index]
        length = int(self.data.nframes[absolute])
        start = np.random.randint(0, length - 150) if length > 150 else 0
        return self.window(absolute, start, random_text=True)

    def window(self, index, start, random_text=False):
        data = self.data
        length = min(150, int(data.nframes[index]) - start)
        if length <= 0:
            raise ValueError("Empty window")
        item = {}
        for key in ("x_lhand", "x_rhand", "x_obj"):
            shape = 9 if key == "x_obj" else 99
            item[key] = np.zeros((150, shape), dtype=np.float32)
            present = key == "x_obj" or bool(getattr(data, "is_" + key[2:])[index])
            if present:
                item[key][:length] = getattr(data, key)[index, start : start + length]
            item["valid_mask_" + key[2:]] = (np.arange(150) < length) & present
        left, right = bool(data.is_lhand[index]), bool(data.is_rhand[index])
        key = process_text(
            data.action_name[index], data.proc_obj_name[index], left, right, data.text_description, return_key=True
        )
        texts = data.text_description[key]
        item["text"] = str(np.random.choice(texts)) if random_text else str(texts[0])
        _, pc, normals, _ = data.object_model(data.obj_name[index])
        normalized, center, scale = pc_normalize(pc, return_params=True)
        item.update(obj_pc=pc, obj_pc_normal=normals, normalized_obj_pc=normalized, obj_cent=center, obj_scale=scale)
        item["cov_map"] = (
            (get_contact_map(data.lcov_idx[index], 1024, left) + get_contact_map(data.rcov_idx[index], 1024, right)) > 0
        ).astype(np.float32)
        return item


class Engine:
    def __init__(self, repo):
        with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
            cfg = compose(config_name="config", overrides=["dataset=grab"])
        self.config = EasyDict(OmegaConf.to_container(cfg, resolve=True))
        self.data = get_dataset("Motiongrab", self.config.dataset)
        self.left = build_mano_aa(is_rhand=False, flat_hand=True).cuda()
        self.right = build_mano_aa(is_rhand=True, flat_hand=True).cuda()
        self.refiner = build_refiner(self.config)
        self.generator, self.diffusion = build_model_and_diffusion(self.config, self.left, self.right, test=True)
        self.pointnet = build_pointnetfeat(self.config, test=True)
        self.clip = load_and_freeze_clip(self.config.clip.clip_version).cuda()
        self.parts = capture_penetration_parts()
        self.weights = dict(lambda_simple=1.0, lambda_penet=1.0, lambda_contact=5.0)

    @torch.no_grad()
    def prepare(self, batch):
        b = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        cfg = self.config
        features = proc_obj_feat_final_train(
            b["cov_map"],
            b["obj_scale"],
            b["obj_cent"],
            self.pointnet(b["normalized_obj_pc"]),
            cfg.texthom.use_obj_scale_centroid,
            cfg.texthom.use_contact_feat,
        )
        coarse = self.diffusion(
            self.generator,
            b["x_lhand"],
            b["x_rhand"],
            b["x_obj"],
            features,
            enc_text=encoded_text(self.clip, b["text"]),
            get_losses=False,
            valid_mask_lhand=b["valid_mask_lhand"],
            valid_mask_rhand=b["valid_mask_rhand"],
            valid_mask_obj=b["valid_mask_obj"],
            obj_verts_org=b["obj_pc"],
        )
        inputs = proc_refiner_input(
            *coarse,
            self.left,
            self.right,
            b["obj_pc"],
            b["obj_pc_normal"],
            b["valid_mask_lhand"],
            b["valid_mask_rhand"],
            b["valid_mask_obj"],
            b["cov_map"],
            "grab",
            return_psuedo_gt=True,
        )
        return b, coarse, inputs

    def loss(self, prepared, baseline=False):
        b, coarse, inputs = prepared
        il, ir, obj, lc, rc, lm, rm = inputs
        self.parts.clear()
        if baseline:
            left, right = coarse[:2]
            losses = dict(
                simple_loss=get_l2_loss(
                    pred_lhand=left,
                    pred_rhand=right,
                    targ_lhand=b["x_lhand"],
                    targ_rhand=b["x_rhand"],
                    mask_lhand=b["valid_mask_lhand"],
                    mask_rhand=b["valid_mask_rhand"],
                ),
                penet_loss=get_penetration_loss(
                    left,
                    right,
                    obj,
                    self.left,
                    self.right,
                    b["obj_pc"],
                    b["valid_mask_lhand"],
                    b["valid_mask_rhand"],
                    "grab",
                ),
                contact_loss=get_joint_contact_loss(left, right, lc, rc, self.left, self.right, lm, rm, loss_type="l1"),
            )
        else:
            left, right, losses = self.refiner.get_loss(
                il,
                ir,
                obj,
                b["x_lhand"],
                b["x_rhand"],
                b["obj_pc"],
                lc,
                rc,
                self.left,
                self.right,
                None,
                None,
                "grab",
                valid_mask_lhand=b["valid_mask_lhand"],
                valid_mask_rhand=b["valid_mask_rhand"],
                lhand_cont_joint_mask=lm,
                rhand_cont_joint_mask=rm,
                lambda_dict=self.weights,
                contact_loss_type="l1",
            )
        return losses, list(self.parts), (left, right)

    @torch.no_grad()
    def evaluate(self, records, split, limit_windows=0):
        if any(row["split"] != split for row in records):
            raise ValueError("Evaluation split contamination")
        saved = rng_state()
        was_training = self.refiner.training
        self.refiner.eval()
        windows = Windows(self.data, [r["index"] for r in records])
        rows = []
        try:
            for row in records:
                for start in range(0, int(row["retained_frames"]), 150):
                    seed_all(100000 + row["index"] * 10000 + start)
                    prepared = self.prepare(default_collate([windows.window(row["index"], start)]))
                    result = dict(clip_id=row["clip_id"], index=row["index"], start=start)
                    for mode in ("before", "after"):
                        losses, parts, predictions = self.loss(prepared, baseline=mode == "before")
                        report = {k: float(v) for k, v in losses.items()}
                        report["penetration_numerators"] = [float(n) for n, _ in parts]
                        report["penetration_counts"] = [n for _, n in parts]
                        distance_sum, count = 0.0, 0
                        for side, layer, predicted in zip(("lhand", "rhand"), (self.left, self.right), predictions):
                            mask = prepared[0]["valid_mask_" + side]
                            difference = get_hand_joints_w_tip(predicted, layer) - get_hand_joints_w_tip(
                                prepared[0]["x_" + side], layer
                            )
                            distances = difference.norm(dim=-1)[mask]
                            distance_sum += float(distances.sum())
                            count += distances.numel()
                        report.update(joint_distance_sum_m=distance_sum, joint_count=count)
                        result[mode] = report
                    rows.append(result)
                    if limit_windows and len(rows) >= limit_windows:
                        break
                if limit_windows and len(rows) >= limit_windows:
                    break
            totals = {}
            for mode in ("before", "after"):
                values = [r[mode] for r in rows]
                simple = sum(r["simple_loss"] for r in values) / len(values)
                contact = sum(r["contact_loss"] for r in values) / len(values)
                penetration = sum(
                    sum(r["penetration_numerators"][side] for r in values)
                    / max(sum(r["penetration_counts"][side] for r in values), 1)
                    for side in range(2)
                )
                joint_error = (
                    1000
                    * sum(r["joint_distance_sum_m"] for r in values)
                    / max(sum(r["joint_count"] for r in values), 1)
                )
                totals[mode] = dict(
                    simple_loss=simple,
                    contact_loss=contact,
                    penetration_loss=penetration,
                    combined_loss=simple + penetration + 5 * contact,
                    joint_error_mm=joint_error,
                )
            return dict(
                split=split,
                windows=len(rows),
                totals=totals,
                windows_detail=rows,
                protocol="Deterministic noised-GRAB denoising/refinement, not free text generation",
                limitation="Penetration uses upstream point/normal proxy; contacts use proximity pseudo-targets",
            )
        finally:
            self.refiner.train(was_training)
            restore_rng(saved)
