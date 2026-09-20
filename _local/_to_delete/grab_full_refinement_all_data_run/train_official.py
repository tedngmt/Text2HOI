# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Local, resumable execution of the upstream GRAB refiner training objective.

The default is the official physical batch of 64. --probe_steps limits execution
without creating a production checkpoint. No external experiment logging occurs.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--effective_batch", type=int, default=64)
    parser.add_argument("--probe_steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.effective_batch % args.batch_size:
        parser.error("effective_batch must be divisible by batch_size")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path("/home/nmt/Projects/Text2HOI")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["dataset=grab", "result_show_freq=50", "save_pth_freq=200"])
    (output / "upstream_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    config = EasyDict(OmegaConf.to_container(cfg, resolve=True))
    config.batch_size = args.batch_size
    config.num_workers = 0  # Avoid replicating the large in-memory dataset in WSL.
    from accumulation import BatchAccumulator, capture_penetration_parts
    from lib.datasets.datasets import get_dataloader
    from lib.models.mano import build_mano_aa
    from lib.networks.clip import encoded_text, load_and_freeze_clip
    from lib.utils.model_utils import build_model_and_diffusion, build_pointnetfeat, build_refiner
    from lib.utils.proc import proc_obj_feat_final_train, proc_refiner_input

    left = build_mano_aa(is_rhand=False, flat_hand=True).cuda()
    right = build_mano_aa(is_rhand=True, flat_hand=True).cuda()
    loader = get_dataloader("Motiongrab", config, config.dataset)
    refiner = build_refiner(config)
    generator, diffusion = build_model_and_diffusion(config, left, right, test=True)
    pointnet = build_pointnetfeat(config, test=True)
    clip = load_and_freeze_clip(config.clip.clip_version).cuda()
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=config.refiner.lr)
    epochs = int(np.ceil(config.refiner.iteration / (config.dataset.data_num / config.dataset.text_num) / 50) * 50)
    weights = {k: config.refiner[k] for k in ("lambda_simple", "lambda_penet", "lambda_contact")}
    accumulator = BatchAccumulator(refiner, args.effective_batch, weights)
    penetration_parts = capture_penetration_parts()
    micro_per_step = args.effective_batch // args.batch_size
    batches_per_epoch = len(loader.dataset) // args.effective_batch
    protocol = dict(
        epochs=epochs,
        batch_size=args.batch_size,
        num_workers=0,
        training_clips=len(loader.dataset),
        batches_per_epoch=batches_per_epoch,
        effective_batch=args.effective_batch,
        accumulation="separate penetration numerator gradients; batch-wide denominators",
        learning_rate=config.refiner.lr,
        loss_weights=weights,
        initialization="upstream zero_initialized=True; no pretrained refiner",
        checkpoint_selection="lowest training loss; no held-out validation",
        periodic_generation_demo="deferred; not part of gradient computation",
        probe_only=bool(args.probe_steps),
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    start_epoch, best_loss, step = 0, float("inf"), 0
    if args.resume:
        state = torch.load(output / "latest.pth")
        if state["protocol"] != protocol:
            raise ValueError("Resume protocol differs from checkpoint")
        refiner.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_loss, step = state["epoch"] + 1, state["best_loss"], state["step"]
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state["rng_cuda"]])
    started = time.monotonic()
    from cooling import CoolingThrottle

    cooling = CoolingThrottle()
    print(json.dumps(protocol), flush=True)
    for epoch in range(start_epoch, epochs):
        refiner.train()
        totals = []
        update_start = time.monotonic()
        update_idle = 0.0
        for batch_index, item in enumerate(loader):
            if batch_index >= batches_per_epoch * micro_per_step:
                break
            micro_start = time.monotonic()
            item = {k: v.cuda() if torch.is_tensor(v) else v for k, v in item.items()}
            target_left, target_right, obj = (item[k] for k in ("x_lhand", "x_rhand", "x_obj"))
            ml, mr, mo = (item[k] for k in ("valid_mask_lhand", "valid_mask_rhand", "valid_mask_obj"))
            pc, normals = item["obj_pc"], item["obj_pc_normal"]
            with torch.no_grad():
                text = encoded_text(clip, item["text"])
                features = proc_obj_feat_final_train(
                    item["cov_map"],
                    item["obj_scale"],
                    item["obj_cent"],
                    pointnet(item["normalized_obj_pc"]),
                    config.texthom.use_obj_scale_centroid,
                    config.texthom.use_contact_feat,
                )
                coarse_l, coarse_r, coarse_o = diffusion(
                    generator,
                    target_left,
                    target_right,
                    obj,
                    features,
                    enc_text=text,
                    get_losses=False,
                    valid_mask_lhand=ml,
                    valid_mask_rhand=mr,
                    valid_mask_obj=mo,
                    ldist_map=item["ldist_map"],
                    rdist_map=item["rdist_map"],
                    obj_verts_org=pc,
                    obj_pc_top_idx=None,
                )
                il, ir, ro, lc, rc, lm, rm = proc_refiner_input(
                    coarse_l,
                    coarse_r,
                    coarse_o,
                    left,
                    right,
                    pc,
                    normals,
                    ml,
                    mr,
                    mo,
                    item["cov_map"],
                    "grab",
                    return_psuedo_gt=True,
                )
            penetration_parts.clear()
            _, _, losses = refiner.get_loss(
                il,
                ir,
                ro,
                target_left,
                target_right,
                pc,
                lc,
                rc,
                left,
                right,
                item["ldist_map"],
                item["rdist_map"],
                "grab",
                valid_mask_lhand=ml,
                valid_mask_rhand=mr,
                lhand_cont_joint_mask=lm,
                rhand_cont_joint_mask=rm,
                lambda_dict=weights,
                contact_loss_type=config.refiner.contact_loss_type,
            )
            loss = sum(losses[k + "_loss"] * weights["lambda_" + k] for k in ("simple", "penet", "contact"))
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite training loss; checkpoint not advanced")
            if len(penetration_parts) != 2:
                raise RuntimeError("Expected two upstream penetration loss components")
            accumulator.add(losses, penetration_parts, args.batch_size)
            # Finish queued GPU work before idling; only wall-clock time changes.
            torch.cuda.synchronize()
            update_idle += cooling.wait(time.monotonic() - micro_start)
            if (batch_index + 1) % micro_per_step:
                continue
            value = accumulator.step(optimizer)
            torch.cuda.synchronize()
            step += 1
            totals.append(value)
            record = dict(
                epoch=epoch + 1,
                batch=(batch_index + 1) // micro_per_step,
                step=step,
                loss=totals[-1],
                seconds=time.monotonic() - update_start,
                peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                elapsed_seconds=time.monotonic() - started,
                cooling_active_percent=cooling.duty,
                cooling_idle_seconds=update_idle,
            )
            print(json.dumps(record), flush=True)
            update_start = time.monotonic()
            update_idle = 0.0
            with (output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if args.probe_steps and step >= args.probe_steps:
                (output / "probe_complete.json").write_text(json.dumps(record, indent=2) + "\n")
                return
        mean = float(np.mean(totals))
        improved = mean < best_loss
        best_loss = min(best_loss, mean)
        state = dict(
            model=refiner.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            loss=mean,
            best_loss=best_loss,
            step=step,
            protocol=protocol,
            rng_python=random.getstate(),
            rng_numpy=np.random.get_state(),
            rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all(),
        )
        torch.save(state, output / "latest.tmp")
        (output / "latest.tmp").replace(output / "latest.pth")
        if improved:
            torch.save({"model": state["model"], "epoch": epoch, "loss": mean}, output / "best.tmp")
            (output / "best.tmp").replace(output / "best.pth")
        if (epoch + 1) % config.save_pth_freq == 0:
            torch.save({"model": state["model"], "epoch": epoch, "loss": mean}, output / f"refiner_{epoch + 1}.pth")
    (output / "training_complete.json").write_text(json.dumps(dict(epochs=epochs, best_loss=best_loss, step=step)))


if __name__ == "__main__":
    main()
