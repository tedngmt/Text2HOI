# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Train a fresh subject-held-out refiner; select on validation, then test once."""

import argparse
import fcntl
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

LOCAL = Path(__file__).resolve().parent
REPO = LOCAL.parents[2] / "Text2HOI"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(LOCAL.parent / "refiner_tools"))
os.chdir(REPO)

from accumulation import BatchAccumulator
from cooling import CoolingThrottle
from engine import Engine, Windows, restore_rng, rng_state, seed_all
from lib.utils.proc_grab import process_text
from prepare_split import sha256


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def verify_manifest(manifest):
    groups = {name: [r for r in manifest["clips"] if r["split"] == name] for name in ("train", "validation", "test")}
    expected = {"train": {f"s{i}" for i in range(1, 9)}, "validation": {"s9"}, "test": {"s10"}}
    assert len({r["index"] for r in manifest["clips"]}) == len(manifest["clips"]) == 1335
    for name, rows in groups.items():
        assert {r["subject"] for r in rows} == expected[name]
    hashes = {}
    for row in manifest["clips"]:
        assert hashes.get(row["motion_sha256"], row["split"]) == row["split"]
        hashes[row["motion_sha256"]] = row["split"]
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "smoke", "test"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = LOCAL / ("smoke" if args.mode == "smoke" else "run")
    output.mkdir(exist_ok=True)
    lock = (LOCAL / "experiment.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = LOCAL / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    groups = verify_manifest(manifest)
    assert sha256(Path(manifest["data_path"])) == manifest["data_sha256"], "Dataset changed after split verification"
    if args.mode == "test":
        if not (output / "training_complete.json").exists():
            raise RuntimeError("Test set stays sealed until training and validation checkpoint selection finish")
        if (output / "test_results.json").exists():
            raise FileExistsError("Final test already evaluated; preserve the original results")
    config = json.loads((LOCAL / "config.json").read_text())
    protocol = dict(
        config=config,
        manifest_sha256=sha256(manifest_path),
        mode="smoke" if args.mode == "smoke" else "train",
        frozen_weights={name: sha256(REPO / "checkpoints/grab" / f"{name}.pth") for name in ("texthom", "pointfeat")},
    )
    if (output / "latest.pth").exists() and args.mode == "train" and not args.resume:
        raise FileExistsError("Use --resume for this existing experiment")
    torch.set_num_threads(4)
    seed_all(config["seed"])
    engine = Engine(REPO)
    optimizer = torch.optim.AdamW(engine.refiner.parameters(), lr=config["learning_rate"])
    start_epoch, step, best = 0, 0, float("inf")
    if args.mode == "test":
        state = torch.load(output / "best_validation.pth", map_location="cpu")
        assert state["protocol"] == protocol and state["selection_split"] == "validation"
        engine.refiner.load_state_dict(state["model"])
        result = engine.evaluate(groups["test"], "test")
        result.update(
            checkpoint_sha256=sha256(output / "best_validation.pth"),
            selected_epoch=state["epoch"] + 1,
            manifest_sha256=protocol["manifest_sha256"],
            scope=manifest["scope"],
            pretrained_exposure=manifest["pretrained_exposure"],
        )
        save_json(output / "test_results.json", result)
        print(json.dumps(result["totals"]), flush=True)
        return
    if args.resume:
        state = torch.load(output / "latest.pth")
        assert state["protocol"] == protocol, "Resume protocol or dataset differs"
        engine.refiner.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        start_epoch, step, best = state["epoch"] + 1, state["step"], state["best_validation_loss"]
    save_json(output / "protocol.json", protocol)
    indices = [r["index"] for r in groups["train"]]
    # Derive class frequencies from training subjects only, not full-data weights.
    keys = [
        process_text(
            engine.data.action_name[i],
            engine.data.proc_obj_name[i],
            engine.data.is_lhand[i],
            engine.data.is_rhand[i],
            engine.data.text_description,
            return_key=True,
        )
        for i in indices
    ]
    frequencies = Counter(keys)
    sampler = WeightedRandomSampler(torch.tensor([1 / frequencies[key] for key in keys]), len(indices))
    loader = DataLoader(
        Windows(engine.data, indices), batch_size=config["microbatch"], sampler=sampler, num_workers=0, drop_last=True
    )
    batches = len(indices) // config["effective_batch"]
    micro_per_step = config["effective_batch"] // config["microbatch"]
    accumulator = BatchAccumulator(engine.refiner, config["effective_batch"], engine.weights)
    cooling = CoolingThrottle(LOCAL / "cooling_settings.json")
    print(
        json.dumps(
            dict(
                mode=args.mode,
                training_clips=len(indices),
                updates_per_epoch=batches,
                maximum_epochs=config["epochs"],
                scope=manifest["scope"],
            )
        ),
        flush=True,
    )
    for epoch in range(start_epoch, config["epochs"]):
        engine.refiner.train()
        epoch_losses = []
        simple_sum, contact_sum = 0.0, 0.0
        update_start = time.monotonic()
        for micro, item in enumerate(loader):
            if micro >= batches * micro_per_step:
                break
            micro_start = time.monotonic()
            prepared = engine.prepare(item)
            losses, parts, _ = engine.loss(prepared)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise FloatingPointError("Nonfinite training loss")
            accumulator.add(losses, parts, config["microbatch"])
            simple_sum += float(losses["simple_loss"].detach()) / micro_per_step
            contact_sum += float(losses["contact_loss"].detach()) / micro_per_step
            torch.cuda.synchronize()
            cooling.wait(time.monotonic() - micro_start)
            if (micro + 1) % micro_per_step:
                continue
            penetration = sum(
                accumulator.values[side + 1] / max(accumulator.denominators[side], 1) for side in range(2)
            )
            value = accumulator.step(optimizer)
            step += 1
            epoch_losses.append(value)
            record = dict(
                epoch=epoch + 1,
                step=step,
                batch=(micro + 1) // micro_per_step,
                combined_loss=value,
                simple_loss=simple_sum,
                contact_loss=contact_sum,
                penetration_loss=penetration,
                seconds=time.monotonic() - update_start,
            )
            with (output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            simple_sum, contact_sum = 0.0, 0.0
            update_start = time.monotonic()
            if args.mode == "smoke":
                break
        report = None
        if args.mode == "smoke" or epoch == 0 or (epoch + 1) % config["validation_interval"] == 0:
            report = engine.evaluate(groups["validation"], "validation", limit_windows=2 if args.mode == "smoke" else 0)
            score = report["totals"]["after"]["combined_loss"]
            report["epoch"] = epoch + 1
            save_json(output / f"validation_{epoch + 1:04d}.json", report)
            print(json.dumps(dict(epoch=epoch + 1, validation=report["totals"])), flush=True)
            if score < best:
                best = score
                save_checkpoint(
                    output / "best_validation.pth",
                    dict(
                        model=engine.refiner.state_dict(),
                        epoch=epoch,
                        validation_loss=score,
                        selection_split="validation",
                        protocol=protocol,
                    ),
                )
        state = dict(
            model=engine.refiner.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            step=step,
            best_validation_loss=best,
            rng=rng_state(),
            protocol=protocol,
        )
        save_checkpoint(output / "latest.pth", state)
        if args.mode == "smoke":
            replay = engine.evaluate(groups["validation"], "validation", limit_windows=2)
            assert replay["totals"] == report["totals"], "Validation randomness is not repeatable"
            loaded = torch.load(output / "latest.pth")
            assert loaded["protocol"] == protocol
            for key, value in engine.refiner.state_dict().items():
                assert torch.equal(value, loaded["model"][key]), "Checkpoint roundtrip failed"
            optimizer.load_state_dict(loaded["optimizer"])
            restore_rng(loaded["rng"])
            save_json(
                output / "smoke_complete.json",
                dict(
                    training_updates=1,
                    validation_windows=2,
                    deterministic_validation=True,
                    checkpoint_roundtrip=True,
                    test_evaluated=False,
                ),
            )
            return
    save_json(
        output / "training_complete.json",
        dict(
            epochs=config["epochs"],
            step=step,
            best_validation_loss=best,
            selection="validation only; test not evaluated",
        ),
    )


if __name__ == "__main__":
    main()
