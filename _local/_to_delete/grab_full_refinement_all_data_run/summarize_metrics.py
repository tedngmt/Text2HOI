# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Summarize completed training updates, excluding replayed partial epochs."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
run = root / "run_official"
output = root / "metrics"
output.mkdir(exist_ok=True)
steps = {}
for line in (run / "steps.jsonl").read_text().splitlines():
    row = json.loads(line)
    steps[row["step"]] = row  # A resumed partial epoch is replayed, not extra training.
assert set(steps) == set(range(1, 13001)), "Expected all 13,000 completed updates"
epochs = []
for epoch in range(1, 651):
    rows = [steps[step] for step in range((epoch - 1) * 20 + 1, epoch * 20 + 1)]
    assert all(row["epoch"] == epoch for row in rows)
    epochs.append({"epoch": epoch, "mean_training_loss": sum(row["loss"] for row in rows) / 20})
best = min(epochs, key=lambda row: row["mean_training_loss"])
completion = json.loads((run / "training_complete.json").read_text())
assert abs(best["mean_training_loss"] - completion["best_loss"]) < 1e-9
with (output / "epoch_losses.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["epoch", "mean_training_loss"])
    writer.writeheader()
    writer.writerows(epochs)
fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
for axis, start in zip(axes, (0, 20)):
    values = epochs[start:]
    axis.plot([r["epoch"] for r in values], [r["mean_training_loss"] for r in values], linewidth=1)
    axis.axvline(best["epoch"], color="tab:orange", linestyle="--", label=f"Selected epoch {best['epoch']}")
    axis.set(xlabel="Epoch", ylabel="Mean combined training loss")
    axis.grid(alpha=0.25)
    axis.legend()
axes[0].set_title("GRAB Text2HOI refiner: training objective (not validation accuracy)")
axes[1].set_title("Detail after epoch 20")
fig.savefig(output / "training_loss.png", dpi=160)
plt.close(fig)
summary = {
    "epochs": 650,
    "updates": len(steps),
    "first_epoch": epochs[0],
    "last_epoch": epochs[-1],
    "selected_epoch": best,
    "selection": "Lowest training loss; no held-out validation",
    "available": ["Combined training loss", "Per-video mean vertex displacement in millimeters"],
    "not_measured": [
        "Held-out accuracy",
        "Before/after penetration improvement",
        "Contact accuracy",
        "Physical grasp success",
        "Text-to-motion generation quality",
    ],
    "logging_limitation": "Individual reconstruction, penetration, and contact loss histories were not saved",
}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
