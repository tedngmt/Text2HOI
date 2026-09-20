# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Check accumulated updates against a full batch with unequal contact counts."""

import copy

import torch
from accumulation import BatchAccumulator

torch.manual_seed(42)
full = torch.nn.Linear(3, 2).double()
chunked = copy.deepcopy(full)
x = torch.randn(6, 3, dtype=torch.float64)
mask = torch.tensor([[0, 0], [0, 0], [1, 0], [1, 1], [1, 0], [1, 1]], dtype=torch.bool)
weights = dict(lambda_simple=1.0, lambda_contact=5.0, lambda_penet=1.0)
optim_full = torch.optim.SGD(full.parameters(), lr=0.01)
optim_chunked = torch.optim.SGD(chunked.parameters(), lr=0.01)
y = full(x)
expected = y.square().mean() + 5 * y.abs().mean()
for side in range(2):
    expected = expected + y[:, side][mask[:, side]].square().mean()
expected.backward()
optim_full.step()
accumulator = BatchAccumulator(chunked, 6, weights)
for start in range(0, 6, 2):
    y = chunked(x[start : start + 2])
    parts = []
    for side in range(2):
        selected = mask[start : start + 2, side]
        parts.append((y[:, side][selected].square().sum(), int(selected.sum())))
    accumulator.add(dict(simple_loss=y.square().mean(), contact_loss=y.abs().mean()), parts, 2)
actual = accumulator.step(optim_chunked)
assert abs(actual - float(expected)) < 1e-12, (actual, float(expected))
for reference, result in zip(full.parameters(), chunked.parameters()):
    torch.testing.assert_close(reference, result, rtol=1e-12, atol=1e-12)
print("PASS: loss and parameter updates match the full batch, including empty penetration groups")
