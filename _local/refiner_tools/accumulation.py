# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Accumulate the upstream loss with batch-wide penetration denominators.

Naively averaging microbatch penetration means changes the official objective.
Keep separate numerator gradients and divide by the total interior-point counts
only when all microbatches of the effective batch have been evaluated.
"""

import torch


class BatchAccumulator:
    def __init__(self, model, effective_batch, weights):
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.effective_batch = effective_batch
        self.weights = weights
        self.reset()

    def reset(self):
        self.buffers = [[None for _ in self.parameters] for _ in range(3)]
        self.denominators = [0, 0]
        self.values = [0.0, 0.0, 0.0]
        self.samples = 0

    def add(self, losses, penetration_parts, sample_count):
        base = (
            losses["simple_loss"] * self.weights["lambda_simple"]
            + losses["contact_loss"] * self.weights["lambda_contact"]
        )
        components = [base * sample_count / self.effective_batch]
        for side, (numerator, denominator) in enumerate(penetration_parts):
            components.append(numerator * self.weights["lambda_penet"])
            self.denominators[side] += denominator
        self.samples += sample_count
        for component_index, component in enumerate(components):
            self.values[component_index] += float(component.detach())
            if not component.requires_grad:
                continue
            gradients = torch.autograd.grad(
                component, self.parameters, retain_graph=component_index < 2, allow_unused=True
            )
            finite = [torch.isfinite(g).all() for g in gradients if g is not None]
            if finite and not torch.stack(finite).all():
                raise FloatingPointError("Nonfinite gradient; optimizer not advanced")
            for index, gradient in enumerate(gradients):
                if gradient is None:
                    continue
                if self.buffers[component_index][index] is None:
                    self.buffers[component_index][index] = gradient.detach()
                else:
                    self.buffers[component_index][index].add_(gradient.detach())

    def step(self, optimizer):
        if self.samples != self.effective_batch:
            raise ValueError("Incomplete effective batch")
        divisors = [1, max(self.denominators[0], 1), max(self.denominators[1], 1)]
        optimizer.zero_grad(set_to_none=True)
        for index, parameter in enumerate(self.parameters):
            for group, divisor in zip(self.buffers, divisors):
                gradient = group[index]
                if gradient is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = gradient / divisor
                else:
                    parameter.grad.add_(gradient, alpha=1 / divisor)
        optimizer.step()
        value = sum(v / d for v, d in zip(self.values, divisors))
        self.reset()
        return value


def capture_penetration_parts():
    """Install the upstream formula with a per-call numerator/count capture."""
    import lib.utils.loss as upstream

    parts = []

    def measured(hand_verts, hand_normal, obj_pc, valid_mask_hand):
        nn_dist, nn_idx = upstream.get_NN(obj_pc, hand_verts)
        interior = upstream.get_interior(hand_normal, hand_verts, obj_pc, nn_idx)
        distances = nn_dist.sqrt()[valid_mask_hand]
        interior = interior[valid_mask_hand]
        count = int(interior.sum())
        numerator = distances[interior].sum() if count else distances.new_zeros(())
        parts.append((numerator, count))
        return numerator / max(count, 1)

    upstream.get_penet_hand_obj_loss = measured
    return parts
