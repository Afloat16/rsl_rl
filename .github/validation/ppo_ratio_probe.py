# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import json
import math
import torch
from rsl_rl.algorithms import PPO
from tests.algorithms.test_ppo import TestPPOLosses

records = []
original_update = PPO.update
def record_update(self):
    assert not self.storage.advantages.requires_grad
    observations = self.storage.observations.flatten(0, 1)
    actions = self.storage.actions.flatten(0, 1)
    with torch.no_grad():
        self.actor(observations, stochastic_output=True)
        log_ratio = self.actor.get_output_log_prob(actions) - self.storage.actions_log_prob.flatten()
        record = {
            "num_actions": actions.shape[-1],
            "advantage": self.storage.advantages.item(),
            "log_ratio": log_ratio.item(),
            "log_ratio_finite": bool(torch.isfinite(log_ratio).all()),
        }
    losses, stats = original_update(self)
    record.update({
        "surrogate_loss": losses["surrogate"] if math.isfinite(losses["surrogate"]) else None,
        "loss_finite": math.isfinite(losses["surrogate"]),
        "all_parameters_finite": all(bool(torch.isfinite(p).all()) for p in self.actor.parameters()),
        "all_gradients_finite": all(bool(torch.isfinite(p.grad).all()) for p in self.actor.parameters() if p.grad is not None),
    })
    records.append(record)
    return losses, stats
PPO.update = record_update
cases = [(64, 2.0, 0.0), (64, 0.0, 0.0), (64, 0.0, 10.0), (4, 2.0, 0.0), (4, -2.0, 0.0), (4, -2.0, 4.0), (4, 2.0, 4.0)]
for num_actions, advantage, offset in cases:
    try:
        TestPPOLosses().test_clipped_surrogate_preserves_finite_gradients(num_actions, advantage, offset)
        records[-1]["matches_float64_loss_and_gradients"] = True
    except AssertionError as error:
        records[-1]["matches_float64_loss_and_gradients"] = False
        records[-1]["failure"] = str(error)[:300]
    records[-1]["mean_offset_in_std_units"] = offset
PPO.update = original_update
print("PPO_RATIO_PROBE=" + json.dumps(records, allow_nan=False), flush=True)

for clip in (0.0, 0.2, 1.2):
    upper = math.log1p(clip)
    for advantage in (-2.0, 0.0, 2.0):
        logits = torch.tensor([-100.0, -1.0, upper - 1e-6, upper, upper + 1e-6, 1.0, 100.0], dtype=torch.float64, requires_grad=True)
        reference_ratio = logits.exp()
        expected = torch.maximum(-advantage * reference_ratio, -advantage * reference_ratio.clamp(1 - clip, 1 + clip))
        bounded = logits.clamp(max=upper) if advantage >= 0 else logits
        actual_ratio = bounded.exp()
        actual = torch.maximum(-advantage * actual_ratio, -advantage * actual_ratio.clamp(1 - clip, 1 + clip))
        expected_grad = torch.autograd.grad(expected.sum(), logits, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(actual.sum(), logits)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)
print("PPO_RATIO_POLICY_GRADIENT_ORACLE=63 cases passed", flush=True)
