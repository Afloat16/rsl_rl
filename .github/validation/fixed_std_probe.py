# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import json
import math
import torch
from rsl_rl.algorithms import SAC
from tests.algorithms.test_sac import test_fixed_standard_deviation_policy_can_train

records = []
original = SAC.update
def record_update(self):
    before_finite = all(bool(torch.isfinite(p).all()) for p in self.actor.parameters())
    losses, stats = original(self)
    std = self.actor.output_std
    records.append({
        "init_std": self.actor.distribution.init_std,
        "bounds": [math.exp(value) for value in self.actor.distribution.log_std_range],
        "initial_parameters_finite": before_finite,
        "final_parameters_finite": all(bool(torch.isfinite(p).all()) for p in self.actor.parameters()),
        "std_finite": bool(torch.isfinite(std).all()),
        "std_min": std.min().item() if bool(torch.isfinite(std).all()) else None,
        "std_max": std.max().item() if bool(torch.isfinite(std).all()) else None,
        "losses": {key: value if math.isfinite(value) else None for key, value in losses.items()},
    })
    return losses, stats
SAC.update = record_update
for init_std in (None, 0.5):
    try:
        test_fixed_standard_deviation_policy_can_train(init_std)
        records[-1]["matches_training_contract"] = True
    except AssertionError as error:
        records[-1]["matches_training_contract"] = False
        records[-1]["failure"] = str(error)[:300]
SAC.update = original
print("FIXED_STD_SAC_PROBE=" + json.dumps(records, allow_nan=False), flush=True)
