# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import copy
import json
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import RNNModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

results = []
for rnn_type in ("gru", "lstm"):
    torch.manual_seed(7)
    obs = make_obs(4, 3)
    groups = {"actor": ["policy"], "critic": ["policy"]}
    cfg = {"hidden_dims": [8], "rnn_hidden_dim": 8, "rnn_type": rnn_type}
    actor = RNNModel(
        obs, groups, "actor", 2,
        distribution_cfg={"class_name": "HeteroscedasticGaussianDistribution", "std_type": "log"},
        **cfg,
    )
    critic = RNNModel(obs, groups, "critic", 1, **cfg)
    with torch.no_grad():
        actor.mlp.output_layer.weight[2:].fill_(0.5)
        for model in (actor, critic):
            model(obs)
            model.reset(torch.ones(4))
    reference = copy.deepcopy(actor)
    storage = RolloutStorage("rl", 4, 1, obs, [2])
    alg = PPO(
        actor, critic, storage, num_learning_epochs=1, num_mini_batches=1, learning_rate=1e-3,
        desired_kl=1e-6, value_loss_coef=0.0, entropy_coef=1.0, max_grad_norm=1e6,
    )
    with torch.no_grad():
        alg.act(obs)
        alg.process_env_step(obs, torch.zeros(4), torch.zeros(4), {})
        alg.compute_returns(obs)
        storage.advantages.zero_()
    reference(obs, stochastic_output=True)
    entropy_by_env = torch.distributions.Normal(reference.output_mean, reference.output_std).entropy().sum(-1)
    expected_loss = -entropy_by_env.mean()
    expected_loss.backward()
    captured = {}
    original_kl = actor.get_kl_divergence
    def record_kl(old, new):
        result = original_kl(old, new)
        captured["mean_kl"] = result.mean().item()
        captured["new_mean_shape"] = list(new[0].shape)
        return result
    actor.get_kl_divergence = record_kl
    losses, _ = alg.update()
    differences = [
        (a.grad - e.grad).abs().max().item()
        for a, e in zip(actor.parameters(), reference.parameters())
        if a.grad is not None and e.grad is not None
    ]
    results.append({
        "rnn_type": rnn_type, "expected_entropy_loss": expected_loss.item(),
        "actual_entropy_loss": losses["entropy"], "entropy_by_env": entropy_by_env.detach().tolist(),
        "actual_learning_rate": alg.learning_rate, "expected_learning_rate": 1e-3,
        "max_gradient_absolute_error": max(differences), **captured,
    })
print("RECURRENT_TIME_AXIS_PROBE=" + json.dumps(results), flush=True)
