import json
import tempfile
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import BetaDistribution


results = []
for dtype, values in [
    (torch.float16, (40000.0, 30000.0)),
    (torch.float32, (3e38, 2e38)),
    (torch.float16, (0.0, 60000.0)),
    (torch.float32, (0.0, 3e38)),
    (torch.float32, (3e38, 0.0)),
    (torch.float64, (0.0, 1e308)),
    (torch.float64, (1e308, 0.0)),
]:
    raw = torch.tensor(values, dtype=dtype).reshape(1, 2, 1).requires_grad_()
    with localcontext() as context:
        context.prec = 100
        concentrations, slopes = [], []
        for value in raw.detach().double().flatten().tolist():
            value = Decimal.from_float(value)
            if value > 1000:
                concentrations.append(value + 1)
                slopes.append(Decimal(1))
            else:
                exponential = value.exp()
                concentrations.append((1 + exponential).ln() + 1)
                slopes.append(exponential / (1 + exponential))
        alpha, beta = concentrations
        total = alpha + beta
        reference_mean = alpha / total
        reference_gradient = [slopes[0] * beta / total**2, -slopes[1] * alpha / total**2]
    expected = torch.tensor(float(reference_mean), dtype=dtype).reshape(1, 1)
    expected_gradient = torch.tensor([float(x) for x in reference_gradient], dtype=dtype).reshape_as(raw)
    distribution = BetaDistribution(1, action_range=(0.0, 1.0))
    actual = distribution.deterministic_output(raw)
    actual.sum().backward()
    smallest = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
    mean_ok = torch.allclose(actual, expected, rtol=0.005, atol=smallest)
    gradient_ok = torch.allclose(raw.grad, expected_gradient, rtol=0.01, atol=smallest)
    results.append({
        "dtype": str(dtype), "raw": values, "actual": actual.item(),
        "decimal_mean": str(reference_mean), "mean_ok": mean_ok,
        "actual_gradient": raw.grad.flatten().tolist(),
        "decimal_gradient": [str(x) for x in reference_gradient], "gradient_ok": gradient_ok,
    })

for dtype, values in [(torch.float16, (40000.0, 30000.0)), (torch.float32, (3e38, 2e38))]:
    obs = TensorDict({"policy": torch.ones(2, 3, dtype=dtype)}, batch_size=[2])
    model = MLPModel(
        obs, {"actor": ["policy"]}, "actor", 1, hidden_dims=[4],
        distribution_cfg={"class_name": "BetaDistribution", "action_range": (0.0, 1.0)},
    ).to(dtype=dtype).eval()
    with torch.no_grad():
        model.mlp.output_layer.weight.zero_()
        model.mlp.output_layer.bias.copy_(torch.tensor(values, dtype=dtype))
    alpha, beta = [float(x) for x in torch.nn.functional.softplus(model.mlp.output_layer.bias.double()) + 1.0]
    expected = alpha / (alpha + beta)
    native = model(obs).detach().numpy()
    scripted = torch.jit.script(model.as_jit())(obs["policy"]).detach().numpy()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "policy.onnx"
        torch.onnx.export(model.as_onnx(), (obs["policy"],), path, opset_version=18, input_names=["obs"], output_names=["actions"])
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        exported = session.run(None, {"obs": obs["policy"].numpy()})[0]
    results.append({
        "dtype": str(dtype), "expected": expected,
        "native": native.tolist(), "scripted": scripted.tolist(), "onnxruntime": exported.tolist(),
        "native_ok": bool(np.allclose(native, expected, rtol=0.005, atol=1e-5)),
        "scripted_ok": bool(np.allclose(scripted, expected, rtol=0.005, atol=1e-5)),
        "onnxruntime_ok": bool(np.allclose(exported, expected, rtol=0.005, atol=1e-5)),
    })

print(json.dumps({"torch": torch.__version__, "onnxruntime": onnxruntime.__version__, "results": results}, indent=2))
assert all(value for result in results for key, value in result.items() if key.endswith("_ok"))
