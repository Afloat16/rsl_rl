import json
import platform
import torch

from rsl_rl.modules.normalization import EmpiricalNormalization
from tests.algorithms.test_distributed import init_gloo, spawn_gloo


def worker(rank, world_size, store_file):
    init_gloo(rank, world_size, store_file)
    try:
        normalizer = EmpiricalNormalization(2)
        first = torch.tensor([[0.0, 2.0], [4.0, 6.0]])
        second = first + 8
        records = []
        for name, local in [
            ("one_empty_rank", first if rank == 1 else first[:0]),
            ("globally_empty_update", first[:0]),
            ("later_nonempty_update", second if rank == 0 else first[:0]),
        ]:
            normalizer.update(local)
            records.append({
                "stage": name, "count": int(normalizer.count),
                "mean": normalizer.mean.tolist(), "variance": normalizer._var.squeeze(0).tolist(),
                "std": normalizer.std.tolist(),
            })
        torch.manual_seed(0)
        learner = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(learner.parameters(), lr=0.1)
        query = torch.tensor([[2.0, 4.0], [10.0, 12.0]])
        before = learner.weight.detach().clone()
        output = normalizer(query)
        loss = learner(output).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if rank == 0:
            print("EMPTY_NORMALIZER_PROBE " + json.dumps({
                "python": platform.python_version(), "torch": torch.__version__,
                "backend": torch.distributed.get_backend(), "world_size": world_size,
                "records": records, "normalized_inputs_finite": bool(torch.isfinite(output).all()),
                "optimizer_loss": float(loss.detach()),
                "optimizer_parameters_finite": bool(torch.isfinite(learner.weight).all()),
                "parameter_change": float((learner.weight.detach() - before).abs().max()),
            }), flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    spawn_gloo(worker)
