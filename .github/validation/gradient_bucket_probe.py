import argparse
import json
import platform
import torch

from rsl_rl.utils import reduce_gradients_in_buckets
from tests.algorithms.test_distributed import init_gloo, spawn_gloo


def worker(rank, world_size, store_file, strided):
    init_gloo(rank, world_size, store_file)
    try:
        params = []
        for shape, dtype, value in [((4000, 100), torch.float16, 40000), ((500, 100), torch.float32, 3 + rank)]:
            grad = torch.full(shape, value, dtype=dtype)
            if strided:
                grad = grad.t()
            parameter = torch.nn.Parameter(torch.zeros_like(grad))
            parameter.grad = grad
            params.append(parameter)
        originals = [parameter.grad for parameter in params]
        records = []
        all_reduce = torch.distributed.all_reduce

        def record_all_reduce(value, op):
            records.append({"dtype": str(value.dtype), "bytes": value.numel() * value.element_size()})
            return all_reduce(value, op=op)

        torch.distributed.all_reduce = record_all_reduce
        try:
            reduce_gradients_in_buckets(params, world_size, bucket_mb=1)
        finally:
            torch.distributed.all_reduce = all_reduce
        expected = [40000, 3 + (world_size - 1) / 2]
        for parameter, original, reference in zip(params, originals, expected):
            assert parameter.grad is original
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter.grad, reference), rtol=0, atol=0)
        if rank == 0:
            print("BUCKET_PROBE " + json.dumps({
                "python": platform.python_version(), "torch": torch.__version__,
                "backend": torch.distributed.get_backend(), "world_size": world_size,
                "threads": torch.get_num_threads(), "strided": strided, "budget_bytes": 1048576,
                "logical_gradient_bytes": 1000000, "collectives": records,
                "max_buffer_bytes": max(record["bytes"] for record in records),
                "total_buffer_bytes": sum(record["bytes"] for record in records),
                "averages": [float(parameter.grad.flatten()[0]) for parameter in params],
                "finite_averages_and_original_storage": True,
            }), flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strided", action="store_true")
    args = parser.parse_args()
    spawn_gloo(worker, args.strided)
