"""Check running-normalizer outputs against exact moments and 100-digit Decimal square roots."""

import argparse
import json
import subprocess
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import torch


def reference(values, epsilon):
    with localcontext() as context:
        context.prec = 100
        samples = [Fraction.from_float(float(value)) for value in values]
        mean = sum(samples) / len(samples)
        variance = sum((value - mean) ** 2 for value in samples) / len(samples)
        std = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
        shifted = [value - mean for value in samples]
        normalized = [
            (Decimal(value.numerator) / Decimal(value.denominator)) / (std + Decimal.from_float(epsilon))
            for value in shifted
        ]
        return float(mean), float(std), [float(value) for value in normalized]


def check(normalizer_type, dtype, values, sizes):
    observations = torch.tensor(values, dtype=dtype).unsqueeze(-1)
    norm = normalizer_type(1).to(dtype=dtype)
    offset = 0
    reports = []
    for size in sizes:
        norm.update(observations[offset : offset + size])
        offset += size
        mean, std, normalized = reference(observations[:offset, 0].tolist(), norm.eps)
        expected = torch.tensor(normalized, dtype=dtype).unsqueeze(-1)
        actual = norm(observations[:offset])
        tolerance = 2e-6 if dtype == torch.float32 else 2e-14
        passed = (
            torch.allclose(norm.mean, torch.tensor([mean], dtype=dtype), rtol=tolerance, atol=1e-14)
            and torch.allclose(norm.std, torch.tensor([std], dtype=dtype), rtol=tolerance, atol=1e-14)
            and torch.allclose(actual, expected, rtol=tolerance, atol=tolerance)
        )
        reports.append(
            {
                "count": int(norm.count),
                "mean": norm.mean.item(),
                "reference_mean": mean,
                "std": norm.std.item(),
                "reference_std": std,
                "normalized_finite": bool(actual.isfinite().all()),
                "normalized_max_abs_error": float((actual - expected).abs().max()),
                "passed": bool(passed),
            }
        )
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("--ref")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    path = "rsl_rl/modules/normalization.py"
    source = (
        subprocess.check_output(["git", "-C", str(args.repository), "show", f"{args.ref}:{path}"], text=True)
        if args.ref
        else (args.repository / path).read_text()
    )
    namespace = {}
    exec(compile(source, str(args.repository / path), "exec"), namespace)
    cases = [
        ("constant_float32", torch.float32, [2e38] * 12, [8, 3, 1]),
        ("constant_float64", torch.float64, [1e308] * 12, [8, 3, 1]),
        ("opposite_means_float32", torch.float32, [-1.5e19, 1.5e19], [1, 1]),
        ("opposite_means_float64", torch.float64, [-1e154, 1e154], [1, 1]),
        ("weighted_float64", torch.float64, [0, 1, 1], [1, 2]),
        ("pooled_float64", torch.float64, [0, 1, 1, 2, 3, 4, 8, 9], [8]),
        ("partitioned_float64", torch.float64, [0, 1, 1, 2, 3, 4, 8, 9], [1, 2, 5]),
    ]
    reports = {name: check(namespace["EmpiricalNormalization"], dtype, values, sizes) for name, dtype, values, sizes in cases}
    print(json.dumps({"reference_precision": 100, "cases": reports}, indent=2, allow_nan=True))
    passed = all(row["passed"] for rows in reports.values() for row in rows)
    if args.require_pass and not passed:
        raise AssertionError("Real normalization disagrees with exact pooled moments and high-precision outputs")


if __name__ == "__main__":
    main()
