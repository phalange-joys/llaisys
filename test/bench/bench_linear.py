"""Local micro-benchmark for Ops.linear (llaisys vs torch, CPU).

Runs (512,4096) x (4096,4096) GEMM for f32/f16/bf16 with warmup and
multiple timed iterations, reporting min/median/mean for both backends.

Usage:
    source venv/bin/activate
    python test/bench/bench_linear.py [--iters N] [--warmup N] [--dtype f32,f16,bf16]
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import llaisys
import torch

from test_utils import check_equal, random_tensor


def bench(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts), statistics.mean(ts)


def bench_linear(dtype_name, iters, warmup, device_name="cpu"):
    out_shape, x_shape, w_shape = (1, 4096), (1, 4096), (4096, 4096)
    x, x_ = random_tensor(x_shape, dtype_name, device_name, scale=0.1)
    w, w_ = random_tensor(w_shape, dtype_name, device_name, scale=0.01)
    bias, bias_ = random_tensor((w_shape[0],), dtype_name, device_name)
    out, out_ = random_tensor(out_shape, dtype_name, device_name)

    def t_linear():
        torch.nn.functional.linear(x, w, bias, out=out)

    def l_linear():
        llaisys.Ops.linear(out_, x_, w_, bias_)

    # correctness first
    llaisys.Ops.linear(out_, x_, w_, bias_)
    torch.nn.functional.linear(x, w, bias, out=out)
    ok = check_equal(
        out_,
        out,
        atol=1e-2 if dtype_name != "f32" else 1e-5,
        rtol=1e-2 if dtype_name != "f32" else 1e-5,
    )
    t_min, t_med, t_mean = bench(t_linear, warmup, iters)
    l_min, l_med, l_mean = bench(l_linear, warmup, iters)
    gflops = 2.0 * out_shape[0] * out_shape[1] * w_shape[1] / 1e9
    print(f"  [{dtype_name}] correct={ok}")
    print(
        f"    torch   min={t_min:7.3f}s med={t_med:7.3f}s mean={t_mean:7.3f}s "
        f"({gflops / t_min:7.1f} GFLOP/s)"
    )
    print(
        f"    llaisys min={l_min:7.3f}s med={l_med:7.3f}s mean={l_mean:7.3f}s "
        f"({gflops / l_min:7.1f} GFLOP/s, torch/llaisys={t_min / l_min:5.2f}x)"
    )
    return (dtype_name, t_min, t_med, t_mean, l_min, l_med, l_mean, ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dtype", type=str, default="f32,f16,bf16")
    args = ap.parse_args()
    dtypes = args.dtype.split(",")
    print(
        f"bench_linear: threads={os.environ.get('OMP_NUM_THREADS', 'default')} "
        f"cpu_count={os.cpu_count()} torch_threads={torch.get_num_threads()}"
    )
    for d in dtypes:
        bench_linear(d, args.iters, args.warmup)
        print()


if __name__ == "__main__":
    main()
