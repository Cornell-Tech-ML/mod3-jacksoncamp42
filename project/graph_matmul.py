import random
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import minitorch

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend, size=16) -> None:
    batch_size = 2
    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y


def plot_results(times):
    plt.figure(figsize=(10, 6))
    sizes = list(times.keys())

    # Extract times for each backend
    fast_times = [times[size]["fast"] for size in sizes]
    gpu_times = [times[size]["gpu"] for size in sizes]

    # Create log-scale plot
    plt.plot(sizes, fast_times, "o-", label="CPU (Fast)", linewidth=2)
    plt.plot(sizes, gpu_times, "s-", label="GPU (CUDA)", linewidth=2)

    plt.xlabel("Matrix Size (N x N)")
    plt.ylabel("Time (seconds)")
    plt.title("Matrix Multiplication Performance Comparison")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    # Use log scale for better visualization
    plt.yscale("log")
    plt.xscale("log", base=2)

    # Add value annotations
    for size in sizes:
        plt.annotate(
            f'{times[size]["fast"]:.3f}s',
            (size, times[size]["fast"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.annotate(
            f'{times[size]["gpu"]:.3f}s',
            (size, times[size]["gpu"]),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
        )

    plt.tight_layout()
    plt.savefig("matmul_benchmark.png")
    plt.close()


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}

    # Test different matrix sizes
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}

        fast_times = []
        gpu_times = []

        for _ in range(ntrials):
            # Time CPU (Fast) implementation
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            # Time GPU implementation
            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])
    print()

    # Print timing summary
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f" {b}: {t:.5f}")

    # Generate plot
    plot_results(times)
    print("\nPlot saved as 'matmul_benchmark.png'")
