"""Comprehensive benchmarking script for Flash Attention implementations.

Compares Triton kernel, PyTorch SDPA, and different backends across metrics.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import rootutils
import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.flop_counter import FlopCounterMode

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from zatom.models.kernels.flash_attention import JVPAttn


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    batch_size: int = 4
    num_heads: int = 8
    head_dim: int = 64
    seq_lengths: list[int] = None
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    num_warmup: int = 10
    num_iterations: int = 100
    enable_jvp: bool = False
    is_causal: bool = False

    def __post_init__(self):
        if self.seq_lengths is None:
            self.seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    seq_len: int
    method: str
    time_ms: float
    memory_mb: float
    flops: int | None = None
    tflops: float | None = None
    accuracy_error: float | None = None
    relative_error: float | None = None


class AttentionBenchmark:
    """Benchmark suite for attention implementations."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: list[BenchmarkResult] = []

    def generate_inputs(self, seq_len: int) -> tuple[Tensor, Tensor, Tensor]:
        """Generate random QKV tensors for benchmarking."""
        shape = (self.config.batch_size, self.config.num_heads, seq_len, self.config.head_dim)

        q = torch.randn(shape, dtype=self.config.dtype, device=self.config.device)
        k = torch.randn(shape, dtype=self.config.dtype, device=self.config.device)
        v = torch.randn(shape, dtype=self.config.dtype, device=self.config.device)

        # Normalize to prevent numerical issues
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)

        return q, k, v

    def generate_dual_inputs(self, seq_len: int) -> tuple[Tensor, Tensor, Tensor]:
        """Generate dual inputs for JVP testing."""
        q, k, v = self.generate_inputs(seq_len)

        if self.config.enable_jvp:
            with fwAD.dual_level():
                # Create tangent vectors
                q_t = torch.randn_like(q) * 0.01
                k_t = torch.randn_like(k) * 0.01
                v_t = torch.randn_like(v) * 0.01

                # Pack into dual tensors
                q = fwAD.make_dual(q, q_t)
                k = fwAD.make_dual(k, k_t)
                v = fwAD.make_dual(v, v_t)

        return q, k, v

    def measure_memory(self, func, *args, **kwargs) -> float:
        """Measure peak memory usage of a function in MB."""
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        func(*args, **kwargs)

        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB

        return peak_memory

    def measure_time(self, func, *args, **kwargs) -> float:
        """Measure execution time of a function in milliseconds."""
        # Warmup
        for _ in range(self.config.num_warmup):
            func(*args, **kwargs)

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(self.config.num_iterations):
            func(*args, **kwargs)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        return (end_time - start_time) * 1000 / self.config.num_iterations

    def measure_flops(self, func, *args, **kwargs) -> int:
        """Measure FLOPs of a function."""
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            func(*args, **kwargs)
        return flop_counter.get_total_flops()

    def compute_accuracy(self, output: Tensor, reference: Tensor) -> tuple[float, float]:
        """Compute absolute and relative error between output and reference."""
        abs_error = (output - reference).abs().max().item()
        rel_error = ((output - reference).abs() / (reference.abs() + 1e-8)).max().item()
        return abs_error, rel_error

    def benchmark_pytorch_sdpa(self, seq_len: int, backend: SDPBackend) -> BenchmarkResult:  # type: ignore
        """Benchmark PyTorch's scaled_dot_product_attention."""
        q, k, v = self.generate_inputs(seq_len)

        with sdpa_kernel(backend):
            # Time measurement
            time_ms = self.measure_time(
                scaled_dot_product_attention, q, k, v, is_causal=self.config.is_causal
            )

            # Memory measurement
            memory_mb = self.measure_memory(
                scaled_dot_product_attention, q, k, v, is_causal=self.config.is_causal
            )

            # FLOPs measurement
            flops = self.measure_flops(
                scaled_dot_product_attention, q, k, v, is_causal=self.config.is_causal
            )

        backend_name = str(backend).split(".")[-1]
        return BenchmarkResult(
            seq_len=seq_len,
            method=f"PyTorch_{backend_name}",
            time_ms=time_ms,
            memory_mb=memory_mb,
            flops=flops,
            tflops=flops / (time_ms * 1e9) if flops else None,
        )

    def benchmark_triton_kernel(
        self, seq_len: int, reference_output: Tensor | None = None
    ) -> BenchmarkResult:
        """Benchmark custom Triton kernel."""
        if self.config.enable_jvp:
            q, k, v = self.generate_dual_inputs(seq_len)

            # Time measurement
            time_ms = self.measure_time(JVPAttn.fwd_dual, q, k, v, causal=self.config.is_causal)

            # Memory measurement
            memory_mb = self.measure_memory(
                JVPAttn.fwd_dual, q, k, v, causal=self.config.is_causal
            )

            # Compute output for accuracy check
            output = JVPAttn.fwd_dual(q, k, v, causal=self.config.is_causal)
            if isinstance(output, tuple):
                output = output[0]
            output = fwAD.unpack_dual(output)[0] if fwAD.is_dual(output) else output

        else:
            q, k, v = self.generate_inputs(seq_len)

            # Time measurement
            time_ms = self.measure_time(JVPAttn.apply, q, k, v, self.config.is_causal)

            # Memory measurement
            memory_mb = self.measure_memory(JVPAttn.apply, q, k, v, self.config.is_causal)

            # Compute output for accuracy check
            output = JVPAttn.apply(q, k, v, self.config.is_causal)

        # Accuracy measurement
        abs_error, rel_error = None, None
        if reference_output is not None:
            abs_error, rel_error = self.compute_accuracy(output, reference_output)

        return BenchmarkResult(
            seq_len=seq_len,
            method="Triton_JVP" if self.config.enable_jvp else "Triton",
            time_ms=time_ms,
            memory_mb=memory_mb,
            accuracy_error=abs_error,
            relative_error=rel_error,
        )

    def run_benchmarks(self) -> pd.DataFrame:
        """Run all benchmarks across sequence lengths."""
        print("Starting benchmarks...")
        print(f"Config: {self.config}")

        for seq_len in self.config.seq_lengths:
            print(f"\nBenchmarking sequence length: {seq_len}")

            # Get reference output from PyTorch MATH backend
            q, k, v = self.generate_inputs(seq_len)
            with sdpa_kernel(SDPBackend.MATH):
                reference_output = scaled_dot_product_attention(
                    q, k, v, is_causal=self.config.is_causal
                )

            # Benchmark PyTorch implementations
            for backend in [
                SDPBackend.MATH,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]:
                try:
                    result = self.benchmark_pytorch_sdpa(seq_len, backend)
                    self.results.append(result)
                    print(f"  {result.method}: {result.time_ms:.2f}ms, {result.memory_mb:.2f}MB")
                except Exception as e:
                    print(f"  {backend} failed: {e}")

            # Benchmark Triton kernel
            try:
                result = self.benchmark_triton_kernel(seq_len, reference_output)
                self.results.append(result)
                print(f"  {result.method}: {result.time_ms:.2f}ms, {result.memory_mb:.2f}MB")
                if result.accuracy_error is not None:
                    print(
                        f"    Accuracy - Abs Error: {result.accuracy_error:.2e}, Rel Error: {result.relative_error:.2e}"
                    )
            except Exception as e:
                print(f"  Triton kernel failed: {e}")

            # Clear cache between runs
            torch.cuda.empty_cache()
            gc.collect()

        return pd.DataFrame([vars(r) for r in self.results])

    def plot_results(self, df: pd.DataFrame):
        """Create visualization plots for benchmark results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Plot 1: Execution time vs sequence length
        ax = axes[0, 0]
        for method in df["method"].unique():
            method_df = df[df["method"] == method]
            ax.plot(
                method_df["seq_len"], method_df["time_ms"], marker="o", label=method, linewidth=2
            )
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Execution Time Comparison")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Memory usage vs sequence length
        ax = axes[0, 1]
        for method in df["method"].unique():
            method_df = df[df["method"] == method]
            ax.plot(
                method_df["seq_len"], method_df["memory_mb"], marker="s", label=method, linewidth=2
            )
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory Usage Comparison")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Speedup relative to PyTorch MATH
        ax = axes[0, 2]
        baseline_method = "PyTorch_MATH"
        if baseline_method in df["method"].values:
            baseline_df = df[df["method"] == baseline_method].set_index("seq_len")
            for method in df["method"].unique():
                if method != baseline_method:
                    method_df = df[df["method"] == method].set_index("seq_len")
                    speedup = baseline_df["time_ms"] / method_df["time_ms"]
                    ax.plot(
                        speedup.index, speedup.values, marker="^", label=f"{method}", linewidth=2
                    )
            ax.axhline(y=1, color="k", linestyle="--", alpha=0.5)
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Speedup")
            ax.set_title(f"Speedup vs {baseline_method}")
            ax.set_xscale("log", base=2)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 4: TFLOPS if available
        ax = axes[1, 0]
        df_with_tflops = df.dropna(subset=["tflops"])
        if not df_with_tflops.empty:
            for method in df_with_tflops["method"].unique():
                method_df = df_with_tflops[df_with_tflops["method"] == method]
                ax.plot(
                    method_df["seq_len"],
                    method_df["tflops"],
                    marker="d",
                    label=method,
                    linewidth=2,
                )
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("TFLOPS")
            ax.set_title("Computational Throughput")
            ax.set_xscale("log", base=2)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 5: Memory efficiency (seq_len^2 / memory)
        ax = axes[1, 1]
        for method in df["method"].unique():
            method_df = df[df["method"] == method]
            efficiency = (method_df["seq_len"] ** 2) / method_df["memory_mb"]
            ax.plot(method_df["seq_len"], efficiency, marker="p", label=method, linewidth=2)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Memory Efficiency (seq²/MB)")
        ax.set_title("Memory Efficiency")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: Accuracy comparison for Triton
        ax = axes[1, 2]
        triton_df = df[df["method"].str.contains("Triton")].dropna(subset=["accuracy_error"])
        if not triton_df.empty:
            ax.plot(
                triton_df["seq_len"],
                triton_df["accuracy_error"],
                marker="o",
                label="Absolute Error",
                linewidth=2,
                color="red",
            )
            ax2 = ax.twinx()
            ax2.plot(
                triton_df["seq_len"],
                triton_df["relative_error"],
                marker="s",
                label="Relative Error",
                linewidth=2,
                color="blue",
            )
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Absolute Error", color="red")
            ax2.set_ylabel("Relative Error", color="blue")
            ax.set_title("Triton Kernel Accuracy")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax2.set_yscale("log")
            ax.tick_params(axis="y", labelcolor="red")
            ax2.tick_params(axis="y", labelcolor="blue")
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Attention Kernel Benchmarks (Batch={self.config.batch_size}, Heads={self.config.num_heads}, Dim={self.config.head_dim})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate a text report of the benchmark results."""
        report = []
        report.append("=" * 80)
        report.append("ATTENTION KERNEL BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("\nConfiguration:")
        report.append(f"  Batch Size: {self.config.batch_size}")
        report.append(f"  Number of Heads: {self.config.num_heads}")
        report.append(f"  Head Dimension: {self.config.head_dim}")
        report.append(f"  Data Type: {self.config.dtype}")
        report.append(f"  Causal Mask: {self.config.is_causal}")
        report.append(f"  JVP Enabled: {self.config.enable_jvp}")

        report.append(f"\n{'='*80}")
        report.append("PERFORMANCE SUMMARY")
        report.append("=" * 80)

        # Group by method and compute statistics
        for method in df["method"].unique():
            method_df = df[df["method"] == method]
            report.append(f"\n{method}:")
            report.append(f"  Avg Time: {method_df['time_ms'].mean():.2f} ms")
            report.append(f"  Avg Memory: {method_df['memory_mb'].mean():.2f} MB")

            if "tflops" in method_df.columns and method_df["tflops"].notna().any():
                report.append(f"  Avg TFLOPS: {method_df['tflops'].mean():.2f}")

            if "accuracy_error" in method_df.columns and method_df["accuracy_error"].notna().any():
                report.append(f"  Max Abs Error: {method_df['accuracy_error'].max():.2e}")
                report.append(f"  Max Rel Error: {method_df['relative_error'].max():.2e}")

        # Compute best performer for each metric
        report.append(f"\n{'='*80}")
        report.append("BEST PERFORMERS")
        report.append("=" * 80)

        for seq_len in self.config.seq_lengths:
            seq_df = df[df["seq_len"] == seq_len]
            if not seq_df.empty:
                fastest = seq_df.loc[seq_df["time_ms"].idxmin()]
                most_efficient = seq_df.loc[seq_df["memory_mb"].idxmin()]
                report.append(f"\nSequence Length {seq_len}:")
                report.append(f"  Fastest: {fastest['method']} ({fastest['time_ms']:.2f} ms)")
                report.append(
                    f"  Most Memory Efficient: {most_efficient['method']} ({most_efficient['memory_mb']:.2f} MB)"
                )

        return "\n".join(report)


def main():
    """Main benchmarking function."""
    # Configure benchmarks
    config = BenchmarkConfig(
        batch_size=4,
        num_heads=8,
        head_dim=64,
        seq_lengths=[128, 256, 512, 1024, 2048, 4096],
        dtype=torch.float16,
        device="cuda",
        num_warmup=10,
        num_iterations=50,
        enable_jvp=False,  # Set to True to test JVP functionality
        is_causal=False,
    )

    # Run benchmarks
    benchmark = AttentionBenchmark(config)
    results_df = benchmark.run_benchmarks()

    # Save results
    results_df.to_csv("attention_benchmark_results.csv", index=False)
    print("\nResults saved to attention_benchmark_results.csv")

    # Generate and print report
    report = benchmark.generate_report(results_df)
    print(report)

    with open("attention_benchmark_report.txt", "w") as f:
        f.write(report)
    print("\nReport saved to attention_benchmark_report.txt")

    # Create plots
    fig = benchmark.plot_results(results_df)
    fig.savefig("attention_benchmark_plots.png", dpi=150, bbox_inches="tight")
    print("Plots saved to attention_benchmark_plots.png")

    # Additional evaluation metrics
    print("\n" + "=" * 80)
    print("ADDITIONAL EVALUATION METRICS")
    print("=" * 80)

    # Test with different configurations
    test_configs = [
        {"is_causal": True, "label": "Causal"},
        {"dtype": torch.float32, "label": "FP32"},
        {"enable_jvp": True, "label": "JVP"} if hasattr(JVPAttn, "fwd_dual") else None,
    ]

    for test_config in filter(None, test_configs):
        label = test_config.pop("label")
        config_copy = BenchmarkConfig(
            **{k: v for k, v in vars(config).items() if k != "seq_lengths"},
            seq_lengths=[512, 1024, 2048],  # Smaller subset for additional tests
        )
        for k, v in test_config.items():
            setattr(config_copy, k, v)

        print(f"\nTesting with {label}:")
        benchmark_test = AttentionBenchmark(config_copy)
        test_df = benchmark_test.run_benchmarks()
        print(f"  Average speedup vs baseline: {test_df.groupby('method')['time_ms'].mean()}")


if __name__ == "__main__":
    main()
