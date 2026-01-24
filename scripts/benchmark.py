#!/usr/bin/env python3
"""
Benchmarking Script for Predator-Prey Simulation Optimizations

Measures and compares performance of:
1. Numba-optimized kernel vs pure Python baseline
2. Cell-list PCF vs brute-force PCF  
3. Grid size scaling behavior
4. Random vs directed hunting overhead
5. Full simulation pipeline

Usage:
    python benchmark.py                    # Run all benchmarks
    python benchmark.py --quick            # Quick benchmarks only
    python benchmark.py --plot             # Generate performance plots
    python benchmark.py --export results   # Export to CSV

Output:
    - Console summary with speedup factors
    - Optional: benchmark_results.csv
    - Optional: benchmark_plots.png
"""

import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Setup path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

# Try to import optimized modules
try:
    from models.numba_optimized import (
        PPKernel,
        compute_pcf_periodic_fast,
        compute_all_pcfs_fast,
        measure_cluster_sizes_fast,
        set_numba_seed,
        NUMBA_AVAILABLE,
    )
except ImportError:
    NUMBA_AVAILABLE = False

try:
    from scipy.ndimage import label
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Store benchmark results."""
    name: str
    grid_size: int
    time_ms: float
    iterations: int
    variant: str = ""
    extra: Dict = None
    
    @property
    def time_per_iter_ms(self) -> float:
        return self.time_ms / self.iterations if self.iterations > 0 else 0


def timeit(func, *args, n_runs: int = 5, warmup: int = 1, **kwargs) -> Tuple[float, float]:
    """Time a function with warmup and multiple runs. Returns (mean_ms, std_ms)."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    
    return np.mean(times), np.std(times)


def create_test_grid(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create test grid and prey_death array."""
    np.random.seed(seed)
    grid = np.random.choice([0, 1, 2], (size, size), p=[0.55, 0.30, 0.15]).astype(np.int32)
    prey_death = np.full((size, size), 0.05, dtype=np.float64)
    prey_death[grid != 1] = np.nan
    return grid, prey_death


def print_header(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def print_result(name: str, time_ms: float, std_ms: float = 0, speedup: float = None, 
                 baseline_name: str = None):
    """Print formatted benchmark result."""
    if speedup and baseline_name:
        print(f"  {name:<35} {time_ms:>8.2f} ± {std_ms:>5.2f} ms  "
              f"({speedup:>5.1f}x vs {baseline_name})")
    else:
        print(f"  {name:<35} {time_ms:>8.2f} ± {std_ms:>5.2f} ms")


# =============================================================================
# BASELINE IMPLEMENTATIONS (for comparison)
# =============================================================================

def pcf_brute_force(positions_i: np.ndarray, positions_j: np.ndarray,
                    grid_shape: Tuple[int, int], max_distance: float,
                    n_bins: int = 50, self_correlation: bool = False):
    """
    Brute-force O(N*M) PCF computation for baseline comparison.
    """
    rows, cols = grid_shape
    L_row, L_col = float(rows), float(cols)
    
    bin_width = max_distance / n_bins
    bin_centers = np.linspace(bin_width/2, max_distance - bin_width/2, n_bins)
    hist = np.zeros(n_bins, dtype=np.int64)
    
    n_i, n_j = len(positions_i), len(positions_j)
    if n_i == 0 or n_j == 0:
        return bin_centers, np.ones(n_bins), 0
    
    # Brute force: check all pairs
    for i in range(n_i):
        r1, c1 = positions_i[i]
        start_j = i + 1 if self_correlation else 0
        
        for j in range(start_j, n_j):
            r2, c2 = positions_j[j]
            
            # Periodic distance
            dr = abs(r1 - r2)
            dc = abs(c1 - c2)
            if dr > L_row / 2:
                dr = L_row - dr
            if dc > L_col / 2:
                dc = L_col - dc
            
            d = np.sqrt(dr*dr + dc*dc)
            
            if 0 < d < max_distance:
                bin_idx = int(d / bin_width)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                hist[bin_idx] += 1
    
    if self_correlation:
        hist *= 2
    
    # Normalization
    area = L_row * L_col
    if self_correlation:
        density_product = n_i * (n_i - 1) / (area * area)
    else:
        density_product = n_i * n_j / (area * area)
    
    pcf = np.ones(n_bins)
    for i in range(n_bins):
        r = bin_centers[i]
        expected = density_product * 2 * np.pi * r * bin_width * area
        if expected > 1:
            pcf[i] = hist[i] / expected
    
    return bin_centers, pcf, int(np.sum(hist))


def cluster_scipy(grid: np.ndarray, species: int) -> np.ndarray:
    """Scipy-based cluster measurement for baseline comparison."""
    mask = (grid == species).astype(int)
    labeled, n_clusters = label(mask)
    
    sizes = []
    for i in range(1, n_clusters + 1):
        sizes.append(np.sum(labeled == i))
    
    return np.array(sizes, dtype=np.int32)


def pp_kernel_python(grid: np.ndarray, prey_death: np.ndarray,
                     prey_birth: float, prey_death_rate: float,
                     pred_birth: float, pred_death: float,
                     n_steps: int = 1) -> np.ndarray:
    """
    Pure Python PP kernel for baseline comparison.
    Simplified version without evolution.
    """
    rows, cols = grid.shape
    dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    n_shifts = 8
    
    for _ in range(n_steps):
        # Get occupied cells
        occupied = np.argwhere(grid != 0)
        np.random.shuffle(occupied)
        
        for idx in range(len(occupied)):
            r, c = occupied[idx]
            state = grid[r, c]
            
            if state == 0:
                continue
            
            # Pick random neighbor
            nbi = np.random.randint(0, n_shifts)
            nr = (r + dr[nbi]) % rows
            nc = (c + dc[nbi]) % cols
            
            if state == 1:  # Prey
                if np.random.random() < prey_death_rate:
                    grid[r, c] = 0
                elif grid[nr, nc] == 0:
                    if np.random.random() < prey_birth:
                        grid[nr, nc] = 1
            
            elif state == 2:  # Predator
                if np.random.random() < pred_death:
                    grid[r, c] = 0
                elif grid[nr, nc] == 1:
                    if np.random.random() < pred_birth:
                        grid[nr, nc] = 2
    
    return grid


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def benchmark_kernel(grid_sizes: List[int] = [50, 100, 150], 
                     n_steps: int = 100,
                     include_baseline: bool = True) -> List[BenchmarkResult]:
    """Benchmark PP kernel performance."""
    print_header("PP KERNEL BENCHMARK")
    
    results = []
    
    for size in grid_sizes:
        print(f"\n  Grid size: {size}x{size}")
        print("-" * 50)
        
        # Numba kernel (random)
        if NUMBA_AVAILABLE:
            grid, prey_death = create_test_grid(size)
            set_numba_seed(42)
            kernel = PPKernel(size, size, "moore", directed_hunting=False)
            
            def run_numba_random():
                g, pd = create_test_grid(size)
                for _ in range(n_steps):
                    kernel.update(g, pd, 0.2, 0.05, 0.2, 0.1)
            
            mean_ms, std_ms = timeit(run_numba_random, n_runs=5, warmup=1)
            print_result(f"Numba (random)", mean_ms, std_ms)
            results.append(BenchmarkResult("kernel_numba_random", size, mean_ms, n_steps, "numba"))
            numba_time = mean_ms
        
        # Numba kernel (directed)
        if NUMBA_AVAILABLE:
            kernel_dir = PPKernel(size, size, "moore", directed_hunting=True)
            
            def run_numba_directed():
                g, pd = create_test_grid(size)
                set_numba_seed(42)
                for _ in range(n_steps):
                    kernel_dir.update(g, pd, 0.2, 0.05, 0.2, 0.1)
            
            mean_ms, std_ms = timeit(run_numba_directed, n_runs=5, warmup=1)
            overhead = (mean_ms / numba_time - 1) * 100 if numba_time > 0 else 0
            print_result(f"Numba (directed)", mean_ms, std_ms)
            print(f"    → Directed hunting overhead: {overhead:+.1f}%")
            results.append(BenchmarkResult("kernel_numba_directed", size, mean_ms, n_steps, "numba"))
        
        # Python baseline (only for smaller grids)
        if include_baseline and size <= 50:
            def run_python():
                g, pd = create_test_grid(size)
                pp_kernel_python(g, pd, 0.2, 0.05, 0.2, 0.1, n_steps=n_steps)
            
            mean_ms, std_ms = timeit(run_python, n_runs=3, warmup=1)
            speedup = mean_ms / numba_time if numba_time > 0 else 0
            print_result(f"Python baseline", mean_ms, std_ms, speedup, "Numba")
            results.append(BenchmarkResult("kernel_python", size, mean_ms, n_steps, "python"))
    
    return results


def benchmark_pcf(grid_sizes: List[int] = [50, 100, 150],
                  include_baseline: bool = True) -> List[BenchmarkResult]:
    """Benchmark PCF computation."""
    print_header("PCF COMPUTATION BENCHMARK")
    
    results = []
    
    for size in grid_sizes:
        grid, _ = create_test_grid(size)
        prey_pos = np.argwhere(grid == 1)
        n_prey = len(prey_pos)
        
        print(f"\n  Grid: {size}x{size}, Prey: {n_prey}")
        print("-" * 50)
        
        # Cell-list (optimized)
        if NUMBA_AVAILABLE:
            set_numba_seed(42)
            
            def run_celllist():
                compute_all_pcfs_fast(grid, max_distance=20.0, n_bins=20)
            
            mean_ms, std_ms = timeit(run_celllist, n_runs=5, warmup=1)
            print_result("Cell-list PCF (Numba)", mean_ms, std_ms)
            results.append(BenchmarkResult("pcf_celllist", size, mean_ms, 1, "numba",
                                          {"n_prey": n_prey}))
            celllist_time = mean_ms
        
        # Brute force baseline (only for smaller grids)
        if include_baseline and size <= 75:
            def run_bruteforce():
                pcf_brute_force(prey_pos, prey_pos, (size, size), 20.0, 20, True)
            
            mean_ms, std_ms = timeit(run_bruteforce, n_runs=3, warmup=1)
            speedup = mean_ms / celllist_time if celllist_time > 0 else 0
            print_result("Brute-force PCF (Python)", mean_ms, std_ms, speedup, "Cell-list")
            results.append(BenchmarkResult("pcf_bruteforce", size, mean_ms, 1, "python",
                                          {"n_prey": n_prey}))
    
    return results


def benchmark_clusters(grid_sizes: List[int] = [50, 100, 200],
                       include_baseline: bool = True) -> List[BenchmarkResult]:
    """Benchmark cluster measurement."""
    print_header("CLUSTER MEASUREMENT BENCHMARK")
    
    results = []
    
    for size in grid_sizes:
        grid, _ = create_test_grid(size)
        n_prey = np.sum(grid == 1)
        
        print(f"\n  Grid: {size}x{size}, Prey: {n_prey}")
        print("-" * 50)
        
        # Numba flood-fill
        if NUMBA_AVAILABLE:
            def run_numba():
                measure_cluster_sizes_fast(grid, 1)
            
            mean_ms, std_ms = timeit(run_numba, n_runs=10, warmup=2)
            print_result("Numba flood-fill", mean_ms, std_ms)
            results.append(BenchmarkResult("cluster_numba", size, mean_ms, 1, "numba"))
            numba_time = mean_ms
        
        # Scipy baseline
        if include_baseline and SCIPY_AVAILABLE:
            def run_scipy():
                cluster_scipy(grid, 1)
            
            mean_ms, std_ms = timeit(run_scipy, n_runs=10, warmup=2)
            speedup = mean_ms / numba_time if numba_time > 0 else 0
            print_result("Scipy label", mean_ms, std_ms, speedup, "Numba")
            results.append(BenchmarkResult("cluster_scipy", size, mean_ms, 1, "scipy"))
    
    return results


def benchmark_full_simulation(grid_sizes: List[int] = [50, 100],
                              n_steps: int = 200) -> List[BenchmarkResult]:
    """Benchmark full simulation with all components."""
    print_header("FULL SIMULATION BENCHMARK")
    
    results = []
    
    for size in grid_sizes:
        print(f"\n  Grid: {size}x{size}, Steps: {n_steps}")
        print("-" * 50)
        
        if NUMBA_AVAILABLE:
            set_numba_seed(42)
            
            def run_full_sim(directed: bool):
                np.random.seed(42)
                set_numba_seed(42)
                grid, prey_death = create_test_grid(size)
                kernel = PPKernel(size, size, "moore", directed_hunting=directed)
                
                for step in range(n_steps):
                    kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1,
                                 evolution_stopped=False)
                    
                    # Compute PCF every 50 steps (realistic scenario)
                    if step % 50 == 49:
                        compute_all_pcfs_fast(grid, max_distance=20.0, n_bins=20)
                        measure_cluster_sizes_fast(grid, 1)
            
            # Random movement
            mean_ms, std_ms = timeit(lambda: run_full_sim(False), n_runs=3, warmup=1)
            print_result("Full sim (random)", mean_ms, std_ms)
            results.append(BenchmarkResult("full_random", size, mean_ms, n_steps, "numba"))
            
            # Directed hunting
            mean_ms_dir, std_ms_dir = timeit(lambda: run_full_sim(True), n_runs=3, warmup=1)
            print_result("Full sim (directed)", mean_ms_dir, std_ms_dir)
            results.append(BenchmarkResult("full_directed", size, mean_ms_dir, n_steps, "numba"))
            
            # Calculate throughput
            steps_per_sec = n_steps / (mean_ms / 1000)
            print(f"    → Throughput: {steps_per_sec:.0f} steps/sec")
    
    return results


def benchmark_scaling(max_size: int = 200, n_points: int = 6) -> List[BenchmarkResult]:
    """Benchmark scaling behavior with grid size."""
    print_header("SCALING ANALYSIS")
    
    sizes = np.linspace(30, max_size, n_points).astype(int)
    results = []
    
    print(f"\n  {'Size':<8} {'Kernel (ms)':<15} {'PCF (ms)':<15} {'Total (ms)':<15}")
    print("-" * 55)
    
    for size in sizes:
        if NUMBA_AVAILABLE:
            grid, prey_death = create_test_grid(size)
            set_numba_seed(42)
            kernel = PPKernel(size, size, "moore", directed_hunting=False)
            
            # Kernel benchmark
            def run_kernel():
                g, pd = create_test_grid(size)
                for _ in range(50):
                    kernel.update(g, pd, 0.2, 0.05, 0.2, 0.1)
            
            kernel_ms, _ = timeit(run_kernel, n_runs=3, warmup=1)
            
            # PCF benchmark
            def run_pcf():
                compute_all_pcfs_fast(grid, max_distance=20.0, n_bins=20)
            
            pcf_ms, _ = timeit(run_pcf, n_runs=3, warmup=1)
            
            total_ms = kernel_ms + pcf_ms
            
            print(f"  {size:<8} {kernel_ms:<15.2f} {pcf_ms:<15.2f} {total_ms:<15.2f}")
            
            results.append(BenchmarkResult("scaling_kernel", size, kernel_ms, 50, "numba"))
            results.append(BenchmarkResult("scaling_pcf", size, pcf_ms, 1, "numba"))
    
    return results


# =============================================================================
# SUMMARY AND EXPORT
# =============================================================================

def print_summary(all_results: List[BenchmarkResult]):
    """Print benchmark summary with key findings."""
    print_header("BENCHMARK SUMMARY")
    
    # Extract key speedups
    print("\n  KEY FINDINGS:")
    print("-" * 50)
    
    # Kernel speedup (Python vs Numba at 50x50)
    kernel_python = [r for r in all_results if r.name == "kernel_python" and r.grid_size == 50]
    kernel_numba = [r for r in all_results if r.name == "kernel_numba_random" and r.grid_size == 50]
    
    if kernel_python and kernel_numba:
        speedup = kernel_python[0].time_ms / kernel_numba[0].time_ms
        print(f"  • Numba kernel speedup:     {speedup:>6.1f}x (vs Python)")
    
    # PCF speedup
    pcf_brute = [r for r in all_results if r.name == "pcf_bruteforce" and r.grid_size == 50]
    pcf_cell = [r for r in all_results if r.name == "pcf_celllist" and r.grid_size == 50]
    
    if pcf_brute and pcf_cell:
        speedup = pcf_brute[0].time_ms / pcf_cell[0].time_ms
        print(f"  • Cell-list PCF speedup:    {speedup:>6.1f}x (vs brute-force)")
    
    # Cluster speedup
    cluster_scipy = [r for r in all_results if r.name == "cluster_scipy" and r.grid_size == 100]
    cluster_numba = [r for r in all_results if r.name == "cluster_numba" and r.grid_size == 100]
    
    if cluster_scipy and cluster_numba:
        speedup = cluster_scipy[0].time_ms / cluster_numba[0].time_ms
        print(f"  • Numba cluster speedup:    {speedup:>6.1f}x (vs scipy)")
    
    # Directed hunting overhead
    kernel_random = [r for r in all_results if r.name == "kernel_numba_random" and r.grid_size == 100]
    kernel_directed = [r for r in all_results if r.name == "kernel_numba_directed" and r.grid_size == 100]
    
    if kernel_random and kernel_directed:
        overhead = (kernel_directed[0].time_ms / kernel_random[0].time_ms - 1) * 100
        print(f"  • Directed hunting overhead: {overhead:>+5.1f}%")
    
    # Throughput
    full_results = [r for r in all_results if r.name == "full_random" and r.grid_size == 100]
    if full_results:
        steps_per_sec = full_results[0].iterations / (full_results[0].time_ms / 1000)
        print(f"  • Simulation throughput:    {steps_per_sec:>6.0f} steps/sec (100x100)")
    
    print("\n" + "=" * 70)


def export_results(results: List[BenchmarkResult], filepath: str):
    """Export results to CSV."""
    with open(filepath, 'w') as f:
        f.write("name,grid_size,time_ms,iterations,variant,time_per_iter_ms\n")
        for r in results:
            f.write(f"{r.name},{r.grid_size},{r.time_ms:.3f},{r.iterations},"
                   f"{r.variant},{r.time_per_iter_ms:.3f}\n")
    print(f"\n  Results exported to: {filepath}")


def generate_plots(results: List[BenchmarkResult], filepath: str):
    """Generate performance plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available - skipping plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Kernel performance by grid size
    ax = axes[0, 0]
    for variant in ["kernel_numba_random", "kernel_numba_directed"]:
        data = [(r.grid_size, r.time_ms) for r in results if r.name == variant]
        if data:
            sizes, times = zip(*sorted(data))
            label = "Random" if "random" in variant else "Directed"
            ax.plot(sizes, times, 'o-', label=label, linewidth=2, markersize=8)
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Kernel Performance (100 steps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: PCF comparison
    ax = axes[0, 1]
    for variant, label, color in [("pcf_celllist", "Cell-list", "green"),
                                   ("pcf_bruteforce", "Brute-force", "red")]:
        data = [(r.grid_size, r.time_ms) for r in results if r.name == variant]
        if data:
            sizes, times = zip(*sorted(data))
            ax.plot(sizes, times, 'o-', label=label, color=color, linewidth=2, markersize=8)
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("PCF Computation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Scaling analysis
    ax = axes[1, 0]
    kernel_data = [(r.grid_size, r.time_ms) for r in results if r.name == "scaling_kernel"]
    pcf_data = [(r.grid_size, r.time_ms) for r in results if r.name == "scaling_pcf"]
    
    if kernel_data:
        sizes, times = zip(*sorted(kernel_data))
        ax.plot(sizes, times, 'o-', label="Kernel (50 steps)", linewidth=2, markersize=8)
    if pcf_data:
        sizes, times = zip(*sorted(pcf_data))
        ax.plot(sizes, times, 's-', label="PCF", linewidth=2, markersize=8)
    
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Scaling Behavior")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Speedup summary (bar chart)
    ax = axes[1, 1]
    speedups = []
    labels = []
    
    # Calculate speedups
    kernel_py = next((r.time_ms for r in results if r.name == "kernel_python"), None)
    kernel_nb = next((r.time_ms for r in results if r.name == "kernel_numba_random" and r.grid_size == 50), None)
    if kernel_py and kernel_nb:
        speedups.append(kernel_py / kernel_nb)
        labels.append("Kernel\n(Numba)")
    
    pcf_bf = next((r.time_ms for r in results if r.name == "pcf_bruteforce" and r.grid_size == 50), None)
    pcf_cl = next((r.time_ms for r in results if r.name == "pcf_celllist" and r.grid_size == 50), None)
    if pcf_bf and pcf_cl:
        speedups.append(pcf_bf / pcf_cl)
        labels.append("PCF\n(Cell-list)")
    
    if speedups:
        colors = plt.cm.viridis(np.linspace(0.3, 0.7, len(speedups)))
        bars = ax.bar(labels, speedups, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel("Speedup Factor (x)")
        ax.set_title("Optimization Speedups")
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, val in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plots saved to: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark Optimization Techniques")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    parser.add_argument("--plot", action="store_true", help="Generate performance plots")
    parser.add_argument("--export", type=str, help="Export results to CSV file")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline comparisons")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("   PREDATOR-PREY SIMULATION - OPTIMIZATION BENCHMARKS")
    print("=" * 70)
    print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Numba: {'Available' if NUMBA_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"   Scipy: {'Available' if SCIPY_AVAILABLE else 'NOT AVAILABLE'}")
    print("=" * 70)
    
    if not NUMBA_AVAILABLE:
        print("\n  ERROR: Numba not available. Cannot run benchmarks.")
        sys.exit(1)
    
    all_results = []
    include_baseline = not args.no_baseline
    
    # Run benchmarks
    if args.quick:
        grid_sizes = [50, 100]
        all_results.extend(benchmark_kernel(grid_sizes, n_steps=50, include_baseline=include_baseline))
        all_results.extend(benchmark_pcf([50], include_baseline=include_baseline))
    else:
        all_results.extend(benchmark_kernel([50, 100, 150], n_steps=100, include_baseline=include_baseline))
        all_results.extend(benchmark_pcf([50, 75, 100], include_baseline=include_baseline))
        all_results.extend(benchmark_clusters([50, 100, 150], include_baseline=include_baseline))
        all_results.extend(benchmark_full_simulation([50, 100], n_steps=200))
        all_results.extend(benchmark_scaling(max_size=200, n_points=6))
    
    # Summary
    print_summary(all_results)
    
    # Export
    if args.export:
        export_results(all_results, args.export)
    
    # Plots
    if args.plot:
        generate_plots(all_results, "benchmark_plots.png")
    
    print("\n  Benchmarking complete!\n")


if __name__ == "__main__":
    main()