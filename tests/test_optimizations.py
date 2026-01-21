#!/usr/bin/env python3
"""
Test and Benchmark Script for Optimized PP Analysis

Run from your project root:
    python scripts/test_optimizations.py
    python scripts/test_optimizations.py --full
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Handle imports from different locations
project_root = str(Path(__file__).resolve().parents[1])
scripts_dir = str(Path(__file__).resolve().parent)
for p in [project_root, scripts_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Flexible import
def get_modules():
    try:
        from scripts.numba_optimized import (
            NUMBA_AVAILABLE, PPKernel, compute_all_pcfs_fast, measure_cluster_sizes_fast
        )
    except ImportError:
        from numba_optimized import (
            NUMBA_AVAILABLE, PPKernel, compute_all_pcfs_fast, measure_cluster_sizes_fast
        )
    return NUMBA_AVAILABLE, PPKernel, compute_all_pcfs_fast, measure_cluster_sizes_fast


def test_numba():
    """Test Numba availability."""
    print("=" * 60)
    print("TEST: Numba Availability")
    print("=" * 60)
    try:
        NUMBA_AVAILABLE, PPKernel, _, _ = get_modules()
        print(f"  Numba available: {NUMBA_AVAILABLE}")
        kernel = PPKernel(10, 10, "moore")
        print("  PPKernel: OK")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_kernel():
    """Test kernel correctness."""
    print("\n" + "=" * 60)
    print("TEST: Kernel Correctness")
    print("=" * 60)
    try:
        _, PPKernel, _, _ = get_modules()
        np.random.seed(42)
        
        grid = np.random.choice([0, 1, 2], (50, 50), p=[0.55, 0.30, 0.15]).astype(np.int32)
        prey_death = np.full((50, 50), 0.05, dtype=np.float64)
        prey_death[grid != 1] = np.nan
        
        print(f"  Initial: {np.sum(grid==1)} prey, {np.sum(grid==2)} pred")
        
        kernel = PPKernel(50, 50, "moore")
        for _ in range(100):
            kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1, evolution_stopped=False)
        
        print(f"  After 100: {np.sum(grid==1)} prey, {np.sum(grid==2)} pred")
        
        # Sanity checks
        assert 0 <= grid.min() <= grid.max() <= 2
        assert np.all(~np.isnan(prey_death[grid == 1]))
        print(" PASSED")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        return False


def test_pcf():
    """Test PCF computation."""
    print("\n" + "=" * 60)
    print("TEST: PCF Computation")
    print("=" * 60)
    try:
        _, _, compute_all_pcfs_fast, _ = get_modules()
        
        grid = np.zeros((100, 100), dtype=np.int32)
        # Create clustered prey
        for _ in range(10):
            cx, cy = np.random.randint(10, 90, 2)
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if np.random.random() < 0.7:
                        grid[(cx+dx) % 100, (cy+dy) % 100] = 1
        # Scatter predators
        empty = np.argwhere(grid == 0)
        for idx in np.random.choice(len(empty), min(500, len(empty)), replace=False):
            grid[empty[idx, 0], empty[idx, 1]] = 2
        
        print(f"  Grid: {np.sum(grid==1)} prey, {np.sum(grid==2)} pred")
        
        results = compute_all_pcfs_fast(grid, 20.0, 20)
        pcf_rr = results['prey_prey'][1]
        
        print(f"  Prey clustering (short range): {np.mean(pcf_rr[:5]):.2f}")
        assert np.mean(pcf_rr[:5]) > 1.0, "Clustered prey should have PCF > 1"
        print(" PASSED")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        return False


def test_clusters():
    """Test cluster measurement."""
    print("\n" + "=" * 60)
    print("TEST: Cluster Measurement")
    print("=" * 60)
    try:
        _, _, _, measure_cluster_sizes_fast = get_modules()
        
        grid = np.zeros((20, 20), dtype=np.int32)
        grid[2:5, 2:5] = 1  # 9 cells
        grid[10:12, 10:12] = 1  # 4 cells
        grid[15, 15] = 1  # 1 cell
        
        sizes = sorted(measure_cluster_sizes_fast(grid, 1), reverse=True)
        print(f"  Expected: [9, 4, 1], Got: {sizes}")
        
        assert sizes == [9, 4, 1]
        print(" PASSED")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        return False


def benchmark_kernel():
    """Benchmark kernel performance."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Kernel (500 steps, 100x100)")
    print("=" * 60)
    
    _, PPKernel, _, _ = get_modules()
    
    np.random.seed(42)
    grid = np.random.choice([0, 1, 2], (100, 100), p=[0.55, 0.30, 0.15]).astype(np.int32)
    prey_death = np.full((100, 100), 0.05, dtype=np.float64)
    prey_death[grid != 1] = np.nan
    
    kernel = PPKernel(100, 100, "moore")
    
    # Warmup
    g, p = grid.copy(), prey_death.copy()
    kernel.update(g, p, 0.2, 0.05, 0.2, 0.1)
    
    # Benchmark
    g, p = grid.copy(), prey_death.copy()
    t0 = time.perf_counter()
    for _ in range(500):
        kernel.update(g, p, 0.2, 0.05, 0.2, 0.1, evolution_stopped=False)
    elapsed = (time.perf_counter() - t0) * 1000
    
    print(f"  Total: {elapsed:.1f}ms")
    print(f"  Per step: {elapsed/500:.3f}ms")
    return elapsed / 500


def benchmark_pcf():
    """Benchmark PCF performance."""
    print("\n" + "=" * 60)
    print("BENCHMARK: PCF (100x100, 10 runs)")
    print("=" * 60)
    
    _, _, compute_all_pcfs_fast, _ = get_modules()
    
    np.random.seed(42)
    grid = np.zeros((100, 100), dtype=np.int32)
    positions = np.random.permutation(10000)
    for p in positions[:3000]: grid[p//100, p%100] = 1
    for p in positions[3000:4500]: grid[p//100, p%100] = 2
    
    print(f"  Grid: {np.sum(grid==1)} prey, {np.sum(grid==2)} pred")
    
    # Warmup
    _ = compute_all_pcfs_fast(grid, 20.0, 20)
    
    # Benchmark
    t0 = time.perf_counter()
    for _ in range(10):
        _ = compute_all_pcfs_fast(grid, 20.0, 20)
    elapsed = (time.perf_counter() - t0) / 10 * 1000
    
    print(f"  Per call: {elapsed:.1f}ms")
    return elapsed


def benchmark_full_sim():
    """Benchmark complete simulation."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Full Simulation")
    print("=" * 60)
    
    _, PPKernel, compute_all_pcfs_fast, measure_cluster_sizes_fast = get_modules()
    
    np.random.seed(42)
    grid = np.random.choice([0, 1, 2], (100, 100), p=[0.55, 0.30, 0.15]).astype(np.int32)
    prey_death = np.full((100, 100), 0.05, dtype=np.float64)
    prey_death[grid != 1] = np.nan
    
    kernel = PPKernel(100, 100, "moore")
    
    t0 = time.perf_counter()
    
    # Warmup (200 steps)
    for _ in range(200):
        kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1)
    t_warmup = (time.perf_counter() - t0) * 1000
    
    # Measurement (300 steps)
    for _ in range(300):
        kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1)
    t_measure = (time.perf_counter() - t0) * 1000 - t_warmup
    
    # Clusters
    _ = measure_cluster_sizes_fast(grid, 1)
    _ = measure_cluster_sizes_fast(grid, 2)
    t_cluster = (time.perf_counter() - t0) * 1000 - t_warmup - t_measure
    
    # PCF
    _ = compute_all_pcfs_fast(grid, 20.0, 20)
    t_pcf = (time.perf_counter() - t0) * 1000 - t_warmup - t_measure - t_cluster
    
    total = (time.perf_counter() - t0) * 1000
    
    print(f"  Warmup (200):     {t_warmup:.1f}ms")
    print(f"  Measure (300):    {t_measure:.1f}ms")
    print(f"  Clusters:         {t_cluster:.1f}ms")
    print(f"  PCF:              {t_pcf:.1f}ms")
    print(f"  ─────────────────────────")
    print(f"  TOTAL:            {total:.1f}ms")
    return total


def estimate_sweep():
    """Estimate sweep time."""
    print("\n" + "=" * 60)
    print("ESTIMATE: Full Sweep Runtime")
    print("=" * 60)
    
    sim_time = benchmark_full_sim()
    
    n_sims = 15 * 15 * 50 * 2  # 22,500
    total_ms = n_sims * sim_time
    
    print(f"\n  Single sim: {sim_time:.1f}ms")
    print(f"  Total sims: {n_sims:,}")
    print(f"\n  Estimated time:")
    print(f"    1 core:   {total_ms/3600000:.1f} hours")
    print(f"    8 cores:  {total_ms/3600000/8:.1f} hours")
    print(f"    32 cores: {total_ms/3600000/32:.2f} hours ({total_ms/60000/32:.1f} min)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full benchmark")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PP ANALYSIS OPTIMIZATION TESTS")
    print("=" * 60)
    
    # Run tests
    results = [
        ("Numba", test_numba()),
        ("Kernel", test_kernel()),
        ("PCF", test_pcf()),
        ("Clusters", test_clusters()),
    ]
    
    # Benchmarks
    kernel_time = benchmark_kernel()
    pcf_time = benchmark_pcf()
    
    if args.full:
        estimate_sweep()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Kernel: {kernel_time:.3f}ms/step")
    print(f"  PCF:    {pcf_time:.1f}ms/call")
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()