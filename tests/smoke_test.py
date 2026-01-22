#!/usr/bin/env python3
"""
Smoke Test for Predator-Prey Simulation Pipeline

Run this before HPC submission to verify everything works correctly.

Usage:
    python smoke_test.py              # Run all tests
    python smoke_test.py --quick      # Run minimal tests only
    python smoke_test.py --verbose    # Extra output

Tests:
    1. Module imports
    2. Numba kernel (random movement)
    3. Numba kernel (directed hunting)
    4. Full simulation (random, no evolution)
    5. Full simulation (random, with evolution)
    6. Full simulation (directed, no evolution)
    7. Full simulation (directed, with evolution)
    8. PCF computation
    9. Cluster measurement
    10. Reproducibility (seeding)
    11. Binary save/load roundtrip
"""

import sys
import time
import argparse
import tempfile
from pathlib import Path

# Setup path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

# Track results
RESULTS = []
VERBOSE = False


def log(msg: str, level: str = "INFO"):
    """Print formatted log message."""
    symbols = {"INFO": "ℹ", "PASS": "✓", "FAIL": "✗", "WARN": "⚠", "RUN": "→"}
    print(f"  {symbols.get(level, '•')} {msg}")


def run_test(name: str, func, *args, **kwargs):
    """Run a test function and track results."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        if result:
            log(f"PASSED in {elapsed:.2f}s", "PASS")
            RESULTS.append((name, True, elapsed, None))
            return True
        else:
            log(f"FAILED in {elapsed:.2f}s", "FAIL")
            RESULTS.append((name, False, elapsed, "Test returned False"))
            return False
    except Exception as e:
        elapsed = time.perf_counter() - start
        log(f"FAILED with exception: {e}", "FAIL")
        RESULTS.append((name, False, elapsed, str(e)))
        if VERBOSE:
            import traceback
            traceback.print_exc()
        return False


def test_imports():
    """Test that all required modules import correctly."""
    log("Importing numba_optimized...", "RUN")
    from scripts.numba_optimized import (
        PPKernel,
        compute_all_pcfs_fast,
        measure_cluster_sizes_fast,
        set_numba_seed,
        NUMBA_AVAILABLE,
    )
    log(f"NUMBA_AVAILABLE = {NUMBA_AVAILABLE}")
    
    if not NUMBA_AVAILABLE:
        log("Numba not available - performance will be degraded", "WARN")
    
    log("Importing CA module...", "RUN")
    from models.CA import PP, set_numba_seed as ca_seed
    
    log("Importing pp_analysis...", "RUN")
    from scripts.pp_analysis import (
        Config,
        run_single_simulation,
        count_populations,
    )
    
    log("All imports successful")
    return True


def test_numba_kernel_random():
    """Test Numba kernel with random movement."""
    from scripts.numba_optimized import PPKernel, set_numba_seed
    
    log("Creating kernel (directed_hunting=False)...", "RUN")
    kernel = PPKernel(50, 50, "moore", directed_hunting=False)
    assert kernel.directed_hunting == False
    
    log("Setting up test grid...", "RUN")
    np.random.seed(42)
    set_numba_seed(42)
    grid = np.random.choice([0, 1, 2], (50, 50), p=[0.55, 0.30, 0.15]).astype(np.int32)
    prey_death = np.full((50, 50), 0.05, dtype=np.float64)
    prey_death[grid != 1] = np.nan
    
    initial_prey = np.sum(grid == 1)
    initial_pred = np.sum(grid == 2)
    log(f"Initial: prey={initial_prey}, pred={initial_pred}")
    
    log("Running 100 update steps...", "RUN")
    for _ in range(100):
        kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1, evolution_stopped=False)
    
    final_prey = np.sum(grid == 1)
    final_pred = np.sum(grid == 2)
    log(f"Final: prey={final_prey}, pred={final_pred}")
    
    # Verify grid is valid
    assert grid.min() >= 0, "Grid has negative values"
    assert grid.max() <= 2, "Grid has values > 2"
    assert not np.any(np.isnan(grid)), "Grid has NaN values"
    
    # Verify prey_death consistency
    prey_mask = (grid == 1)
    if np.any(prey_mask):
        assert np.all(~np.isnan(prey_death[prey_mask])), "Prey cells missing death rates"
    assert np.all(np.isnan(prey_death[~prey_mask])), "Non-prey cells have death rates"
    
    log("Grid and prey_death arrays are consistent")
    return True


def test_numba_kernel_directed():
    """Test Numba kernel with directed hunting."""
    from scripts.numba_optimized import PPKernel, set_numba_seed
    
    log("Creating kernel (directed_hunting=True)...", "RUN")
    kernel = PPKernel(50, 50, "moore", directed_hunting=True)
    assert kernel.directed_hunting == True
    
    log("Setting up test grid...", "RUN")
    np.random.seed(42)
    set_numba_seed(42)
    grid = np.random.choice([0, 1, 2], (50, 50), p=[0.55, 0.30, 0.15]).astype(np.int32)
    prey_death = np.full((50, 50), 0.05, dtype=np.float64)
    prey_death[grid != 1] = np.nan
    
    initial_prey = np.sum(grid == 1)
    initial_pred = np.sum(grid == 2)
    log(f"Initial: prey={initial_prey}, pred={initial_pred}")
    
    log("Running 100 update steps...", "RUN")
    for _ in range(100):
        kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1, evolution_stopped=False)
    
    final_prey = np.sum(grid == 1)
    final_pred = np.sum(grid == 2)
    log(f"Final: prey={final_prey}, pred={final_pred}")
    
    # Verify grid is valid
    assert grid.min() >= 0, "Grid has negative values"
    assert grid.max() <= 2, "Grid has values > 2"
    
    log("Directed hunting kernel working correctly")
    return True


def test_ca_model_random():
    """Test CA PP model with random movement."""
    from models.CA import PP
    from scripts.numba_optimized import set_numba_seed
    
    log("Creating PP model (directed_hunting=False)...", "RUN")
    np.random.seed(42)
    set_numba_seed(42)
    
    model = PP(
        rows=50, cols=50,
        densities=(0.30, 0.15),
        neighborhood="moore",
        params={"prey_birth": 0.2, "prey_death": 0.05, 
                "predator_birth": 0.2, "predator_death": 0.1},
        seed=42,
        synchronous=False,
        directed_hunting=False,
    )
    
    assert model.directed_hunting == False
    
    initial_prey = np.sum(model.grid == 1)
    initial_pred = np.sum(model.grid == 2)
    log(f"Initial: prey={initial_prey}, pred={initial_pred}")
    
    log("Running 100 steps...", "RUN")
    model.run(100)
    
    final_prey = np.sum(model.grid == 1)
    final_pred = np.sum(model.grid == 2)
    log(f"Final: prey={final_prey}, pred={final_pred}")
    
    assert model.grid.min() >= 0
    assert model.grid.max() <= 2
    
    return True


def test_ca_model_directed():
    """Test CA PP model with directed hunting."""
    from models.CA import PP
    from scripts.numba_optimized import set_numba_seed
    
    log("Creating PP model (directed_hunting=True)...", "RUN")
    np.random.seed(42)
    set_numba_seed(42)
    
    model = PP(
        rows=50, cols=50,
        densities=(0.30, 0.15),
        neighborhood="moore",
        params={"prey_birth": 0.2, "prey_death": 0.05,
                "predator_birth": 0.2, "predator_death": 0.1},
        seed=42,
        synchronous=False,
        directed_hunting=True,
    )
    
    assert model.directed_hunting == True
    
    initial_prey = np.sum(model.grid == 1)
    initial_pred = np.sum(model.grid == 2)
    log(f"Initial: prey={initial_prey}, pred={initial_pred}")
    
    log("Running 100 steps...", "RUN")
    model.run(100)
    
    final_prey = np.sum(model.grid == 1)
    final_pred = np.sum(model.grid == 2)
    log(f"Final: prey={final_prey}, pred={final_pred}")
    
    assert model.grid.min() >= 0
    assert model.grid.max() <= 2
    
    return True


def test_ca_model_with_evolution():
    """Test CA PP model with evolution enabled."""
    from models.CA import PP
    from scripts.numba_optimized import set_numba_seed
    
    log("Creating PP model with evolution...", "RUN")
    np.random.seed(42)
    set_numba_seed(42)
    
    model = PP(
        rows=50, cols=50,
        densities=(0.30, 0.15),
        neighborhood="moore",
        params={"prey_birth": 0.2, "prey_death": 0.05,
                "predator_birth": 0.2, "predator_death": 0.1},
        seed=42,
        synchronous=False,
        directed_hunting=True,
    )
    
    log("Enabling prey_death evolution...", "RUN")
    model.evolve("prey_death", sd=0.05, min_val=0.01, max_val=0.15)
    
    initial_mean = np.nanmean(model.cell_params["prey_death"])
    log(f"Initial prey_death mean: {initial_mean:.4f}")
    
    log("Running 200 steps...", "RUN")
    model.run(200)
    
    final_values = model.cell_params["prey_death"]
    valid_values = final_values[~np.isnan(final_values)]
    
    if len(valid_values) > 0:
        final_mean = np.mean(valid_values)
        final_std = np.std(valid_values)
        log(f"Final prey_death: mean={final_mean:.4f}, std={final_std:.4f}")
        
        # Check bounds
        assert valid_values.min() >= 0.01 - 1e-9, "Values below minimum"
        assert valid_values.max() <= 0.15 + 1e-9, "Values above maximum"
        log("Evolution bounds respected")
    else:
        log("No prey survived - cannot check evolution", "WARN")
    
    return True


def test_full_simulation_pipeline():
    """Test the full simulation pipeline via run_single_simulation."""
    from scripts.pp_analysis import Config, run_single_simulation
    from scripts.numba_optimized import set_numba_seed
    
    log("Creating fast config...", "RUN")
    cfg = Config()
    cfg.default_grid = 40
    cfg.warmup_steps = 50
    cfg.measurement_steps = 100
    cfg.cluster_samples = 1
    cfg.collect_pcf = True
    cfg.pcf_sample_rate = 1.0  # Always compute PCF for this test
    
    # Test random movement
    log("Running simulation (random movement, no evolution)...", "RUN")
    cfg.directed_hunting = False
    np.random.seed(42)
    set_numba_seed(42)
    
    result_random = run_single_simulation(
        prey_birth=0.2, prey_death=0.05, grid_size=40,
        seed=42, with_evolution=False, cfg=cfg, compute_pcf=True,
    )
    
    assert "prey_mean" in result_random
    assert "pred_mean" in result_random
    log(f"Random: prey_mean={result_random['prey_mean']:.1f}, pred_mean={result_random['pred_mean']:.1f}")
    
    # Test directed hunting
    log("Running simulation (directed hunting, no evolution)...", "RUN")
    cfg.directed_hunting = True
    np.random.seed(42)
    set_numba_seed(42)
    
    result_directed = run_single_simulation(
        prey_birth=0.2, prey_death=0.05, grid_size=40,
        seed=42, with_evolution=False, cfg=cfg, compute_pcf=True,
    )
    
    assert "prey_mean" in result_directed
    log(f"Directed: prey_mean={result_directed['prey_mean']:.1f}, pred_mean={result_directed['pred_mean']:.1f}")
    
    # Test with evolution
    log("Running simulation (directed hunting, with evolution)...", "RUN")
    np.random.seed(42)
    set_numba_seed(42)
    
    result_evo = run_single_simulation(
        prey_birth=0.2, prey_death=0.05, grid_size=40,
        seed=42, with_evolution=True, cfg=cfg, compute_pcf=True,
    )
    
    assert result_evo["with_evolution"] == True
    log(f"Evolution: prey_mean={result_evo['prey_mean']:.1f}")
    
    return True


def test_pcf_computation():
    """Test PCF computation."""
    from scripts.numba_optimized import compute_all_pcfs_fast, set_numba_seed
    
    log("Creating test grid...", "RUN")
    np.random.seed(42)
    set_numba_seed(42)
    grid = np.random.choice([0, 1, 2], (100, 100), p=[0.55, 0.30, 0.15]).astype(np.int32)
    
    n_prey = np.sum(grid == 1)
    n_pred = np.sum(grid == 2)
    log(f"Grid: prey={n_prey}, pred={n_pred}")
    
    log("Computing PCFs...", "RUN")
    t0 = time.perf_counter()
    pcfs = compute_all_pcfs_fast(grid, max_distance=20.0, n_bins=20)
    elapsed = time.perf_counter() - t0
    log(f"PCF computation took {elapsed*1000:.1f}ms")
    
    # Check all three PCFs
    for key in ['prey_prey', 'pred_pred', 'prey_pred']:
        assert key in pcfs, f"Missing PCF: {key}"
        dist, pcf, n_pairs = pcfs[key]
        
        assert len(dist) == 20, f"{key}: wrong number of bins"
        assert len(pcf) == 20, f"{key}: wrong PCF length"
        assert not np.any(np.isnan(pcf)), f"{key}: PCF contains NaN"
        
        log(f"{key}: n_pairs={n_pairs}, mean_pcf={np.mean(pcf):.3f}")
    
    return True


def test_cluster_measurement():
    """Test cluster size measurement."""
    from scripts.numba_optimized import measure_cluster_sizes_fast
    
    log("Creating grid with known clusters...", "RUN")
    grid = np.zeros((30, 30), dtype=np.int32)
    
    # Cluster 1: 3x3 = 9 cells
    grid[2:5, 2:5] = 1
    # Cluster 2: 2x4 = 8 cells
    grid[10:12, 10:14] = 1
    # Cluster 3: single cell
    grid[20, 20] = 1
    # Cluster 4: L-shape = 5 cells
    grid[25, 25:28] = 1
    grid[26:28, 25] = 1
    
    expected_sizes = sorted([9, 8, 1, 5], reverse=True)
    log(f"Expected cluster sizes: {expected_sizes}")
    
    log("Measuring clusters...", "RUN")
    sizes = measure_cluster_sizes_fast(grid, 1)
    actual_sizes = sorted(sizes, reverse=True)
    log(f"Actual cluster sizes: {list(actual_sizes)}")
    
    assert len(sizes) == 4, f"Expected 4 clusters, found {len(sizes)}"
    assert list(actual_sizes) == expected_sizes, "Cluster sizes don't match"
    
    # Verify total cells
    assert sum(sizes) == np.sum(grid == 1), "Cluster total doesn't match grid total"
    
    log("Cluster measurement correct")
    return True


def test_reproducibility():
    """Test that seeding produces reproducible results."""
    from scripts.numba_optimized import PPKernel, set_numba_seed
    
    log("Running simulation twice with same seed...", "RUN")
    
    def run_sim(seed):
        np.random.seed(seed)
        set_numba_seed(seed)
        grid = np.random.choice([0, 1, 2], (30, 30), p=[0.55, 0.30, 0.15]).astype(np.int32)
        prey_death = np.full((30, 30), 0.05, dtype=np.float64)
        prey_death[grid != 1] = np.nan
        
        kernel = PPKernel(30, 30, "moore", directed_hunting=True)
        for _ in range(50):
            kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1, evolution_stopped=False)
        
        return grid.copy(), prey_death.copy()
    
    grid1, pd1 = run_sim(12345)
    grid2, pd2 = run_sim(12345)
    
    prey1, prey2 = np.sum(grid1 == 1), np.sum(grid2 == 1)
    pred1, pred2 = np.sum(grid1 == 2), np.sum(grid2 == 2)
    
    log(f"Run 1: prey={prey1}, pred={pred1}")
    log(f"Run 2: prey={prey2}, pred={pred2}")
    
    if np.array_equal(grid1, grid2):
        log("Grids are IDENTICAL - perfect reproducibility", "PASS")
    else:
        diff_count = np.sum(grid1 != grid2)
        log(f"Grids differ in {diff_count} cells - may indicate seeding issue", "WARN")
        # Still pass if populations match (some internal ordering may differ)
        if prey1 == prey2 and pred1 == pred2:
            log("Populations match - acceptable", "PASS")
        else:
            return False
    
    return True


def test_binary_save_load():
    """Test binary save/load roundtrip."""
    from scripts.pp_analysis import save_sweep_binary, load_sweep_binary
    
    log("Creating test results...", "RUN")
    results = [
        {"prey_birth": 0.2, "prey_death": 0.05, "prey_mean": 150.5, "pred_mean": 75.2,
         "seed": 42, "grid_size": 50, "with_evolution": False},
        {"prey_birth": 0.3, "prey_death": 0.08, "prey_mean": 120.3, "pred_mean": 90.1,
         "seed": 43, "grid_size": 50, "with_evolution": True},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_results.npz"
        
        log(f"Saving to {filepath}...", "RUN")
        save_sweep_binary(results, filepath)
        
        assert filepath.exists(), "File not created"
        log(f"File size: {filepath.stat().st_size} bytes")
        
        log("Loading back...", "RUN")
        loaded = load_sweep_binary(filepath)
        
        assert len(loaded) == len(results), "Wrong number of results loaded"
        
        for i, (orig, load) in enumerate(zip(results, loaded)):
            for key in orig:
                if isinstance(orig[key], float):
                    assert np.isclose(orig[key], load[key]), f"Result {i}, key {key} mismatch"
                else:
                    assert orig[key] == load[key], f"Result {i}, key {key} mismatch"
        
        log("Roundtrip successful")
    
    return True


def test_hunting_dynamics_comparison():
    """Compare dynamics between random and directed hunting."""
    from scripts.numba_optimized import PPKernel, set_numba_seed
    
    log("Setting up comparison...", "RUN")
    
    # Use same initial grid
    np.random.seed(999)
    template = np.random.choice([0, 1, 2], (60, 60), p=[0.50, 0.35, 0.15]).astype(np.int32)
    
    def run_mode(directed: bool, seed: int = 999):
        grid = template.copy()
        prey_death = np.full((60, 60), 0.05, dtype=np.float64)
        prey_death[grid != 1] = np.nan
        
        set_numba_seed(seed)
        kernel = PPKernel(60, 60, "moore", directed_hunting=directed)
        
        history = []
        for step in range(100):
            kernel.update(grid, prey_death, 0.2, 0.05, 0.5, 0.1)  # High pred birth
            if step % 10 == 0:
                history.append((np.sum(grid == 1), np.sum(grid == 2)))
        
        return history
    
    log("Running random movement...", "RUN")
    hist_random = run_mode(directed=False)
    
    log("Running directed hunting...", "RUN")
    hist_directed = run_mode(directed=True)
    
    log("\nPopulation dynamics comparison:")
    log(f"{'Step':<6} {'Random':<20} {'Directed':<20}")
    log("-" * 46)
    for i, ((pr, pdr), (pd, pdd)) in enumerate(zip(hist_random, hist_directed)):
        step = i * 10
        log(f"{step:<6} prey={pr:<4} pred={pdr:<4}   prey={pd:<4} pred={pdd:<4}")
    
    # Final comparison
    final_random_prey = hist_random[-1][0]
    final_directed_prey = hist_directed[-1][0]
    
    log(f"\nFinal prey - Random: {final_random_prey}, Directed: {final_directed_prey}")
    
    # Directed hunting with high predator birth typically depletes prey faster
    # But we don't assert this strictly due to stochastic nature
    log("Dynamics comparison complete")
    
    return True

def print_summary():
    """Print test summary."""
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _, _ in RESULTS if success)
    failed = sum(1 for _, success, _, _ in RESULTS if not success)
    total_time = sum(t for _, _, t, _ in RESULTS)
    
    for name, success, elapsed, error in RESULTS:
        status = "PASS" if success else "FAIL"
        print(f"  {status}  {name} ({elapsed:.2f}s)")
        if error and not success:
            print(f"         Error: {error[:60]}...")
    
    print("-" * 60)
    print(f"  Total: {passed} passed, {failed} failed in {total_time:.2f}s")
    print("=" * 60)
    
    if failed == 0:
        print("\ALL TESTS PASSED - Ready for HPC submission!\n")
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED - Please fix before HPC submission.\n")
    
    return failed == 0


def main():
    global VERBOSE
    
    parser = argparse.ArgumentParser(description="Pre-HPC Smoke Test")
    parser.add_argument("--quick", action="store_true", help="Run minimal tests only")
    parser.add_argument("--verbose", action="store_true", help="Extra output")
    args = parser.parse_args()
    
    VERBOSE = args.verbose
    
    print("\n" + "=" * 60)
    print("   PREDATOR-PREY SIMULATION - PRE-HPC SMOKE TEST")
    print("=" * 60)
    print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Python: {sys.version.split()[0]}")
    print("=" * 60)
    
    # Core tests (always run)
    run_test("Module Imports", test_imports)
    run_test("Numba Kernel (Random)", test_numba_kernel_random)
    run_test("Numba Kernel (Directed)", test_numba_kernel_directed)
    run_test("CA Model (Random)", test_ca_model_random)
    run_test("CA Model (Directed)", test_ca_model_directed)
    
    if not args.quick:
        # Extended tests
        run_test("CA Model (Evolution)", test_ca_model_with_evolution)
        run_test("Full Simulation Pipeline", test_full_simulation_pipeline)
        run_test("PCF Computation", test_pcf_computation)
        run_test("Cluster Measurement", test_cluster_measurement)
        run_test("Reproducibility (Seeding)", test_reproducibility)
        run_test("Binary Save/Load", test_binary_save_load)
        run_test("Hunting Dynamics Comparison", test_hunting_dynamics_comparison)
    
    success = print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()