#!/usr/bin/env python3
"""
Unit Tests for numba_optimized.py

Run with:
    pytest test_numba_optimized.py -v
    pytest test_numba_optimized.py -v --tb=short  # shorter traceback
    python test_numba_optimized.py  # without pytest
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Setup path
project_root = str(Path(__file__).resolve().parents[1])
scripts_dir = str(Path(__file__).resolve().parent)
for p in [project_root, scripts_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import module under test
try:
    from scripts.numba_optimized import (
        NUMBA_AVAILABLE,
        PPKernel,
        compute_pcf_periodic_fast,
        compute_all_pcfs_fast,
        measure_cluster_sizes_fast,
        warmup_numba_kernels,
        set_numba_seed
    )
except ImportError:
    from scripts.numba_optimized import (
        NUMBA_AVAILABLE,
        PPKernel,
        compute_pcf_periodic_fast,
        compute_all_pcfs_fast,
        measure_cluster_sizes_fast,
        set_numba_seed,
        warmup_numba_kernels,
    )


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def small_grid():
    """Small 20x20 grid for quick tests."""
    np.random.seed(42)
    grid = np.random.choice([0, 1, 2], size=(20, 20), p=[0.5, 0.3, 0.2]).astype(np.int32)
    return grid


@pytest.fixture
def medium_grid():
    """Medium 50x50 grid for correctness tests."""
    np.random.seed(42)
    grid = np.random.choice([0, 1, 2], size=(50, 50), p=[0.55, 0.30, 0.15]).astype(np.int32)
    return grid


@pytest.fixture
def large_grid():
    """Large 100x100 grid for performance tests."""
    np.random.seed(42)
    grid = np.random.choice([0, 1, 2], size=(100, 100), p=[0.55, 0.30, 0.15]).astype(np.int32)
    return grid


@pytest.fixture
def clustered_grid():
    """Grid with known clusters for testing cluster detection."""
    grid = np.zeros((30, 30), dtype=np.int32)
    # Cluster 1: 3x3 = 9 cells at (2,2)
    grid[2:5, 2:5] = 1
    # Cluster 2: 2x4 = 8 cells at (10,10)
    grid[10:12, 10:14] = 1
    # Cluster 3: single cell at (20,20)
    grid[20, 20] = 1
    # Cluster 4: L-shape = 5 cells
    grid[25, 25:28] = 1  # 3 horizontal
    grid[26:28, 25] = 1  # 2 vertical
    return grid


@pytest.fixture
def prey_death_array(medium_grid):
    """Prey death rate array matching medium_grid."""
    arr = np.full(medium_grid.shape, np.nan, dtype=np.float64)
    arr[medium_grid == 1] = 0.05
    return arr


# ============================================================================
# TEST: NUMBA AVAILABILITY
# ============================================================================

class TestNumbaAvailability:
    """Tests for Numba availability and basic imports."""
    
    def test_numba_available(self):
        """Numba should be available."""
        assert NUMBA_AVAILABLE, "Numba is not available - install with: pip install numba"
    
    def test_ppkernel_importable(self):
        """PPKernel class should be importable."""
        assert PPKernel is not None
    
    def test_pcf_functions_importable(self):
        """PCF functions should be importable."""
        assert compute_pcf_periodic_fast is not None
        assert compute_all_pcfs_fast is not None
    
    def test_cluster_function_importable(self):
        """Cluster measurement function should be importable."""
        assert measure_cluster_sizes_fast is not None


# ============================================================================
# TEST: PPKernel
# ============================================================================

class TestPPKernel:
    """Tests for the PPKernel class."""
    
    def test_kernel_initialization_moore(self):
        """Kernel should initialize with Moore neighborhood."""
        kernel = PPKernel(50, 50, "moore")
        assert kernel.rows == 50
        assert kernel.cols == 50
        assert len(kernel._dr) == 8  # Moore has 8 neighbors
    
    def test_kernel_initialization_neumann(self):
        """Kernel should initialize with von Neumann neighborhood."""
        kernel = PPKernel(50, 50, "neumann")
        assert len(kernel._dr) == 4  # von Neumann has 4 neighbors
    
    def test_kernel_buffer_allocation(self):
        """Kernel should pre-allocate work buffer."""
        kernel = PPKernel(100, 100, "moore")
        assert kernel._occupied_buffer.shape == (10000, 2)
        assert kernel._occupied_buffer.dtype == np.int32
    
    def test_kernel_update_preserves_grid_shape(self, medium_grid, prey_death_array):
        """Update should not change grid shape."""
        kernel = PPKernel(50, 50, "moore")
        original_shape = medium_grid.shape
        
        kernel.update(medium_grid, prey_death_array, 0.2, 0.05, 0.2, 0.1)
        
        assert medium_grid.shape == original_shape
    
    def test_kernel_update_valid_states(self, medium_grid, prey_death_array):
        """Grid should only contain valid states (0, 1, 2) after update."""
        kernel = PPKernel(50, 50, "moore")
        
        for _ in range(10):
            kernel.update(medium_grid, prey_death_array, 0.2, 0.05, 0.2, 0.1)
        
        assert medium_grid.min() >= 0
        assert medium_grid.max() <= 2
    
    def test_kernel_update_no_nan_in_grid(self, medium_grid, prey_death_array):
        """Grid should not contain NaN values."""
        kernel = PPKernel(50, 50, "moore")
        
        for _ in range(10):
            kernel.update(medium_grid, prey_death_array, 0.2, 0.05, 0.2, 0.1)
        
        assert not np.any(np.isnan(medium_grid))
    
    def test_kernel_prey_death_consistency(self, medium_grid, prey_death_array):
        """Prey death array should have values only where prey exist."""
        kernel = PPKernel(50, 50, "moore")
        
        for _ in range(10):
            kernel.update(medium_grid, prey_death_array, 0.2, 0.05, 0.2, 0.1,
                         evolution_stopped=False)
        
        prey_mask = (medium_grid == 1)
        non_prey_mask = (medium_grid != 1)
        
        # Prey cells should have non-NaN death rates
        assert np.all(~np.isnan(prey_death_array[prey_mask])), "Prey cells missing death rates"
        # Non-prey cells should have NaN death rates
        assert np.all(np.isnan(prey_death_array[non_prey_mask])), "Non-prey cells have death rates"
    
    def test_kernel_evolution_changes_values(self, medium_grid, prey_death_array):
        """Evolution should change prey death values over time."""
        kernel = PPKernel(50, 50, "moore")
        
        initial_mean = np.nanmean(prey_death_array)
        
        for _ in range(50):
            kernel.update(medium_grid, prey_death_array, 0.2, 0.05, 0.2, 0.1,
                         evolve_sd=0.1, evolve_min=0.001, evolve_max=0.2,
                         evolution_stopped=False)
        
        # Values should have changed (with high probability)
        final_values = prey_death_array[~np.isnan(prey_death_array)]
        if len(final_values) > 0:
            # Check that not all values are exactly 0.05
            assert not np.allclose(final_values, 0.05), "Evolution did not change values"
    
    def test_kernel_evolution_respects_bounds(self, medium_grid, prey_death_array):
        """Evolved values should stay within bounds."""
        kernel = PPKernel(50, 50, "moore")
        evolve_min, evolve_max = 0.01, 0.15
        
        for _ in range(100):
            kernel.update(medium_grid, prey_death_array, 0.2, 0.05, 0.2, 0.1,
                         evolve_sd=0.1, evolve_min=evolve_min, evolve_max=evolve_max,
                         evolution_stopped=False)
        
        valid_values = prey_death_array[~np.isnan(prey_death_array)]
        if len(valid_values) > 0:
            assert valid_values.min() >= evolve_min - 1e-10
            assert valid_values.max() <= evolve_max + 1e-10
    
    def test_kernel_evolution_stopped(self, medium_grid, prey_death_array):
        """When evolution stopped, values should only change by inheritance."""
        kernel = PPKernel(50, 50, "moore")
        
        # Set all prey to same value
        prey_death_array[medium_grid == 1] = 0.05
        
        for _ in range(20):
            kernel.update(medium_grid, prey_death_array, 0.2, 0.05, 0.2, 0.1,
                         evolve_sd=0.1, evolve_min=0.001, evolve_max=0.2,
                         evolution_stopped=True)
        
        # All values should still be exactly 0.05 (inherited without mutation)
        valid_values = prey_death_array[~np.isnan(prey_death_array)]
        if len(valid_values) > 0:
            assert np.allclose(valid_values, 0.05), "Evolution should be stopped"
    
    def test_kernel_deterministic_with_seed(self):
        """Same seed should produce same results."""
        results = []
        
        for _ in range(2):
            np.random.seed(12345)
            set_numba_seed(12345)
            grid = np.random.choice([0, 1, 2], (30, 30), p=[0.5, 0.3, 0.2]).astype(np.int32)
            prey_death = np.full((30, 30), 0.05, dtype=np.float64)
            prey_death[grid != 1] = np.nan
            
            kernel = PPKernel(30, 30, "moore")
            for _ in range(10):
                kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1)
            
            results.append(grid.copy())
        
        assert np.array_equal(results[0], results[1]), "Results should be deterministic"


# ============================================================================
# TEST: PCF COMPUTATION
# ============================================================================

class TestPCFComputation:
    """Tests for pair correlation function computation."""
    
    def test_pcf_returns_correct_shapes(self, medium_grid):
        """PCF should return arrays of correct shapes."""
        prey_pos = np.argwhere(medium_grid == 1)
        pred_pos = np.argwhere(medium_grid == 2)
        
        n_bins = 20
        dist, pcf, n_pairs = compute_pcf_periodic_fast(
            prey_pos, pred_pos, medium_grid.shape, 15.0, n_bins, False
        )
        
        assert len(dist) == n_bins
        assert len(pcf) == n_bins
        assert isinstance(n_pairs, int)
    
    def test_pcf_empty_positions(self):
        """PCF should handle empty position arrays."""
        empty = np.array([]).reshape(0, 2)
        positions = np.array([[5, 5], [10, 10]])
        
        dist, pcf, n_pairs = compute_pcf_periodic_fast(
            empty, positions, (50, 50), 15.0, 20, False
        )
        
        assert len(pcf) == 20
        assert np.allclose(pcf, 1.0)  # Default value for empty
        assert n_pairs == 0
    
    def test_pcf_values_reasonable(self, large_grid):
        """PCF values should be positive and reasonable."""
        prey_pos = np.argwhere(large_grid == 1)
        
        dist, pcf, n_pairs = compute_pcf_periodic_fast(
            prey_pos, prey_pos, large_grid.shape, 20.0, 20, True
        )
        
        assert np.all(pcf >= 0), "PCF should be non-negative"
        assert np.all(np.isfinite(pcf)), "PCF should be finite"
        # For random distribution, PCF should be around 1.0 on average
        assert 0.5 < np.mean(pcf) < 2.0, f"Mean PCF {np.mean(pcf)} seems unreasonable"
    
    def test_pcf_clustered_higher_than_random(self):
        """Clustered points should have higher short-range PCF than random."""
        grid_size = 100
        
        # Create clustered distribution
        clustered_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        for _ in range(10):
            cx, cy = np.random.randint(10, 90, 2)
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if np.random.random() < 0.8:
                        clustered_grid[(cx+dx) % grid_size, (cy+dy) % grid_size] = 1
        
        # Create random distribution with same density
        n_clustered = np.sum(clustered_grid == 1)
        random_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        positions = np.random.permutation(grid_size * grid_size)[:n_clustered]
        for pos in positions:
            random_grid[pos // grid_size, pos % grid_size] = 1
        
        # Compute PCFs
        clustered_pos = np.argwhere(clustered_grid == 1)
        random_pos = np.argwhere(random_grid == 1)
        
        _, pcf_clustered, _ = compute_pcf_periodic_fast(
            clustered_pos, clustered_pos, (grid_size, grid_size), 20.0, 20, True
        )
        _, pcf_random, _ = compute_pcf_periodic_fast(
            random_pos, random_pos, (grid_size, grid_size), 20.0, 20, True
        )
        
        # Short-range PCF should be higher for clustered
        short_range_clustered = np.mean(pcf_clustered[:5])
        short_range_random = np.mean(pcf_random[:5])
        
        assert short_range_clustered > short_range_random, \
            f"Clustered PCF ({short_range_clustered:.2f}) should be > random ({short_range_random:.2f})"
    
    def test_compute_all_pcfs_keys(self, medium_grid):
        """compute_all_pcfs_fast should return dict with correct keys."""
        results = compute_all_pcfs_fast(medium_grid, 15.0, 20)
        
        assert 'prey_prey' in results
        assert 'pred_pred' in results
        assert 'prey_pred' in results
    
    def test_compute_all_pcfs_structure(self, medium_grid):
        """Each PCF result should be a tuple of (distances, pcf, n_pairs)."""
        results = compute_all_pcfs_fast(medium_grid, 15.0, 20)
        
        for key in ['prey_prey', 'pred_pred', 'prey_pred']:
            assert len(results[key]) == 3, f"{key} should have 3 elements"
            dist, pcf, n_pairs = results[key]
            assert len(dist) == 20
            assert len(pcf) == 20
            assert isinstance(n_pairs, int)


# ============================================================================
# TEST: CLUSTER MEASUREMENT
# ============================================================================

class TestClusterMeasurement:
    """Tests for cluster size measurement."""
    
    def test_cluster_known_sizes(self, clustered_grid):
        """Should correctly identify known cluster sizes."""
        sizes = measure_cluster_sizes_fast(clustered_grid, 1)
        sizes_sorted = sorted(sizes, reverse=True)
        
        # Expected: 9 (3x3), 8 (2x4), 5 (L-shape), 1 (single)
        expected = [9, 8, 5, 1]
        
        assert len(sizes) == 4, f"Expected 4 clusters, got {len(sizes)}"
        assert list(sizes_sorted) == expected, f"Expected {expected}, got {list(sizes_sorted)}"
    
    def test_cluster_empty_grid(self):
        """Should return empty array for grid with no target species."""
        grid = np.zeros((20, 20), dtype=np.int32)
        sizes = measure_cluster_sizes_fast(grid, 1)
        
        assert len(sizes) == 0
    
    def test_cluster_full_grid(self):
        """Single cluster when grid is full of one species."""
        grid = np.ones((10, 10), dtype=np.int32)
        sizes = measure_cluster_sizes_fast(grid, 1)
        
        assert len(sizes) == 1
        assert sizes[0] == 100
    
    def test_cluster_diagonal_not_connected(self):
        """Diagonally adjacent cells should NOT be connected (4-connectivity)."""
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[0, 0] = 1
        grid[1, 1] = 1  # Diagonal from (0,0)
        grid[2, 2] = 1  # Diagonal from (1,1)
        
        sizes = measure_cluster_sizes_fast(grid, 1)
        
        # Each cell should be its own cluster (4-connectivity)
        assert len(sizes) == 3, f"Expected 3 separate clusters, got {len(sizes)}"
        assert all(s == 1 for s in sizes)
    
    def test_cluster_orthogonal_connected(self):
        """Orthogonally adjacent cells should be connected."""
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[2, 1:4] = 1  # Horizontal line of 3
        grid[1, 2] = 1    # One above middle
        grid[3, 2] = 1    # One below middle
        
        sizes = measure_cluster_sizes_fast(grid, 1)
        
        # Should be one connected cluster of 5
        assert len(sizes) == 1
        assert sizes[0] == 5
    
    def test_cluster_species_separation(self):
        """Clusters of different species should be separate."""
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[0:3, 0:3] = 1  # 9 prey
        grid[5:8, 5:8] = 2  # 9 predators
        
        prey_sizes = measure_cluster_sizes_fast(grid, 1)
        pred_sizes = measure_cluster_sizes_fast(grid, 2)
        
        assert len(prey_sizes) == 1
        assert prey_sizes[0] == 9
        assert len(pred_sizes) == 1
        assert pred_sizes[0] == 9
    
    def test_cluster_total_cells(self, medium_grid):
        """Total cells in clusters should equal total cells of that species."""
        for species in [1, 2]:
            sizes = measure_cluster_sizes_fast(medium_grid, species)
            total_in_clusters = sum(sizes)
            total_in_grid = np.sum(medium_grid == species)
            
            assert total_in_clusters == total_in_grid, \
                f"Species {species}: cluster total {total_in_clusters} != grid total {total_in_grid}"


# ============================================================================
# TEST: WARMUP FUNCTION
# ============================================================================

class TestWarmup:
    """Tests for JIT warmup function."""
    
    def test_warmup_runs_without_error(self):
        """Warmup should complete without errors."""
        try:
            warmup_numba_kernels(50)
        except Exception as e:
            pytest.fail(f"Warmup failed with error: {e}")
    
    def test_warmup_compiles_kernel(self):
        """After warmup, kernel should run faster."""
        import time
        
        # First call (might trigger compilation)
        warmup_numba_kernels(30)
        
        # Timed call (should be fast)
        np.random.seed(42)
        grid = np.random.choice([0, 1, 2], (30, 30), p=[0.5, 0.3, 0.2]).astype(np.int32)
        prey_death = np.full((30, 30), 0.05, dtype=np.float64)
        prey_death[grid != 1] = np.nan
        
        kernel = PPKernel(30, 30, "moore")
        
        t0 = time.perf_counter()
        for _ in range(10):
            kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1)
        elapsed = time.perf_counter() - t0
        
        # Should complete quickly (less than 1 second for 10 iterations)
        assert elapsed < 1.0, f"Kernel too slow after warmup: {elapsed:.2f}s"


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_cell_grid(self):
        """Should handle 1x1 grid."""
        grid = np.array([[1]], dtype=np.int32)
        prey_death = np.array([[0.05]], dtype=np.float64)
        
        kernel = PPKernel(1, 1, "moore")
        # Should not crash
        kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1)
    
    def test_very_small_grid(self):
        """Should handle very small grids."""
        grid = np.array([[1, 0], [2, 1]], dtype=np.int32)
        prey_death = np.full((2, 2), np.nan, dtype=np.float64)
        prey_death[grid == 1] = 0.05
        
        kernel = PPKernel(2, 2, "moore")
        for _ in range(10):
            kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1)
        
        assert grid.min() >= 0
        assert grid.max() <= 2
    
    def test_all_empty_grid(self):
        """Should handle grid with no organisms."""
        grid = np.zeros((20, 20), dtype=np.int32)
        prey_death = np.full((20, 20), np.nan, dtype=np.float64)
        
        kernel = PPKernel(20, 20, "moore")
        kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1)
        
        # Grid should still be all zeros
        assert np.all(grid == 0)
    
    def test_all_prey_grid(self):
        """Should handle grid with only prey."""
        grid = np.ones((20, 20), dtype=np.int32)
        prey_death = np.full((20, 20), 0.05, dtype=np.float64)
        
        kernel = PPKernel(20, 20, "moore")
        for _ in range(10):
            kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1)
        
        # Some prey should have died
        assert np.sum(grid == 0) > 0
    
    def test_all_predator_grid(self):
        """Should handle grid with only predators."""
        grid = np.full((20, 20), 2, dtype=np.int32)
        prey_death = np.full((20, 20), np.nan, dtype=np.float64)
        
        kernel = PPKernel(20, 20, "moore")
        for _ in range(50):
            kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1)
        
        # All predators should have died (no prey to eat)
        assert np.sum(grid == 2) < 400  # Most should be dead
    
    def test_extreme_parameters(self):
        """Should handle extreme parameter values."""
        np.random.seed(42)
        grid = np.random.choice([0, 1, 2], (30, 30), p=[0.5, 0.3, 0.2]).astype(np.int32)
        prey_death = np.full((30, 30), 0.5, dtype=np.float64)
        prey_death[grid != 1] = np.nan
        
        kernel = PPKernel(30, 30, "moore")
        
        # High death rates
        kernel.update(grid, prey_death, 0.99, 0.99, 0.99, 0.99)
        
        # Low death rates
        grid = np.random.choice([0, 1, 2], (30, 30), p=[0.5, 0.3, 0.2]).astype(np.int32)
        prey_death = np.full((30, 30), 0.001, dtype=np.float64)
        prey_death[grid != 1] = np.nan
        kernel.update(grid, prey_death, 0.01, 0.01, 0.01, 0.01)
        
        # Should not crash
        assert True


# ============================================================================
# MAIN
# ============================================================================

def run_tests_without_pytest():
    """Run tests without pytest for basic verification."""
    print("=" * 60)
    print("Running tests without pytest...")
    print("=" * 60)
    
    test_classes = [
        TestNumbaAvailability,
        TestPPKernel,
        TestPCFComputation,
        TestClusterMeasurement,
        TestWarmup,
        TestEdgeCases,
    ]
    
    # Create fixtures
    np.random.seed(42)
    small_grid = np.random.choice([0, 1, 2], (20, 20), p=[0.5, 0.3, 0.2]).astype(np.int32)
    medium_grid = np.random.choice([0, 1, 2], (50, 50), p=[0.55, 0.30, 0.15]).astype(np.int32)
    large_grid = np.random.choice([0, 1, 2], (100, 100), p=[0.55, 0.30, 0.15]).astype(np.int32)
    
    clustered_grid = np.zeros((30, 30), dtype=np.int32)
    clustered_grid[2:5, 2:5] = 1
    clustered_grid[10:12, 10:14] = 1
    clustered_grid[20, 20] = 1
    clustered_grid[25, 25:28] = 1
    clustered_grid[26:28, 25] = 1
    
    prey_death_array = np.full(medium_grid.shape, np.nan, dtype=np.float64)
    prey_death_array[medium_grid == 1] = 0.05
    
    fixtures = {
        'small_grid': small_grid,
        'medium_grid': medium_grid,
        'large_grid': large_grid,
        'clustered_grid': clustered_grid,
        'prey_death_array': prey_death_array,
    }
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                method = getattr(instance, method_name)
                
                # Get fixture arguments
                import inspect
                sig = inspect.signature(method)
                kwargs = {}
                for param in sig.parameters:
                    if param in fixtures:
                        # Create fresh copy for each test
                        kwargs[param] = fixtures[param].copy()
                
                try:
                    method(**kwargs)
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--no-pytest":
        success = run_tests_without_pytest()
        sys.exit(0 if success else 1)
    else:
        try:
            import pytest
            sys.exit(pytest.main([__file__, "-v"]))
        except ImportError:
            print("pytest not installed, running basic tests...")
            success = run_tests_without_pytest()
            sys.exit(0 if success else 1)