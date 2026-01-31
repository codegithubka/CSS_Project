"""
Tests for Numba-optimized kernels and spatial analysis functions.

Covers:
- Cluster detection (measure_cluster_sizes_fast, detect_clusters_fast, get_cluster_stats_fast)
- Pair Correlation Function (PCF) computation
- PPKernel class
- Warmup and seeding functions
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.numba_optimized import (
    set_numba_seed,
    PPKernel,
    measure_cluster_sizes_fast,
    detect_clusters_fast,
    get_cluster_stats_fast,
    compute_pcf_periodic_fast,
    compute_all_pcfs_fast,
    warmup_numba_kernels,
    NUMBA_AVAILABLE,
)


# =============================================================================
# Seed and Warmup Tests
# =============================================================================


class TestSeedingAndWarmup:
    """Tests for RNG seeding and kernel warmup."""

    def test_set_numba_seed_does_not_raise(self):
        """set_numba_seed should execute without error."""
        set_numba_seed(42)  # Should not raise

    def test_warmup_numba_kernels_does_not_raise(self):
        """warmup_numba_kernels should execute without error."""
        warmup_numba_kernels(grid_size=20, directed_hunting=False)
        warmup_numba_kernels(grid_size=20, directed_hunting=True)

    def test_numba_available_flag(self):
        """NUMBA_AVAILABLE should be True when numba is installed."""
        assert NUMBA_AVAILABLE is True


# =============================================================================
# Cluster Detection Tests
# =============================================================================


class TestMeasureClusterSizesFast:
    """Tests for measure_cluster_sizes_fast function."""

    def test_empty_grid_returns_empty_array(self, empty_grid_10x10):
        """Empty grid should return no clusters."""
        sizes = measure_cluster_sizes_fast(empty_grid_10x10, species=1)
        assert len(sizes) == 0

    def test_single_cluster_correct_size(self, single_cluster_grid):
        """Single connected cluster should return correct size."""
        sizes = measure_cluster_sizes_fast(single_cluster_grid, species=1)
        assert len(sizes) == 1
        assert sizes[0] == 4  # 2x2 block

    def test_two_clusters_correct_sizes(self, two_cluster_grid):
        """Two separate clusters should return two sizes."""
        sizes = measure_cluster_sizes_fast(two_cluster_grid, species=1)
        assert len(sizes) == 2
        assert sorted(sizes) == [3, 4]  # Clusters of size 3 and 4

    def test_periodic_boundary_connects_clusters(self, periodic_cluster_grid):
        """Clusters should connect via periodic boundaries (Moore)."""
        sizes = measure_cluster_sizes_fast(periodic_cluster_grid, species=1, neighborhood="moore")
        # All 3 cells should be one cluster due to periodic connections
        assert len(sizes) == 1
        assert sizes[0] == 3

    def test_neumann_neighborhood_fewer_connections(self):
        """Von Neumann should produce more clusters than Moore for diagonal patterns."""
        grid = np.zeros((5, 5), dtype=np.int32)
        # Diagonal line - connected in Moore, not in Neumann
        grid[0, 0] = 1
        grid[1, 1] = 1
        grid[2, 2] = 1

        sizes_moore = measure_cluster_sizes_fast(grid, species=1, neighborhood="moore")
        sizes_neumann = measure_cluster_sizes_fast(grid, species=1, neighborhood="neumann")

        assert len(sizes_moore) == 1  # One connected cluster
        assert len(sizes_neumann) == 3  # Three separate cells

    def test_species_filtering(self, mixed_grid_10x10):
        """Should only count clusters for specified species."""
        prey_sizes = measure_cluster_sizes_fast(mixed_grid_10x10, species=1)
        pred_sizes = measure_cluster_sizes_fast(mixed_grid_10x10, species=2)

        assert sum(prey_sizes) == 9  # Total prey count
        assert sum(pred_sizes) == 4  # Total predator count

    def test_checkerboard_many_clusters(self, checkerboard_grid):
        """Checkerboard pattern should produce many small clusters in Neumann."""
        sizes = measure_cluster_sizes_fast(checkerboard_grid, species=1, neighborhood="neumann")
        # Each cell is isolated in Neumann neighborhood
        assert len(sizes) == 18  # Half of 6x6 = 18 cells
        assert all(s == 1 for s in sizes)


class TestDetectClustersFast:
    """Tests for detect_clusters_fast function."""

    def test_returns_labels_and_size_dict(self, single_cluster_grid):
        """Should return both label array and size dictionary."""
        labels, sizes = detect_clusters_fast(single_cluster_grid, species=1)

        assert isinstance(labels, np.ndarray)
        assert labels.shape == single_cluster_grid.shape
        assert isinstance(sizes, dict)

    def test_labels_match_cluster_membership(self, two_cluster_grid):
        """Labels should correctly identify cluster membership."""
        labels, sizes = detect_clusters_fast(two_cluster_grid, species=1)

        # All cells in a cluster should have same label
        assert labels[0, 0] == labels[0, 1] == labels[1, 0]  # Cluster 1
        assert labels[4, 4] == labels[4, 5] == labels[5, 4] == labels[5, 5]  # Cluster 2

        # Different clusters should have different labels
        assert labels[0, 0] != labels[4, 4]

    def test_non_species_cells_have_zero_label(self, mixed_grid_10x10):
        """Cells not belonging to target species should have label 0."""
        labels, _ = detect_clusters_fast(mixed_grid_10x10, species=1)

        # Predator cells and empty cells should be 0
        assert labels[6, 6] == 0  # Predator cell
        assert labels[5, 5] == 0  # Empty cell

    def test_size_dict_matches_cluster_count(self, two_cluster_grid):
        """Size dictionary should have entry for each cluster."""
        labels, sizes = detect_clusters_fast(two_cluster_grid, species=1)

        assert len(sizes) == 2
        assert set(sizes.values()) == {3, 4}


class TestGetClusterStatsFast:
    """Tests for get_cluster_stats_fast function."""

    def test_returns_comprehensive_stats(self, single_cluster_grid):
        """Should return dictionary with all expected keys."""
        stats = get_cluster_stats_fast(single_cluster_grid, species=1)

        expected_keys = [
            "n_clusters",
            "sizes",
            "largest",
            "largest_fraction",
            "mean_size",
            "size_distribution",
            "labels",
            "size_dict",
        ]
        for key in expected_keys:
            assert key in stats

    def test_empty_grid_stats(self, empty_grid_10x10):
        """Empty grid should return zero-valued stats."""
        stats = get_cluster_stats_fast(empty_grid_10x10, species=1)

        assert stats["n_clusters"] == 0
        assert stats["largest"] == 0
        assert stats["largest_fraction"] == 0.0
        assert stats["mean_size"] == 0.0

    def test_largest_fraction_calculation(self, two_cluster_grid):
        """largest_fraction should be largest cluster / total population."""
        stats = get_cluster_stats_fast(two_cluster_grid, species=1)

        total_prey = 3 + 4  # Two clusters
        expected_fraction = 4 / total_prey

        assert stats["largest"] == 4
        assert abs(stats["largest_fraction"] - expected_fraction) < 1e-10

    def test_size_distribution_counts(self, checkerboard_grid):
        """size_distribution should count clusters of each size."""
        stats = get_cluster_stats_fast(checkerboard_grid, species=1, neighborhood="neumann")

        # All 18 clusters are size 1
        assert stats["size_distribution"] == {1: 18}

    def test_sizes_sorted_descending(self, two_cluster_grid):
        """sizes array should be sorted in descending order."""
        stats = get_cluster_stats_fast(two_cluster_grid, species=1)

        sizes = stats["sizes"]
        assert list(sizes) == sorted(sizes, reverse=True)


# =============================================================================
# PCF Tests
# =============================================================================


class TestComputePcfPeriodicFast:
    """Tests for compute_pcf_periodic_fast function."""

    def test_empty_positions_returns_ones(self):
        """PCF of empty positions should return 1.0 (no correlation)."""
        empty_pos = np.array([]).reshape(0, 2)
        grid_shape = (50, 50)

        dist, pcf, n_pairs = compute_pcf_periodic_fast(
            empty_pos, empty_pos, grid_shape, max_distance=10.0, n_bins=10
        )

        assert len(dist) == 10
        assert np.allclose(pcf, 1.0)
        assert n_pairs == 0

    def test_bin_centers_correct_spacing(self):
        """Bin centers should be evenly spaced."""
        pos = np.array([[10.0, 10.0], [15.0, 15.0]])
        grid_shape = (50, 50)

        dist, _, _ = compute_pcf_periodic_fast(
            pos, pos, grid_shape, max_distance=20.0, n_bins=10, self_correlation=True
        )

        expected_spacing = 20.0 / 10
        actual_spacing = dist[1] - dist[0]
        assert abs(actual_spacing - expected_spacing) < 1e-10

    def test_self_correlation_excludes_self_pairs(self):
        """Self-correlation should not count i==j pairs."""
        # Single point - self correlation should find 0 pairs
        pos = np.array([[25.0, 25.0]])
        grid_shape = (50, 50)

        _, _, n_pairs = compute_pcf_periodic_fast(
            pos, pos, grid_shape, max_distance=10.0, self_correlation=True
        )

        assert n_pairs == 0

    def test_cross_correlation_counts_all_pairs(self):
        """Cross-correlation should count all i-j pairs."""
        pos_i = np.array([[10.0, 10.0]])
        pos_j = np.array([[12.0, 10.0]])  # Distance = 2
        grid_shape = (50, 50)

        _, _, n_pairs = compute_pcf_periodic_fast(
            pos_i, pos_j, grid_shape, max_distance=10.0, self_correlation=False
        )

        assert n_pairs == 1

    def test_periodic_distance_calculation(self):
        """Distances should respect periodic boundaries."""
        # Two points on opposite edges - should be close via periodicity
        pos_i = np.array([[0.5, 25.0]])
        pos_j = np.array([[49.5, 25.0]])  # Periodic distance = 1.0
        grid_shape = (50, 50)

        _, pcf, n_pairs = compute_pcf_periodic_fast(
            pos_i, pos_j, grid_shape, max_distance=5.0, n_bins=5, self_correlation=False
        )

        assert n_pairs == 1  # Should find the pair


class TestComputeAllPcfsFast:
    """Tests for compute_all_pcfs_fast function."""

    def test_returns_all_three_pcfs(self, mixed_grid_10x10):
        """Should return prey-prey, pred-pred, and prey-pred PCFs."""
        results = compute_all_pcfs_fast(mixed_grid_10x10, max_distance=3.0, n_bins=5)

        assert "prey_prey" in results
        assert "pred_pred" in results
        assert "prey_pred" in results

    def test_each_pcf_has_correct_structure(self, mixed_grid_10x10):
        """Each PCF result should be (distances, values, count) tuple."""
        results = compute_all_pcfs_fast(mixed_grid_10x10, max_distance=3.0, n_bins=5)

        for key in ["prey_prey", "pred_pred", "prey_pred"]:
            dist, pcf, n = results[key]
            assert isinstance(dist, np.ndarray)
            assert isinstance(pcf, np.ndarray)
            assert isinstance(n, int)
            assert len(dist) == len(pcf) == 5

    def test_default_max_distance(self, mixed_grid_10x10):
        """Default max_distance should be grid_size / 4."""
        results = compute_all_pcfs_fast(mixed_grid_10x10, n_bins=5)

        # For 10x10 grid, default max_distance = 2.5
        dist, _, _ = results["prey_prey"]
        assert dist[-1] < 2.5  # Last bin center should be less than max

    def test_empty_species_returns_ones(self, prey_only_grid_10x10):
        """PCF for missing species should return 1.0."""
        results = compute_all_pcfs_fast(prey_only_grid_10x10, max_distance=3.0, n_bins=5)

        _, pred_pred_pcf, _ = results["pred_pred"]
        assert np.allclose(pred_pred_pcf, 1.0)


# =============================================================================
# PPKernel Tests
# =============================================================================


class TestPPKernel:
    """Tests for PPKernel class."""

    def test_kernel_initialization_moore(self):
        """Moore kernel should have 8-direction offsets."""
        kernel = PPKernel(10, 10, neighborhood="moore")
        assert len(kernel._dr) == 8
        assert len(kernel._dc) == 8

    def test_kernel_initialization_neumann(self):
        """Von Neumann kernel should have 4-direction offsets."""
        kernel = PPKernel(10, 10, neighborhood="von_neumann")
        assert len(kernel._dr) == 4
        assert len(kernel._dc) == 4

    def test_kernel_preallocates_buffer(self):
        """Kernel should preallocate occupied_buffer."""
        kernel = PPKernel(15, 20)
        assert kernel._occupied_buffer.shape == (15 * 20, 2)

    def test_kernel_update_modifies_grid(self):
        """update() should modify the grid in place."""
        set_numba_seed(42)
        kernel = PPKernel(10, 10, neighborhood="moore", directed_hunting=False)

        grid = np.zeros((10, 10), dtype=np.int32)
        grid[3:6, 3:6] = 1  # Prey block
        grid[7, 7] = 2  # One predator

        prey_death_arr = np.full((10, 10), 0.05, dtype=np.float64)
        prey_death_arr[grid != 1] = np.nan

        initial_grid = grid.copy()

        kernel.update(
            grid, prey_death_arr,
            prey_birth=0.3, prey_death=0.05,
            pred_birth=0.5, pred_death=0.1,
        )

        # Grid should have changed
        assert not np.array_equal(grid, initial_grid)

    def test_kernel_update_preserves_dtype(self):
        """update() should preserve grid dtype."""
        kernel = PPKernel(10, 10)

        grid = np.zeros((10, 10), dtype=np.int32)
        grid[5, 5] = 1
        prey_death_arr = np.full((10, 10), 0.05, dtype=np.float64)

        kernel.update(grid, prey_death_arr, 0.2, 0.05, 0.2, 0.1)

        assert grid.dtype == np.int32

    def test_kernel_directed_hunting_option(self):
        """directed_hunting flag should be stored correctly."""
        kernel_random = PPKernel(10, 10, directed_hunting=False)
        kernel_directed = PPKernel(10, 10, directed_hunting=True)

        assert kernel_random.directed_hunting is False
        assert kernel_directed.directed_hunting is True

    def test_kernel_update_with_evolution(self):
        """update() should handle evolution parameters."""
        set_numba_seed(42)
        kernel = PPKernel(10, 10)

        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2:5, 2:5] = 1  # Prey
        prey_death_arr = np.full((10, 10), 0.05, dtype=np.float64)
        prey_death_arr[grid != 1] = np.nan

        # Run with evolution active
        kernel.update(
            grid, prey_death_arr,
            prey_birth=0.3, prey_death=0.05,
            pred_birth=0.5, pred_death=0.1,
            evolve_sd=0.02, evolve_min=0.01, evolve_max=0.15,
            evolution_stopped=False,
        )

        # Check that new prey have evolved values
        new_prey_mask = (grid == 1) & ~np.isnan(prey_death_arr)
        if np.any(new_prey_mask):
            values = prey_death_arr[new_prey_mask]
            assert np.all(values >= 0.01)
            assert np.all(values <= 0.15)


# =============================================================================
# Edge Cases
# =============================================================================


class TestNumbaEdgeCases:
    """Edge case tests for Numba functions."""

    def test_cluster_detection_1x1_grid(self):
        """Should handle minimal 1x1 grid."""
        grid = np.array([[1]], dtype=np.int32)
        sizes = measure_cluster_sizes_fast(grid, species=1)
        assert len(sizes) == 1
        assert sizes[0] == 1

    def test_cluster_detection_full_grid(self):
        """Should handle grid completely filled with one species."""
        grid = np.ones((10, 10), dtype=np.int32)
        stats = get_cluster_stats_fast(grid, species=1)

        assert stats["n_clusters"] == 1
        assert stats["largest"] == 100
        assert stats["largest_fraction"] == 1.0

    def test_pcf_single_point(self):
        """PCF should handle single-point case."""
        grid = np.zeros((20, 20), dtype=np.int32)
        grid[10, 10] = 1

        results = compute_all_pcfs_fast(grid, max_distance=5.0, n_bins=5)
        _, pcf, n = results["prey_prey"]

        assert n == 0  # No pairs with single point

    def test_kernel_empty_grid(self):
        """Kernel should handle completely empty grid."""
        kernel = PPKernel(10, 10)
        grid = np.zeros((10, 10), dtype=np.int32)
        prey_death_arr = np.full((10, 10), np.nan, dtype=np.float64)

        # Should not raise
        kernel.update(grid, prey_death_arr, 0.2, 0.05, 0.2, 0.1)

        # Grid should still be empty
        assert np.sum(grid) == 0

    def test_kernel_high_death_rates(self):
        """Kernel should handle extreme death rates."""
        set_numba_seed(42)
        kernel = PPKernel(10, 10)

        grid = np.zeros((10, 10), dtype=np.int32)
        grid[::2, ::2] = 1  # Sparse prey
        prey_death_arr = np.full((10, 10), 0.99, dtype=np.float64)  # Very high death
        prey_death_arr[grid != 1] = np.nan

        initial_prey = np.sum(grid == 1)

        kernel.update(grid, prey_death_arr, 0.2, 0.99, 0.2, 0.1)

        # Most prey should die
        final_prey = np.sum(grid == 1)
        assert final_prey < initial_prey

    def test_cluster_large_grid_performance(self):
        """Cluster detection should complete quickly on moderate grid."""
        import time

        grid = np.zeros((200, 200), dtype=np.int32)
        # Random scattered prey
        np.random.seed(42)
        grid[np.random.random((200, 200)) < 0.3] = 1

        start = time.perf_counter()
        stats = get_cluster_stats_fast(grid, species=1)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0  # Should complete in under 1 second
        assert stats["n_clusters"] > 0