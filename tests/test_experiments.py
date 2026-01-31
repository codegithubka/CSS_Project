"""
Tests for experiments.py utility functions and simulation runner.

Covers:
- Utility functions (generate_unique_seed, count_populations, etc.)
- I/O functions (save_results_jsonl, load_results_jsonl, save_results_npz)
- run_single_simulation with real (tiny) configs
- Basic phase runner validation
"""

import pytest
import json
import numpy as np
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions to test
from experiments import (
    generate_unique_seed,
    count_populations,
    get_evolved_stats,
    average_pcfs,
    save_results_jsonl,
    load_results_jsonl,
    save_results_npz,
    run_single_simulation,
    PHASE_RUNNERS,
)

# Import the real Config for integration tests
from models.config import Config


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestGenerateUniqueSeed:
    """Tests for generate_unique_seed function."""

    def test_same_params_same_rep_same_seed(self):
        """Identical inputs should produce identical seeds."""
        params = {"a": 1, "b": 2.5}
        seed1 = generate_unique_seed(params, rep=0)
        seed2 = generate_unique_seed(params, rep=0)
        assert seed1 == seed2

    def test_different_rep_different_seed(self):
        """Different rep values should produce different seeds."""
        params = {"a": 1, "b": 2.5}
        seed1 = generate_unique_seed(params, rep=0)
        seed2 = generate_unique_seed(params, rep=1)
        assert seed1 != seed2

    def test_different_params_different_seed(self):
        """Different parameters should produce different seeds."""
        params1 = {"a": 1}
        params2 = {"a": 2}
        seed1 = generate_unique_seed(params1, rep=0)
        seed2 = generate_unique_seed(params2, rep=0)
        assert seed1 != seed2

    def test_key_order_does_not_matter(self):
        """Dict key order should not affect seed (sorted keys)."""
        params1 = {"b": 2, "a": 1}
        params2 = {"a": 1, "b": 2}
        assert generate_unique_seed(params1, 0) == generate_unique_seed(params2, 0)

    def test_returns_positive_integer(self):
        """Seed should be a positive integer."""
        seed = generate_unique_seed({"x": 100}, rep=5)
        assert isinstance(seed, int)
        assert seed >= 0


class TestCountPopulations:
    """Tests for count_populations function."""

    def test_empty_grid(self):
        """Empty grid should return (total_cells, 0, 0)."""
        grid = np.zeros((5, 5), dtype=int)
        empty, prey, pred = count_populations(grid)
        assert empty == 25
        assert prey == 0
        assert pred == 0

    def test_full_prey_grid(self):
        """Grid full of prey should return (0, total, 0)."""
        grid = np.ones((4, 4), dtype=int)
        empty, prey, pred = count_populations(grid)
        assert empty == 0
        assert prey == 16
        assert pred == 0

    def test_mixed_population(self):
        """Mixed grid should return correct counts."""
        grid = np.array([[0, 1, 2], [1, 0, 1], [2, 2, 0]])
        empty, prey, pred = count_populations(grid)
        assert empty == 3
        assert prey == 3
        assert pred == 3

    def test_returns_integers(self):
        """Counts should be Python ints, not numpy types."""
        grid = np.array([[0, 1], [2, 1]])
        empty, prey, pred = count_populations(grid)
        assert type(empty) is int
        assert type(prey) is int
        assert type(pred) is int


class TestGetEvolvedStats:
    """Tests for get_evolved_stats function."""

    def test_missing_param_returns_nan(self):
        """Missing parameter should return NaN stats."""
        mock_model = MagicMock()
        mock_model.cell_params.get.return_value = None

        stats = get_evolved_stats(mock_model, "nonexistent")

        assert np.isnan(stats["mean"])
        assert np.isnan(stats["std"])
        assert stats["n"] == 0

    def test_all_nan_returns_nan(self):
        """Array of all NaN should return NaN stats."""
        mock_model = MagicMock()
        mock_model.cell_params.get.return_value = np.array([np.nan, np.nan, np.nan])

        stats = get_evolved_stats(mock_model, "param")

        assert np.isnan(stats["mean"])
        assert stats["n"] == 0

    def test_valid_values_return_correct_stats(self):
        """Valid values should return correct statistics."""
        mock_model = MagicMock()
        mock_model.cell_params.get.return_value = np.array([1.0, 2.0, 3.0, np.nan])

        stats = get_evolved_stats(mock_model, "param")

        assert stats["mean"] == 2.0
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert stats["n"] == 3


class TestAveragePcfs:
    """Tests for average_pcfs function."""

    def test_empty_list_returns_empty(self):
        """Empty input should return empty arrays."""
        dist, mean, se = average_pcfs([])
        assert len(dist) == 0
        assert len(mean) == 0
        assert len(se) == 0

    def test_single_pcf_returns_itself(self):
        """Single PCF should return itself as mean."""
        distances = np.array([1.0, 2.0, 3.0])
        values = np.array([0.5, 1.0, 1.5])
        pcf_list = [(distances, values, 100)]

        dist, mean, se = average_pcfs(pcf_list)

        assert np.array_equal(dist, distances)
        assert np.array_equal(mean, values)
        assert np.allclose(se, 0.0)  # No variance with single sample

    def test_multiple_pcfs_averaged(self):
        """Multiple PCFs should be averaged correctly."""
        d = np.array([1.0, 2.0])
        pcf_list = [
            (d, np.array([1.0, 2.0]), 10),
            (d, np.array([1.2, 1.8]), 12),
        ]

        dist, mean, se = average_pcfs(pcf_list)

        expected_mean = np.array([1.1, 1.9])
        assert np.allclose(mean, expected_mean)
        assert len(se) == 2


# =============================================================================
# I/O Function Tests
# =============================================================================


class TestSaveLoadJsonl:
    """Tests for JSONL save/load functions."""

    def test_save_and_load_roundtrip(self, temp_output_dir, sample_results):
        """Data should survive save/load roundtrip."""
        output_path = temp_output_dir / "test.jsonl"

        save_results_jsonl(sample_results, output_path)
        loaded = load_results_jsonl(output_path)

        assert len(loaded) == len(sample_results)
        for orig, load in zip(sample_results, loaded):
            assert orig["prey_birth"] == load["prey_birth"]
            assert orig["prey_mean"] == load["prey_mean"]

    def test_each_line_is_valid_json(self, temp_output_dir):
        """Each line should be independently valid JSON."""
        results = [{"a": 1}, {"b": 2}]
        output_path = temp_output_dir / "test.jsonl"

        save_results_jsonl(results, output_path)

        with open(output_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # Should not raise

    def test_handles_numpy_arrays(self, temp_output_dir):
        """Should handle numpy arrays via default=str."""
        results = [{"array": np.array([1, 2, 3])}]
        output_path = temp_output_dir / "test.jsonl"

        save_results_jsonl(results, output_path)  # Should not raise

    def test_load_nonexistent_file_raises(self, temp_output_dir):
        """Loading nonexistent file should raise."""
        with pytest.raises(FileNotFoundError):
            load_results_jsonl(temp_output_dir / "nonexistent.jsonl")


class TestSaveResultsNpz:
    """Tests for save_results_npz function."""

    def test_creates_npz_file(self, temp_output_dir):
        """Should create a valid NPZ file."""
        results = [{"energy": [1, 2, 3]}, {"energy": [4, 5, 6]}]
        output_path = temp_output_dir / "test.npz"

        save_results_npz(results, output_path)

        assert output_path.exists()

    def test_npz_contains_prefixed_keys(self, temp_output_dir):
        """Keys should be prefixed with run index."""
        results = [{"x": [1]}, {"x": [2]}]
        output_path = temp_output_dir / "test.npz"

        save_results_npz(results, output_path)

        data = np.load(output_path)
        assert "run_0_x" in data.files
        assert "run_1_x" in data.files


# =============================================================================
# run_single_simulation Tests (using real Config)
# =============================================================================


@pytest.fixture
def tiny_config():
    """Tiny config for fast integration tests."""
    return Config(
        grid_size=10,
        n_prey_death=2,
        n_replicates=1,
        warmup_steps=3,
        measurement_steps=5,
        collect_pcf=False,
        save_timeseries=False,
        directed_hunting=False,
    )


@pytest.fixture
def tiny_config_with_pcf():
    """Tiny config with PCF enabled."""
    return Config(
        grid_size=15,
        n_prey_death=2,
        n_replicates=1,
        warmup_steps=3,
        measurement_steps=5,
        collect_pcf=True,
        pcf_sample_rate=1.0,
        save_timeseries=False,
    )


@pytest.fixture
def tiny_config_with_timeseries():
    """Tiny config with timeseries enabled."""
    return Config(
        grid_size=10,
        n_prey_death=2,
        n_replicates=1,
        warmup_steps=3,
        measurement_steps=5,
        collect_pcf=False,
        save_timeseries=True,
        timeseries_subsample=1,
    )


@pytest.fixture
def tiny_config_directed():
    """Tiny config with directed hunting."""
    return Config(
        grid_size=10,
        n_prey_death=2,
        n_replicates=1,
        warmup_steps=3,
        measurement_steps=5,
        collect_pcf=False,
        save_timeseries=False,
        directed_hunting=True,
    )


class TestRunSingleSimulation:
    """Tests for run_single_simulation with real tiny configs."""

    def test_returns_required_keys(self, tiny_config):
        """Result should contain all required keys."""
        result = run_single_simulation(
            prey_birth=0.2,
            prey_death=0.05,
            predator_birth=0.8,
            predator_death=0.1,
            grid_size=10,
            seed=42,
            cfg=tiny_config,
            with_evolution=False,
            compute_pcf=False,
        )

        required_keys = [
            "prey_birth", "prey_death", "predator_birth", "predator_death",
            "grid_size", "seed", "prey_mean", "prey_std", "pred_mean", "pred_std",
            "prey_survived", "pred_survived", "prey_n_clusters", "pred_n_clusters",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_with_evolution_returns_evolution_stats(self, tiny_config):
        """Evolution mode should return evolution statistics."""
        result = run_single_simulation(
            prey_birth=0.2,
            prey_death=0.05,
            predator_birth=0.8,
            predator_death=0.1,
            grid_size=10,
            seed=42,
            cfg=tiny_config,
            with_evolution=True,
            compute_pcf=False,
        )

        assert "evolved_prey_death_mean" in result
        assert "evolve_sd" in result

    def test_with_pcf_returns_pcf_data(self, tiny_config_with_pcf):
        """PCF mode should return PCF statistics."""
        result = run_single_simulation(
            prey_birth=0.2,
            prey_death=0.05,
            predator_birth=0.8,
            predator_death=0.1,
            grid_size=15,
            seed=42,
            cfg=tiny_config_with_pcf,
            with_evolution=False,
            compute_pcf=True,
        )

        # PCF data should be present if both species survived
        if result["prey_survived"] and result["pred_survived"]:
            assert "pcf_distances" in result

    def test_with_timeseries_returns_population_history(self, tiny_config_with_timeseries):
        """Timeseries mode should return population history."""
        result = run_single_simulation(
            prey_birth=0.2,
            prey_death=0.05,
            predator_birth=0.8,
            predator_death=0.1,
            grid_size=10,
            seed=42,
            cfg=tiny_config_with_timeseries,
            with_evolution=False,
            compute_pcf=False,
        )

        assert "prey_timeseries" in result
        assert "pred_timeseries" in result
        assert len(result["prey_timeseries"]) > 0

    def test_seed_reproducibility(self, tiny_config):
        """Same seed should produce same results."""
        kwargs = dict(
            prey_birth=0.2,
            prey_death=0.05,
            predator_birth=0.8,
            predator_death=0.1,
            grid_size=10,
            seed=12345,
            cfg=tiny_config,
            with_evolution=False,
            compute_pcf=False,
        )

        result1 = run_single_simulation(**kwargs)
        result2 = run_single_simulation(**kwargs)

        assert result1["prey_mean"] == result2["prey_mean"]
        assert result1["pred_mean"] == result2["pred_mean"]

    def test_directed_hunting_mode(self, tiny_config_directed):
        """Should work with directed hunting enabled."""
        result = run_single_simulation(
            prey_birth=0.2,
            prey_death=0.05,
            predator_birth=0.8,
            predator_death=0.1,
            grid_size=10,
            seed=42,
            cfg=tiny_config_directed,
            with_evolution=False,
            compute_pcf=False,
        )

        assert "prey_mean" in result  # Completed successfully


# =============================================================================
# Phase Runner Tests
# =============================================================================


class TestPhaseRunners:
    """Basic tests for phase runner registration."""

    def test_all_phases_registered(self):
        """All phases should be in PHASE_RUNNERS dict."""
        assert 1 in PHASE_RUNNERS
        assert 2 in PHASE_RUNNERS
        assert 3 in PHASE_RUNNERS
        assert 4 in PHASE_RUNNERS
        assert 5 in PHASE_RUNNERS

    def test_phase_runners_are_callable(self):
        """Each phase runner should be callable."""
        for phase, runner in PHASE_RUNNERS.items():
            assert callable(runner), f"Phase {phase} runner is not callable"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """End-to-end integration tests with actual tiny simulations."""

    def test_full_simulation_tiny_grid(self, tiny_config):
        """Run actual simulation on tiny grid."""
        result = run_single_simulation(
            prey_birth=0.3,
            prey_death=0.05,
            predator_birth=0.5,
            predator_death=0.1,
            grid_size=8,
            seed=42,
            cfg=tiny_config,
            with_evolution=False,
            compute_pcf=False,
        )

        # Basic sanity checks
        assert result["prey_mean"] >= 0
        assert result["pred_mean"] >= 0
        assert isinstance(result["prey_survived"], bool)
        assert isinstance(result["pred_survived"], bool)

    def test_evolution_changes_death_rates(self, tiny_config):
        """Evolution should cause prey death rates to vary."""
        result = run_single_simulation(
            prey_birth=0.3,
            prey_death=0.05,
            predator_birth=0.5,
            predator_death=0.1,
            grid_size=15,
            seed=42,
            cfg=tiny_config,
            with_evolution=True,
            compute_pcf=False,
        )

        if result["prey_survived"]:
            assert "evolved_prey_death_mean" in result

    def test_save_load_integration(self, tiny_config, temp_output_dir):
        """Full save/load cycle with real simulation results."""
        results = []
        for seed in [1, 2, 3]:
            result = run_single_simulation(
                prey_birth=0.2,
                prey_death=0.05,
                predator_birth=0.6,
                predator_death=0.1,
                grid_size=8,
                seed=seed,
                cfg=tiny_config,
                with_evolution=False,
                compute_pcf=False,
            )
            results.append(result)

        output_path = temp_output_dir / "integration_test.jsonl"
        save_results_jsonl(results, output_path)

        loaded = load_results_jsonl(output_path)

        assert len(loaded) == 3
        for orig, load in zip(results, loaded):
            assert orig["seed"] == load["seed"]
            assert abs(orig["prey_mean"] - load["prey_mean"]) < 1e-10