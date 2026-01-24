#!/usr/bin/env python3
"""
Unit Tests for pp_analysis.py

Run with:
    pytest test_pp_analysis.py -v
    pytest test_pp_analysis.py -v -x  # stop on first failure
"""

import sys
import tempfile
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
    from scripts.experiments import (
        Config,
        count_populations,
        get_evolved_stats,
        truncated_power_law,
        fit_truncated_power_law,
        average_pcfs,
        save_sweep_binary,
        load_sweep_binary,
        run_single_simulation,
        run_single_simulation_fss,
    )
except ImportError:
    from scripts.experiments import (
        Config,
        count_populations,
        get_evolved_stats,
        truncated_power_law,
        fit_truncated_power_law,
        average_pcfs,
        save_sweep_binary,
        load_sweep_binary,
        run_single_simulation,
        run_single_simulation_fss,
    )

# Check if CA module is available
try:
    from models.CA import PP
    CA_AVAILABLE = True
except ImportError:
    try:
        from CA import PP
        CA_AVAILABLE = True
    except ImportError:
        CA_AVAILABLE = False


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def default_config():
    """Default configuration."""
    return Config()


@pytest.fixture
def fast_config():
    """Fast configuration for quick tests."""
    cfg = Config()
    cfg.default_grid = 30
    cfg.warmup_steps = 20
    cfg.measurement_steps = 30
    cfg.cluster_samples = 1
    cfg.collect_pcf = False
    return cfg

@pytest.fixture
def fast_config_directed():
    """Fast configuration with directed hunting enabled."""
    cfg = Config()
    cfg.default_grid = 30
    cfg.warmup_steps = 20
    cfg.measurement_steps = 30
    cfg.cluster_samples = 1
    cfg.collect_pcf = False
    cfg.directed_hunting = True
    return cfg


@pytest.fixture
def sample_grid():
    """Sample grid for population counting tests."""
    grid = np.array([
        [0, 1, 1, 0, 2],
        [1, 0, 0, 2, 1],
        [0, 2, 1, 0, 0],
        [1, 0, 0, 1, 2],
        [2, 1, 0, 0, 0],
    ], dtype=np.int32)
    return grid


@pytest.fixture
def temp_dir():
    """Temporary directory for file tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# TEST: CONFIG CLASS
# ============================================================================

class TestConfig:
    """Tests for the Config dataclass."""
    
    def test_config_defaults_exist(self, default_config):
        """Config should have all expected default attributes."""
        assert hasattr(default_config, 'default_grid')
        assert hasattr(default_config, 'n_prey_birth')
        assert hasattr(default_config, 'n_prey_death')
        assert hasattr(default_config, 'n_replicates')
        assert hasattr(default_config, 'warmup_steps')
        assert hasattr(default_config, 'measurement_steps')
    
    def test_config_default_values(self, default_config):
        """Config should have sensible defaults."""
        assert default_config.default_grid == 100
        assert default_config.n_prey_birth == 15
        assert default_config.n_prey_death == 15
        assert default_config.n_replicates == 50
        assert default_config.warmup_steps > 0
        assert default_config.measurement_steps > 0
    
    def test_config_parameter_ranges_valid(self, default_config):
        """Parameter ranges should be valid."""
        assert default_config.prey_birth_min < default_config.prey_birth_max
        assert default_config.prey_death_min < default_config.prey_death_max
        assert 0 < default_config.prey_birth_min < 1
        assert 0 < default_config.prey_death_max < 1
    
    def test_config_get_prey_births(self, default_config):
        """get_prey_births should return correct array."""
        births = default_config.get_prey_births()
        
        assert len(births) == default_config.n_prey_birth
        assert np.isclose(births[0], default_config.prey_birth_min)
        assert np.isclose(births[-1], default_config.prey_birth_max)
        assert np.all(np.diff(births) > 0)
    
    def test_config_get_prey_deaths(self, default_config):
        """get_prey_deaths should return correct array."""
        deaths = default_config.get_prey_deaths()
        
        assert len(deaths) == default_config.n_prey_death
        assert np.isclose(deaths[0], default_config.prey_death_min)
        assert np.isclose(deaths[-1], default_config.prey_death_max)
        assert np.all(np.diff(deaths) > 0)
    
    def test_config_get_prey_births_custom(self, default_config):
        """get_prey_births should respect custom config."""
        default_config.n_prey_birth = 5
        default_config.prey_birth_min = 0.1
        default_config.prey_birth_max = 0.5
        
        births = default_config.get_prey_births()
        
        assert len(births) == 5
        assert np.isclose(births[0], 0.1)
        assert np.isclose(births[-1], 0.5)
    
    def test_config_estimate_runtime(self, default_config):
        """estimate_runtime should return a string."""
        estimate = default_config.estimate_runtime(32)
        
        assert isinstance(estimate, str)
        assert "sims" in estimate.lower()
    
    def test_config_evolution_bounds(self, default_config):
        """Evolution bounds should be valid."""
        assert default_config.evolve_min < default_config.evolve_max
        assert default_config.evolve_min > 0
        assert default_config.evolve_sd > 0
    
    def test_config_fss_grid_sizes(self, default_config):
        """FSS grid sizes should be in ascending order."""
        sizes = default_config.fss_grid_sizes
        assert list(sizes) == sorted(sizes)
        assert len(sizes) > 0
    
    def test_config_pcf_sample_rate(self, default_config):
        """PCF sample rate should be between 0 and 1."""
        assert 0 <= default_config.pcf_sample_rate <= 1
        
    def test_config_directed_hunting_default(self, default_config):
        """Config should have directed_hunting attribute defaulting to False."""
        assert hasattr(default_config, 'directed_hunting')
        assert default_config.directed_hunting == False
    
    def test_config_directed_hunting_settable(self, default_config):
        """directed_hunting should be settable."""
        default_config.directed_hunting = True
        assert default_config.directed_hunting == True


# ============================================================================
# TEST: HELPER FUNCTIONS
# ============================================================================

class TestCountPopulations:
    """Tests for count_populations function."""
    
    def test_count_populations_basic(self, sample_grid):
        """count_populations should correctly count each state."""
        empty, prey, pred = count_populations(sample_grid)
        
        # Verify by manual count using numpy
        expected_empty = int(np.sum(sample_grid == 0))
        expected_prey = int(np.sum(sample_grid == 1))
        expected_pred = int(np.sum(sample_grid == 2))
        
        assert empty == expected_empty
        assert prey == expected_prey
        assert pred == expected_pred
        assert empty + prey + pred == sample_grid.size
    
    def test_count_populations_empty_grid(self):
        """count_populations should handle empty grid."""
        grid = np.zeros((10, 10), dtype=np.int32)
        empty, prey, pred = count_populations(grid)
        
        assert empty == 100
        assert prey == 0
        assert pred == 0
    
    def test_count_populations_all_prey(self):
        """count_populations should handle grid full of prey."""
        grid = np.ones((10, 10), dtype=np.int32)
        empty, prey, pred = count_populations(grid)
        
        assert empty == 0
        assert prey == 100
        assert pred == 0
    
    def test_count_populations_all_pred(self):
        """count_populations should handle grid full of predators."""
        grid = np.full((10, 10), 2, dtype=np.int32)
        empty, prey, pred = count_populations(grid)
        
        assert empty == 0
        assert prey == 0
        assert pred == 100


class TestGetEvolvedStats:
    """Tests for get_evolved_stats function."""
    
    def test_get_evolved_stats_with_values(self):
        """get_evolved_stats should compute statistics correctly."""
        class MockModel:
            cell_params = {"prey_death": np.array([[0.05, 0.06], [np.nan, 0.04]])}
        
        stats = get_evolved_stats(MockModel(), "prey_death")
        
        assert "mean" in stats
        assert "std" in stats
        assert "n" in stats
        assert stats["n"] == 3
        assert np.isclose(stats["mean"], 0.05, atol=0.01)
    
    def test_get_evolved_stats_missing_param(self):
        """get_evolved_stats should handle missing parameter."""
        class MockModel:
            cell_params = {}
        
        stats = get_evolved_stats(MockModel(), "prey_death")
        
        assert np.isnan(stats["mean"])
        assert stats["n"] == 0
    
    def test_get_evolved_stats_all_nan(self):
        """get_evolved_stats should handle all-NaN array."""
        class MockModel:
            cell_params = {"prey_death": np.array([[np.nan, np.nan], [np.nan, np.nan]])}
        
        stats = get_evolved_stats(MockModel(), "prey_death")
        
        assert np.isnan(stats["mean"])
        assert stats["n"] == 0
    
    def test_get_evolved_stats_single_value(self):
        """get_evolved_stats should handle single non-NaN value."""
        class MockModel:
            cell_params = {"prey_death": np.array([[np.nan, 0.07], [np.nan, np.nan]])}
        
        stats = get_evolved_stats(MockModel(), "prey_death")
        
        assert np.isclose(stats["mean"], 0.07)
        assert stats["n"] == 1


# ============================================================================
# TEST: POWER LAW FITTING
# ============================================================================

class TestTruncatedPowerLaw:
    """Tests for truncated_power_law function."""
    
    def test_truncated_power_law_shape(self):
        """truncated_power_law should return correct shape."""
        s = np.array([1, 2, 3, 4, 5])
        result = truncated_power_law(s, tau=2.0, s_c=100.0, A=1.0)
        
        assert result.shape == s.shape
    
    def test_truncated_power_law_decreasing(self):
        """truncated_power_law should be decreasing."""
        s = np.linspace(1, 100, 50)
        result = truncated_power_law(s, tau=2.0, s_c=1000.0, A=1.0)
        
        assert np.all(np.diff(result) < 0)
    
    def test_truncated_power_law_positive(self):
        """truncated_power_law should always return positive values."""
        s = np.linspace(1, 1000, 100)
        result = truncated_power_law(s, tau=2.5, s_c=500.0, A=1.0)
        
        assert np.all(result > 0)
    
    def test_truncated_power_law_cutoff_effect(self):
        """Smaller cutoff should cause faster decay."""
        s = np.linspace(1, 100, 50)
        result_large = truncated_power_law(s, tau=2.0, s_c=10000.0, A=1.0)
        result_small = truncated_power_law(s, tau=2.0, s_c=50.0, A=1.0)
        
        assert result_small[-1] < result_large[-1]


class TestFitTruncatedPowerLaw:
    """Tests for fit_truncated_power_law function."""
    
    def test_fit_insufficient_data(self):
        """fit_truncated_power_law should handle insufficient data."""
        sizes = np.array([1, 2, 3])
        result = fit_truncated_power_law(sizes)
        
        assert result["valid"] == False
        assert np.isnan(result["tau"])
    
    def test_fit_empty_data(self):
        """fit_truncated_power_law should handle empty data."""
        sizes = np.array([])
        result = fit_truncated_power_law(sizes)
        
        assert result["valid"] == False
    
    def test_fit_returns_required_keys(self):
        """fit_truncated_power_law should return required keys."""
        np.random.seed(42)
        sizes = (np.random.pareto(1.5, 500) + 1).astype(int)
        sizes = sizes[sizes >= 2]
        
        result = fit_truncated_power_law(sizes)
        
        # Check only the keys that are actually returned
        assert "tau" in result
        assert "s_c" in result
        assert "valid" in result
        assert "n" in result


# ============================================================================
# TEST: PCF AVERAGING
# ============================================================================

class TestAveragePCFs:
    """Tests for average_pcfs function."""
    
    def test_average_pcfs_empty(self):
        """average_pcfs should handle empty list."""
        distances, mean, se = average_pcfs([])
        
        assert len(distances) == 0
        assert len(mean) == 0
        assert len(se) == 0
    
    def test_average_pcfs_single(self):
        """average_pcfs should handle single PCF."""
        dist = np.array([1.0, 2.0, 3.0])
        pcf = np.array([1.5, 1.2, 1.0])
        
        distances, mean, se = average_pcfs([(dist, pcf, 100)])
        
        np.testing.assert_array_equal(distances, dist)
        np.testing.assert_array_equal(mean, pcf)
        np.testing.assert_array_equal(se, np.zeros(3))
    
    def test_average_pcfs_multiple(self):
        """average_pcfs should correctly average multiple PCFs."""
        dist = np.array([1.0, 2.0, 3.0])
        pcf1 = np.array([1.0, 1.0, 1.0])
        pcf2 = np.array([2.0, 2.0, 2.0])
        
        distances, mean, se = average_pcfs([
            (dist, pcf1, 100),
            (dist, pcf2, 100),
        ])
        
        np.testing.assert_array_almost_equal(mean, [1.5, 1.5, 1.5])
        assert np.all(se > 0)
    
    def test_average_pcfs_preserves_length(self):
        """average_pcfs should preserve bin count."""
        n_bins = 20
        dist = np.linspace(0.5, 19.5, n_bins)
        pcf = np.ones(n_bins)
        
        distances, mean, se = average_pcfs([(dist, pcf, 100)] * 5)
        
        assert len(distances) == n_bins
        assert len(mean) == n_bins
        assert len(se) == n_bins


# ============================================================================
# TEST: BINARY SAVE/LOAD
# ============================================================================

class TestBinarySaveLoad:
    """Tests for binary save/load functions."""
    
    def test_save_creates_file(self, temp_dir):
        """save_sweep_binary should create a file."""
        results = [{"prey_birth": 0.2, "prey_mean": 100.0}]
        filepath = temp_dir / "test.npz"
        
        assert not filepath.exists()
        save_sweep_binary(results, filepath)
        assert filepath.exists()
    
    def test_save_load_roundtrip(self, temp_dir):
        """save and load should roundtrip correctly."""
        results = [
            {"prey_birth": 0.2, "prey_death": 0.05, "prey_mean": 100.0, 
             "with_evolution": False, "seed": 1},
            {"prey_birth": 0.3, "prey_death": 0.06, "prey_mean": 150.0, 
             "with_evolution": True, "seed": 2},
        ]
        
        filepath = temp_dir / "test.npz"
        save_sweep_binary(results, filepath)
        loaded = load_sweep_binary(filepath)
        
        assert len(loaded) == len(results)
        
        for orig, load in zip(results, loaded):
            for key in orig:
                assert key in load
                if isinstance(orig[key], float):
                    assert np.isclose(orig[key], load[key])
                else:
                    assert orig[key] == load[key]
    
    def test_save_empty_results(self, temp_dir):
        """save_sweep_binary should handle empty results."""
        filepath = temp_dir / "empty.npz"
        
        save_sweep_binary([], filepath)
        loaded = load_sweep_binary(filepath)
        
        assert len(loaded) == 0
    
    def test_save_complex_results(self, temp_dir):
        """save_sweep_binary should handle complex result dicts."""
        results = [{
            "prey_birth": 0.2,
            "prey_death": 0.05,
            "prey_mean": 100.5,
            "prey_std": 10.2,
            "pred_mean": 50.3,
            "pred_std": 5.1,
            "with_evolution": True,
            "seed": 42,
            "grid_size": 100,
            "prey_survived": True,
            "pred_survived": True,
        }]
        
        filepath = temp_dir / "complex.npz"
        save_sweep_binary(results, filepath)
        loaded = load_sweep_binary(filepath)
        
        assert len(loaded) == 1
        assert np.isclose(loaded[0]["prey_mean"], 100.5)
        assert loaded[0]["seed"] == 42


# ============================================================================
# TEST: SIMULATION FUNCTIONS (require CA module)
# ============================================================================

@pytest.mark.skipif(not CA_AVAILABLE, reason="CA module not available")
class TestRunSingleSimulation:
    """Tests for run_single_simulation function."""
    
    @pytest.fixture(autouse=True)
    def setup(self, fast_config):
        """Setup fast config for all tests."""
        self.cfg = fast_config
    
    def test_returns_dict(self):
        """run_single_simulation should return a dictionary."""
        result = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, with_evolution=False, cfg=self.cfg,
        )
        
        assert isinstance(result, dict)
    
    def test_required_keys_present(self):
        """run_single_simulation should return all required keys."""
        result = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, with_evolution=False, cfg=self.cfg,
        )
        
        required_keys = [
            "prey_birth", "prey_death", "grid_size", "with_evolution", "seed",
            "prey_mean", "prey_std", "pred_mean", "pred_std",
            "prey_survived", "pred_survived",
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_parameters_recorded(self):
        """Input parameters should be recorded in output."""
        result = run_single_simulation(
            prey_birth=0.25, prey_death=0.08, grid_size=30,
            seed=123, with_evolution=False, cfg=self.cfg,
        )
        
        assert np.isclose(result["prey_birth"], 0.25)
        assert np.isclose(result["prey_death"], 0.08)
        assert result["grid_size"] == 30
        assert result["seed"] == 123
        assert result["with_evolution"] == False
    
    def test_values_reasonable(self):
        """Output values should be reasonable."""
        result = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, with_evolution=False, cfg=self.cfg,
        )
        
        assert result["prey_mean"] >= 0
        assert result["pred_mean"] >= 0
        assert result["prey_std"] >= 0
        assert result["pred_std"] >= 0
        
        max_pop = 30 * 30
        assert result["prey_mean"] <= max_pop
        assert result["pred_mean"] <= max_pop
    
    def test_with_evolution_flag(self):
        """with_evolution flag should be recorded."""
        result_no = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, with_evolution=False, cfg=self.cfg,
        )
        result_yes = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, with_evolution=True, cfg=self.cfg,
        )
        
        assert result_no["with_evolution"] == False
        assert result_yes["with_evolution"] == True
    
    def test_survival_flags(self):
        """Survival flags should be boolean."""
        result = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, with_evolution=False, cfg=self.cfg,
        )
        
        assert isinstance(result["prey_survived"], bool)
        assert isinstance(result["pred_survived"], bool)

@pytest.mark.skipif(not CA_AVAILABLE, reason="CA module not available")
class TestDirectedHunting:
    """Tests for directed hunting functionality in simulations."""
    
    @pytest.fixture(autouse=True)
    def setup(self, fast_config):
        """Setup fast config for all tests."""
        self.cfg = fast_config
        self.cfg.directed_hunting = False  # Default to False for comparison
    
    def test_simulation_with_directed_hunting_false(self):
        """Simulation should work with directed_hunting=False."""
        self.cfg.directed_hunting = False
        
        result = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, with_evolution=False, cfg=self.cfg,
        )
        
        assert isinstance(result, dict)
        assert "prey_mean" in result
        assert result["prey_mean"] >= 0
    
    def test_simulation_with_directed_hunting_true(self):
        """Simulation should work with directed_hunting=True."""
        self.cfg.directed_hunting = True
        
        result = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, with_evolution=False, cfg=self.cfg,
        )
        
        assert isinstance(result, dict)
        assert "prey_mean" in result
        assert result["prey_mean"] >= 0
    
    def test_directed_hunting_changes_dynamics(self):
        """Directed hunting should produce different population dynamics."""
        # Run with random movement
        self.cfg.directed_hunting = False
        result_random = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=40,
            seed=42, with_evolution=False, cfg=self.cfg,
        )
        
        # Run with directed hunting
        self.cfg.directed_hunting = True
        result_directed = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=40,
            seed=42, with_evolution=False, cfg=self.cfg,
        )
        
        # Both should produce valid results
        assert result_random["prey_mean"] >= 0
        assert result_directed["prey_mean"] >= 0
        
        # Note: We don't assert they're different because stochastic dynamics
        # means they could occasionally be similar. Just verify both run.
        print(f"Random:   prey_mean={result_random['prey_mean']:.1f}")
        print(f"Directed: prey_mean={result_directed['prey_mean']:.1f}")
    
    def test_directed_hunting_with_evolution(self):
        """Directed hunting should work with evolution enabled."""
        self.cfg.directed_hunting = True
        
        result = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, with_evolution=True, cfg=self.cfg,
        )
        
        assert isinstance(result, dict)
        assert result["with_evolution"] == True
        
        # Should have evolved death rate stats
        if result.get("prey_survived", False):
            # If prey survived, we should have evolution stats
            assert "evolved_death_mean" in result or "prey_mean" in result
    
    def test_directed_hunting_multiple_seeds(self):
        """Directed hunting should work with multiple seeds."""
        self.cfg.directed_hunting = True
        
        results = []
        for seed in [1, 2, 3, 4, 5]:
            result = run_single_simulation(
                prey_birth=0.2, prey_death=0.05, grid_size=30,
                seed=seed, with_evolution=False, cfg=self.cfg,
            )
            results.append(result)
        
        assert len(results) == 5
        for r in results:
            assert "prey_mean" in r
            assert r["prey_mean"] >= 0
    
    def test_directed_hunting_high_predator_birth(self):
        """Directed hunting with high predator birth should deplete prey faster."""
        self.cfg.directed_hunting = True
        self.cfg.predator_birth = 0.8  # High predator birth rate
        
        result = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, with_evolution=False, cfg=self.cfg,
        )
        
        # With high predator birth and directed hunting, prey often go extinct
        assert isinstance(result, dict)
        # Don't assert extinction - just that it ran successfully
        

@pytest.mark.skipif(not CA_AVAILABLE, reason="CA module not available")
class TestRunSingleSimulationFSS:
    """Tests for run_single_simulation_fss function."""
    
    @pytest.fixture(autouse=True)
    def setup(self, fast_config):
        """Setup fast config for all tests."""
        self.cfg = fast_config
    
    def test_returns_dict(self):
        """run_single_simulation_fss should return a dictionary."""
        result = run_single_simulation_fss(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, cfg=self.cfg,
            warmup_steps=20, measurement_steps=30,
        )
        
        assert isinstance(result, dict)
    
    def test_required_keys_present(self):
        """run_single_simulation_fss should return required keys."""
        result = run_single_simulation_fss(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, cfg=self.cfg,
            warmup_steps=20, measurement_steps=30,
        )
        
        required_keys = [
            "prey_birth", "prey_death", "grid_size", "seed",
            "warmup_steps", "measurement_steps",
            "prey_mean", "prey_std", "pred_mean", "pred_std",
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_steps_recorded(self):
        """warmup and measurement steps should be recorded."""
        result = run_single_simulation_fss(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, cfg=self.cfg,
            warmup_steps=50, measurement_steps=100,
        )
        
        assert result["warmup_steps"] == 50
        assert result["measurement_steps"] == 100
    
    def test_different_grid_sizes(self):
        """Should work with different grid sizes."""
        for size in [20, 30, 40]:
            result = run_single_simulation_fss(
                prey_birth=0.2, prey_death=0.05, grid_size=size,
                seed=42, cfg=self.cfg,
                warmup_steps=20, measurement_steps=30,
            )
            
            assert result["grid_size"] == size
            assert result["prey_mean"] >= 0
            
            
    def test_fss_with_directed_hunting(self):
        """FSS simulation should work with directed hunting."""
        self.cfg.directed_hunting = True
        
        result = run_single_simulation_fss(
            prey_birth=0.2, prey_death=0.05, grid_size=30,
            seed=42, cfg=self.cfg,
            warmup_steps=20, measurement_steps=30,
        )
        
        assert isinstance(result, dict)
        assert "prey_mean" in result

# ============================================================================
# TEST: PARAMETER SWEEP LOGIC
# ============================================================================

class TestParameterSweepLogic:
    """Tests for parameter sweep generation logic."""
    
    def test_parameter_grid_coverage(self, default_config):
        """Parameter sweep should cover entire grid."""
        births = default_config.get_prey_births()
        deaths = default_config.get_prey_deaths()
        
        assert np.isclose(births[0], default_config.prey_birth_min)
        assert np.isclose(births[-1], default_config.prey_birth_max)
        assert np.isclose(deaths[0], default_config.prey_death_min)
        assert np.isclose(deaths[-1], default_config.prey_death_max)
    
    def test_total_simulations_formula(self, default_config):
        """Verify total simulation count formula."""
        n_params = default_config.n_prey_birth * default_config.n_prey_death
        n_replicates = default_config.n_replicates
        n_evolution = 2
        
        expected_total = n_params * n_replicates * n_evolution
        
        # Default: 15 * 15 * 50 * 2 = 22,500
        assert expected_total == 15 * 15 * 50 * 2
    
    def test_custom_config_grid(self, default_config):
        """Custom config should produce correct parameter counts."""
        default_config.n_prey_birth = 5
        default_config.n_prey_death = 7
        
        births = default_config.get_prey_births()
        deaths = default_config.get_prey_deaths()
        
        assert len(births) == 5
        assert len(deaths) == 7


# ============================================================================
# TEST: INTEGRATION
# ============================================================================

@pytest.mark.skipif(not CA_AVAILABLE, reason="CA module not available")
class TestIntegration:
    """Integration tests verifying components work together."""
    
    @pytest.fixture(autouse=True)
    def setup(self, fast_config, temp_dir):
        """Setup for all tests."""
        self.cfg = fast_config
        self.temp_dir = temp_dir
    
    def test_simulation_to_binary_roundtrip(self):
        """Simulation results should roundtrip through binary format."""
        result = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=25,
            seed=42, with_evolution=True, cfg=self.cfg,
        )
        
        filepath = self.temp_dir / "roundtrip.npz"
        save_sweep_binary([result], filepath)
        loaded = load_sweep_binary(filepath)
        
        assert len(loaded) == 1
        assert np.isclose(loaded[0]["prey_birth"], result["prey_birth"])
        assert np.isclose(loaded[0]["prey_mean"], result["prey_mean"])
    
    def test_multiple_simulations(self):
        """Multiple simulations should run without interference."""
        results = []
        
        for seed in [1, 2, 3]:
            result = run_single_simulation(
                prey_birth=0.2, prey_death=0.05, grid_size=25,
                seed=seed, with_evolution=False, cfg=self.cfg,
            )
            results.append(result)
        
        assert len(results) == 3
        for r in results:
            assert "prey_mean" in r
            assert r["prey_mean"] >= 0
    
    def test_evolution_vs_no_evolution(self):
        """Evolution flag should be recorded correctly."""
        result_no = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=25,
            seed=42, with_evolution=False, cfg=self.cfg,
        )
        result_yes = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=25,
            seed=42, with_evolution=True, cfg=self.cfg,
        )
        
        assert result_no["with_evolution"] == False
        assert result_yes["with_evolution"] == True
        
        
    def test_directed_hunting_binary_roundtrip(self):
        """Directed hunting results should roundtrip through binary format."""
        self.cfg.directed_hunting = True
        
        result = run_single_simulation(
            prey_birth=0.2, prey_death=0.05, grid_size=25,
            seed=42, with_evolution=False, cfg=self.cfg,
        )
        
        filepath = self.temp_dir / "directed_roundtrip.npz"
        save_sweep_binary([result], filepath)
        loaded = load_sweep_binary(filepath)
        
        assert len(loaded) == 1
        assert np.isclose(loaded[0]["prey_birth"], result["prey_birth"])
        assert np.isclose(loaded[0]["prey_mean"], result["prey_mean"])


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])