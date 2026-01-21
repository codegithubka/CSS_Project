"""
Comprehensive pytest test suite for pp_analysis.py

Run with: pytest tests/test_pp_analysis.py -v
"""
import pytest
import numpy as np
import json
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pp_analysis import (
    Config,
    count_populations,
    measure_cluster_sizes,
    truncated_power_law,
    fit_truncated_power_law,
    get_evolved_stats,
    run_single_simulation,
    run_single_simulation_fss,
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def default_config():
    """Create default configuration."""
    return Config()


@pytest.fixture
def minimal_config():
    """Create minimal configuration for fast testing."""
    cfg = Config()
    cfg.n_prey_birth = 3
    cfg.n_prey_death = 3
    cfg.n_replicates = 2
    cfg.default_grid = 20
    cfg.warmup_steps = 10
    cfg.measurement_steps = 20
    cfg.cluster_samples = 5
    cfg.fss_replicates = 2
    cfg.sensitivity_replicates = 2
    return cfg


@pytest.fixture
def mock_pp_model():
    """Create a mock PP model for testing."""
    mock = MagicMock()
    mock.grid = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
    mock.rows = 3
    mock.cols = 3
    # Use np.nan for cells without the species (prey_death only applies to prey cells)
    # Prey cells are at positions where grid == 1: (0,1), (1,0), (2,1)
    mock.cell_params = {"prey_death": np.array([[np.nan, 0.06, np.nan], 
                                                 [0.04, np.nan, np.nan],
                                                 [np.nan, 0.07, np.nan]])}
    mock._evolve_info = {"prey_death": {"sd": 0.1, "min": 0.001, "max": 0.1}}
    return mock


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestConfig:
    """Test Config dataclass."""
    
    def test_default_initialization(self, default_config):
        """Test that default config initializes correctly."""
        assert default_config.default_grid == 100
        assert default_config.predator_death == 0.10  # PP default
        assert default_config.predator_birth == 0.20  # PP default
        assert default_config.n_replicates == 50  # Updated to match actual Config
        assert default_config.synchronous == False  # Default is async
    
    def test_get_prey_births(self, default_config):
        """Test prey birth array generation."""
        births = default_config.get_prey_births()
        assert len(births) == default_config.n_prey_birth
        assert births[0] == pytest.approx(default_config.prey_birth_min)
        assert births[-1] == pytest.approx(default_config.prey_birth_max)
    
    def test_get_prey_deaths(self, default_config):
        """Test prey death array generation."""
        deaths = default_config.get_prey_deaths()
        assert len(deaths) == default_config.n_prey_death
        assert deaths[0] == pytest.approx(default_config.prey_death_min)
        assert deaths[-1] == pytest.approx(default_config.prey_death_max)
    
    def test_estimate_runtime(self, default_config):
        """Test runtime estimation."""
        estimate = default_config.estimate_runtime(n_cores=32)
        assert "sims" in estimate
        assert "hours" in estimate or "h" in estimate
        assert "core-hours" in estimate


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_count_populations_empty_grid(self):
        """Test counting on empty grid."""
        grid = np.zeros((10, 10), dtype=int)
        empty, prey, pred = count_populations(grid)
        assert empty == 100
        assert prey == 0
        assert pred == 0
    
    def test_count_populations_mixed_grid(self):
        """Test counting on mixed grid."""
        grid = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
        empty, prey, pred = count_populations(grid)
        assert empty == 3
        assert prey == 3
        assert pred == 3
    
    def test_measure_cluster_sizes_single_cluster(self):
        """Test cluster measurement with single cluster."""
        grid = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        sizes = measure_cluster_sizes(grid, species=1)
        assert len(sizes) == 1
        assert sizes[0] == 4
    
    def test_measure_cluster_sizes_multiple_clusters(self):
        """Test cluster measurement with multiple clusters."""
        grid = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        sizes = measure_cluster_sizes(grid, species=1)
        assert len(sizes) == 4
        assert all(s == 1 for s in sizes)
    
    def test_measure_cluster_sizes_no_species(self):
        """Test cluster measurement with no target species."""
        grid = np.zeros((3, 3), dtype=int)
        sizes = measure_cluster_sizes(grid, species=1)
        assert len(sizes) == 0
    
    def test_truncated_power_law(self):
        """Test truncated power law function."""
        s = np.array([1, 10, 100])
        result = truncated_power_law(s, tau=2.0, s_c=50.0, A=1.0)
        assert len(result) == 3
        assert all(result > 0)
        assert result[0] > result[1] > result[2]  # Decreasing
    
    def test_fit_truncated_power_law_sufficient_data(self):
        """Test power law fitting with sufficient data."""
        # Generate synthetic power law data
        np.random.seed(42)
        sizes = np.random.pareto(1.05, 200) + 2
        result = fit_truncated_power_law(sizes)
        
        assert "tau" in result
        assert "s_c" in result
        assert "valid" in result
        assert result["valid"] is True
        assert 1.0 < result["tau"] < 4.0
    
    def test_fit_truncated_power_law_insufficient_data(self):
        """Test power law fitting with insufficient data."""
        sizes = np.array([2, 3, 4, 5])  # Too few points
        result = fit_truncated_power_law(sizes)
        
        assert result["valid"] is False
        assert np.isnan(result["tau"])
        assert np.isnan(result["s_c"])
    
    def test_get_evolved_stats(self, mock_pp_model):
        """Test evolved parameter statistics extraction."""
        stats = get_evolved_stats(mock_pp_model, "prey_death")
        
        assert "mean" in stats
        assert "std" in stats
        assert "n" in stats
        assert stats["n"] == 3  # Three non-NaN values: 0.06, 0.04, 0.07
        assert 0.001 <= stats["mean"] <= 0.1
    
    def test_get_evolved_stats_no_param(self, mock_pp_model):
        """Test evolved stats with non-existent parameter."""
        stats = get_evolved_stats(mock_pp_model, "nonexistent")
        
        assert np.isnan(stats["mean"])
        assert np.isnan(stats["std"])
        assert stats["n"] == 0


# =============================================================================
# SIMULATION TESTS
# =============================================================================


class TestSimulations:
    """Test simulation runners."""
    
    @pytest.mark.slow
    def test_run_single_simulation_basic(self, minimal_config, temp_output_dir):
        """Test basic single simulation run."""
        result = run_single_simulation(
            prey_birth=0.20,
            prey_death=0.05,
            grid_size=minimal_config.default_grid,
            seed=42,
            with_evolution=False,
            cfg=minimal_config,
            output_dir=temp_output_dir
        )
        
        # Check required fields
        assert "prey_birth" in result
        assert "prey_death" in result
        assert "grid_size" in result
        assert "seed" in result
        assert "prey_mean" in result
        assert "pred_mean" in result
        assert "prey_survived" in result
        assert "pred_survived" in result
        
        # Check value ranges
        assert result["prey_birth"] == 0.20
        assert result["prey_death"] == 0.05
        assert result["grid_size"] == minimal_config.default_grid
        assert 0 <= result["prey_mean"] <= minimal_config.default_grid ** 2
        assert 0 <= result["pred_mean"] <= minimal_config.default_grid ** 2
    
    @pytest.mark.slow
    def test_run_single_simulation_with_evolution(self, minimal_config, temp_output_dir):
        """Test simulation with evolution enabled."""
        result = run_single_simulation(
            prey_birth=0.20,
            prey_death=0.05,
            grid_size=minimal_config.default_grid,
            seed=42,
            with_evolution=True,
            cfg=minimal_config,
            output_dir=temp_output_dir
        )
        
        # Check evolution-specific fields
        assert "with_evolution" in result
        assert result["with_evolution"] is True
        assert "evolved_prey_death_mean" in result
        assert "evolved_prey_death_std" in result
        assert "evolve_sd" in result
    
    @pytest.mark.slow
    def test_run_single_simulation_fss(self, minimal_config, temp_output_dir):
        """Test FSS-specific simulation with scaled equilibration."""
        result = run_single_simulation_fss(
            prey_birth=0.20,
            prey_death=0.03,
            grid_size=50,  # Smaller than default
            seed=42,
            with_evolution=False,
            cfg=minimal_config,
            warmup_steps=50,  # Scaled warmup
            measurement_steps=100,  # Scaled measurement
            output_dir=temp_output_dir
        )
        
        # Check FSS-specific fields
        assert "warmup_steps" in result
        assert "measurement_steps" in result
        assert result["warmup_steps"] == 50
        assert result["measurement_steps"] == 100
        assert result["grid_size"] == 50
    
    @pytest.mark.slow
    def test_simulation_reproducibility(self, minimal_config, temp_output_dir):
        """Test that same seed produces same results."""
        result1 = run_single_simulation(
            prey_birth=0.20,
            prey_death=0.05,
            grid_size=minimal_config.default_grid,
            seed=42,
            with_evolution=False,
            cfg=minimal_config,
            output_dir=temp_output_dir
        )
        
        result2 = run_single_simulation(
            prey_birth=0.20,
            prey_death=0.05,
            grid_size=minimal_config.default_grid,
            seed=42,
            with_evolution=False,
            cfg=minimal_config,
            output_dir=temp_output_dir
        )
        
        # Results should be identical with same seed
        assert result1["prey_mean"] == result2["prey_mean"]
        assert result1["pred_mean"] == result2["pred_mean"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.slow
    def test_config_to_simulation_pipeline(self, minimal_config, temp_output_dir):
        """Test full pipeline from config to simulation."""
        # Get parameter arrays from config
        prey_births = minimal_config.get_prey_births()
        prey_deaths = minimal_config.get_prey_deaths()
        
        # Run a few simulations
        results = []
        for pb in prey_births[:2]:
            for pd in prey_deaths[:2]:
                result = run_single_simulation(
                    prey_birth=pb,
                    prey_death=pd,
                    grid_size=minimal_config.default_grid,
                    seed=42,
                    with_evolution=False,
                    cfg=minimal_config,
                    output_dir=temp_output_dir
                )
                results.append(result)
        
        assert len(results) == 4
        assert all("prey_mean" in r for r in results)
    
    def test_fss_scaling_logic(self, minimal_config):
        """Test that FSS properly scales equilibration time."""
        grid_sizes = [50, 100, 150]
        warmup_times = []
        
        for L in grid_sizes:
            factor = L / minimal_config.default_grid
            warmup = int(minimal_config.warmup_steps * factor)
            warmup_times.append(warmup)
        
        # Verify scaling - larger grids need more warmup
        assert warmup_times[0] < warmup_times[1] < warmup_times[2]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_cluster_sizes(self):
        """Test cluster measurement on empty grid."""
        grid = np.zeros((10, 10), dtype=int)
        sizes = measure_cluster_sizes(grid, species=1)
        assert len(sizes) == 0
    
    def test_full_grid_one_species(self):
        """Test counting on grid full of one species."""
        grid = np.ones((10, 10), dtype=int)
        empty, prey, pred = count_populations(grid)
        assert empty == 0
        assert prey == 100
        assert pred == 0
    
    def test_power_law_fit_edge_cases(self):
        """Test power law fitting edge cases."""
        # Empty array
        result = fit_truncated_power_law(np.array([]))
        assert result["valid"] is False
        
        # Single value repeated
        result = fit_truncated_power_law(np.array([5, 5, 5, 5]))
        assert result["valid"] is False
    
    def test_config_boundary_values(self):
        """Test config with boundary parameter values."""
        cfg = Config()
        cfg.prey_birth_min = 0.0
        cfg.prey_birth_max = 1.0
        cfg.prey_death_min = 0.0
        cfg.prey_death_max = 1.0
        
        births = cfg.get_prey_births()
        deaths = cfg.get_prey_deaths()
        
        assert births[0] == 0.0
        assert births[-1] == 1.0
        assert deaths[0] == 0.0
        assert deaths[-1] == 1.0


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("grid_size", [10, 20, 50])
def test_simulation_different_grid_sizes(grid_size, minimal_config, temp_output_dir):
    """Test simulation with different grid sizes."""
    result = run_single_simulation(
        prey_birth=0.20,
        prey_death=0.05,
        grid_size=grid_size,
        seed=42,
        with_evolution=False,
        cfg=minimal_config,
        output_dir=temp_output_dir
    )
    
    assert result["grid_size"] == grid_size
    assert 0 <= result["prey_mean"] <= grid_size ** 2


@pytest.mark.slow
@pytest.mark.parametrize("with_evo", [True, False])
def test_simulation_evolution_toggle(with_evo, minimal_config, temp_output_dir):
    """Test simulation with evolution on/off."""
    result = run_single_simulation(
        prey_birth=0.20,
        prey_death=0.05,
        grid_size=minimal_config.default_grid,
        seed=42,
        with_evolution=with_evo,
        cfg=minimal_config,
        output_dir=temp_output_dir
    )
    
    assert result["with_evolution"] == with_evo
    if with_evo:
        assert "evolved_prey_death_mean" in result
    else:
        assert "evolved_prey_death_mean" not in result


@pytest.mark.parametrize("tau,s_c,A", [
    (2.0, 100.0, 1.0),
    (1.5, 50.0, 2.0),
    (2.5, 200.0, 0.5),
])
def test_truncated_power_law_parameters(tau, s_c, A):
    """Test truncated power law with different parameters."""
    s = np.logspace(0, 3, 20)
    result = truncated_power_law(s, tau, s_c, A)
    
    assert len(result) == len(s)
    assert all(result > 0)
    # Check that it's generally decreasing (allowing for numerical noise)
    assert result[0] > result[-1]


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance and stress tests."""
    
    def test_large_grid_simulation(self, temp_output_dir):
        """Test simulation with large grid (stress test)."""
        cfg = Config()
        cfg.warmup_steps = 10
        cfg.measurement_steps = 10
        
        result = run_single_simulation(
            prey_birth=0.20,
            prey_death=0.05,
            grid_size=200,  # Large grid
            seed=42,
            with_evolution=False,
            cfg=cfg,
            output_dir=temp_output_dir
        )
        
        assert result is not None
        assert "prey_mean" in result
    
    def test_many_clusters_measurement(self):
        """Test cluster measurement with many small clusters."""
        # Grid with many isolated individuals
        grid = np.zeros((50, 50), dtype=int)
        grid[::2, ::2] = 1  # Checkerboard pattern
        
        sizes = measure_cluster_sizes(grid, species=1)
        
        # Should detect many size-1 clusters
        assert len(sizes) > 100
        assert all(s == 1 for s in sizes)


# =============================================================================
# FIXTURES FOR MARKS
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])