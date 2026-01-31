"""
Shared pytest fixtures for Predator-Prey CA test suite.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# Minimal Config for Testing (avoids importing full config module)
# =============================================================================


@dataclass
class MinimalConfig:
    """Minimal configuration for fast test simulations."""

    grid_size: int = 10
    densities: Tuple[float, float] = (0.3, 0.15)
    grid_sizes: Tuple[int, ...] = (5, 10)
    prey_birth: float = 0.2
    prey_death: float = 0.05
    predator_birth: float = 0.8
    predator_death: float = 0.05
    critical_prey_birth: float = 0.2
    critical_prey_death: float = 0.097
    prey_death_range: Tuple[float, float] = (0.05, 0.15)
    n_prey_death: int = 3
    n_replicates: int = 2
    warmup_steps: int = 5
    measurement_steps: int = 10
    evolve_sd: float = 0.05
    evolve_min: float = 0.01
    evolve_max: float = 0.15
    directed_hunting: bool = False
    save_timeseries: bool = False
    timeseries_subsample: int = 2
    collect_pcf: bool = False
    pcf_sample_rate: float = 0.0
    pcf_max_distance: float = 5.0
    pcf_n_bins: int = 10
    min_density_for_analysis: float = 0.01
    n_jobs: int = 1

    def get_prey_deaths(self) -> np.ndarray:
        return np.linspace(
            self.prey_death_range[0], self.prey_death_range[1], self.n_prey_death
        )

    def get_warmup_steps(self, L: int) -> int:
        return self.warmup_steps

    def get_measurement_steps(self, L: int) -> int:
        return self.measurement_steps


# =============================================================================
# Grid Fixtures
# =============================================================================


@pytest.fixture
def empty_grid_10x10():
    """10x10 grid with no species."""
    return np.zeros((10, 10), dtype=np.int32)


@pytest.fixture
def prey_only_grid_10x10():
    """10x10 grid with only prey (species 1) in a known pattern."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2:5, 2:5] = 1  # 3x3 block of prey = 9 cells
    return grid


@pytest.fixture
def predator_only_grid_10x10():
    """10x10 grid with only predators (species 2)."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[0, 0] = 2
    grid[0, 9] = 2
    grid[9, 0] = 2
    grid[9, 9] = 2  # 4 predators in corners
    return grid


@pytest.fixture
def mixed_grid_10x10():
    """10x10 grid with both prey and predators."""
    grid = np.zeros((10, 10), dtype=np.int32)
    # Prey cluster
    grid[1:4, 1:4] = 1  # 9 prey
    # Predator cluster
    grid[6:8, 6:8] = 2  # 4 predators
    return grid


@pytest.fixture
def single_cluster_grid():
    """Grid with exactly one connected cluster of prey."""
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[1, 1] = 1
    grid[1, 2] = 1
    grid[2, 1] = 1
    grid[2, 2] = 1  # 2x2 block = 4 connected cells
    return grid


@pytest.fixture
def two_cluster_grid():
    """Grid with two separate prey clusters (no periodic connection)."""
    grid = np.zeros((10, 10), dtype=np.int32)
    # Cluster 1: top-left corner
    grid[0, 0] = 1
    grid[0, 1] = 1
    grid[1, 0] = 1  # 3 cells
    # Cluster 2: center (far enough to avoid periodic Moore connection)
    grid[4, 4] = 1
    grid[4, 5] = 1
    grid[5, 4] = 1
    grid[5, 5] = 1  # 4 cells
    return grid


@pytest.fixture
def periodic_cluster_grid():
    """Grid where prey connect via periodic boundary."""
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[0, 0] = 1  # Top-left
    grid[4, 0] = 1  # Bottom-left (connects to top-left via periodic)
    grid[0, 4] = 1  # Top-right (connects to top-left via periodic)
    return grid


@pytest.fixture
def checkerboard_grid():
    """Alternating pattern - many small clusters."""
    grid = np.zeros((6, 6), dtype=np.int32)
    for i in range(6):
        for j in range(6):
            if (i + j) % 2 == 0:
                grid[i, j] = 1
    return grid


# =============================================================================
# Config Fixtures
# =============================================================================


@pytest.fixture
def minimal_config():
    """Minimal config for fast test runs."""
    return MinimalConfig()


@pytest.fixture
def minimal_config_with_pcf():
    """Config with PCF collection enabled."""
    return MinimalConfig(collect_pcf=True, pcf_sample_rate=1.0)


@pytest.fixture
def minimal_config_with_timeseries():
    """Config with time series collection enabled."""
    return MinimalConfig(save_timeseries=True)


@pytest.fixture
def minimal_config_directed():
    """Config with directed hunting enabled."""
    return MinimalConfig(directed_hunting=True)


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def pp_model_small():
    """Small PP model for quick tests."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.CA import PP

    return PP(
        rows=10,
        cols=10,
        densities=(0.3, 0.15),
        neighborhood="moore",
        seed=42,
        directed_hunting=False,
    )


@pytest.fixture
def pp_model_with_evolution():
    """PP model with evolution enabled."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.CA import PP

    model = PP(
        rows=10,
        cols=10,
        densities=(0.3, 0.15),
        neighborhood="moore",
        seed=42,
    )
    model.evolve("prey_death", sd=0.05, min_val=0.01, max_val=0.15)
    return model


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "test_results"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_results():
    """Sample simulation results for I/O testing."""
    return [
        {
            "prey_birth": 0.2,
            "prey_death": 0.05,
            "predator_birth": 0.8,
            "predator_death": 0.1,
            "grid_size": 10,
            "seed": 42,
            "prey_mean": 25.5,
            "prey_std": 3.2,
            "pred_mean": 12.1,
            "pred_std": 2.5,
            "prey_survived": True,
            "pred_survived": True,
            "prey_cluster_sizes": [10, 5, 3],
            "pred_cluster_sizes": [8, 4],
        },
        {
            "prey_birth": 0.2,
            "prey_death": 0.10,
            "predator_birth": 0.8,
            "predator_death": 0.1,
            "grid_size": 10,
            "seed": 43,
            "prey_mean": 20.0,
            "prey_std": 4.0,
            "pred_mean": 15.0,
            "pred_std": 3.0,
            "prey_survived": True,
            "pred_survived": True,
            "prey_cluster_sizes": [12, 8],
            "pred_cluster_sizes": [10, 5],
        },
    ]