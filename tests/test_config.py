"""
Tests for configuration module.

Covers:
- Config dataclass defaults and validation
- Phase config retrieval
- Helper methods
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import (
    Config,
    get_phase_config,
    PHASE_CONFIGS,
    PHASE1_CONFIG,
    PHASE2_CONFIG,
    PHASE3_CONFIG,
    PHASE4_CONFIG,
    PHASE5_CONFIG,
)


# =============================================================================
# Config Defaults Tests
# =============================================================================


class TestConfigDefaults:
    """Tests for Config default values."""

    def test_default_grid_size(self):
        """Default grid size should be 1000."""
        cfg = Config()
        assert cfg.grid_size == 1000

    def test_default_densities(self):
        """Default densities should be (0.30, 0.15)."""
        cfg = Config()
        assert cfg.densities == (0.30, 0.15)

    def test_default_species_parameters(self):
        """Default species parameters should be set."""
        cfg = Config()
        assert cfg.prey_birth == 0.2
        assert cfg.prey_death == 0.05
        assert cfg.predator_birth == 0.8
        assert cfg.predator_death == 0.05

    def test_default_replicates(self):
        """Default replicates should be 15."""
        cfg = Config()
        assert cfg.n_replicates == 15

    def test_default_parallelization(self):
        """Default n_jobs should be -1 (all cores)."""
        cfg = Config()
        assert cfg.n_jobs == -1


class TestConfigCustomization:
    """Tests for Config customization."""

    def test_override_grid_size(self):
        """Should accept custom grid size."""
        cfg = Config(grid_size=500)
        assert cfg.grid_size == 500

    def test_override_multiple_params(self):
        """Should accept multiple overrides."""
        cfg = Config(
            grid_size=200,
            n_replicates=5,
            warmup_steps=100,
            directed_hunting=True,
        )
        assert cfg.grid_size == 200
        assert cfg.n_replicates == 5
        assert cfg.warmup_steps == 100
        assert cfg.directed_hunting is True

    def test_override_preserves_other_defaults(self):
        """Overriding one param should not affect others."""
        cfg = Config(grid_size=500)
        assert cfg.prey_birth == 0.2  # Still default
        assert cfg.n_replicates == 15  # Still default


# =============================================================================
# Config Helper Methods Tests
# =============================================================================


class TestConfigHelpers:
    """Tests for Config helper methods."""

    def test_get_prey_deaths_returns_array(self):
        """get_prey_deaths should return numpy array."""
        cfg = Config(prey_death_range=(0.0, 0.1), n_prey_death=5)
        deaths = cfg.get_prey_deaths()

        assert isinstance(deaths, np.ndarray)
        assert len(deaths) == 5

    def test_get_prey_deaths_correct_range(self):
        """get_prey_deaths should cover specified range."""
        cfg = Config(prey_death_range=(0.05, 0.15), n_prey_death=11)
        deaths = cfg.get_prey_deaths()

        assert deaths[0] == pytest.approx(0.05)
        assert deaths[-1] == pytest.approx(0.15)

    def test_get_warmup_steps_returns_configured_value(self):
        """get_warmup_steps should return warmup_steps."""
        cfg = Config(warmup_steps=500)
        assert cfg.get_warmup_steps(L=100) == 500

    def test_get_measurement_steps_returns_configured_value(self):
        """get_measurement_steps should return measurement_steps."""
        cfg = Config(measurement_steps=1000)
        assert cfg.get_measurement_steps(L=100) == 1000

    def test_estimate_runtime_returns_string(self):
        """estimate_runtime should return formatted string."""
        cfg = Config(grid_size=100, n_prey_death=5, n_replicates=2)
        estimate = cfg.estimate_runtime(n_cores=4)

        assert isinstance(estimate, str)
        assert "sims" in estimate
        assert "cores" in estimate


# =============================================================================
# Phase Config Tests
# =============================================================================


class TestPhaseConfigs:
    """Tests for pre-defined phase configurations."""

    def test_all_phases_exist(self):
        """All 5 phases should have configs."""
        assert 1 in PHASE_CONFIGS
        assert 2 in PHASE_CONFIGS
        assert 3 in PHASE_CONFIGS
        assert 4 in PHASE_CONFIGS
        assert 5 in PHASE_CONFIGS

    def test_get_phase_config_returns_correct_config(self):
        """get_phase_config should return correct instance."""
        assert get_phase_config(1) is PHASE1_CONFIG
        assert get_phase_config(2) is PHASE2_CONFIG
        assert get_phase_config(3) is PHASE3_CONFIG
        assert get_phase_config(4) is PHASE4_CONFIG
        assert get_phase_config(5) is PHASE5_CONFIG

    def test_get_phase_config_invalid_raises(self):
        """get_phase_config should raise for invalid phase."""
        with pytest.raises(ValueError, match="Unknown phase"):
            get_phase_config(99)

        with pytest.raises(ValueError, match="Unknown phase"):
            get_phase_config(0)

    def test_phase1_config_values(self):
        """Phase 1 config should have expected values."""
        cfg = PHASE1_CONFIG
        assert cfg.grid_size == 1000
        assert cfg.collect_pcf is False
        assert cfg.directed_hunting is False

    def test_phase2_config_evolution_settings(self):
        """Phase 2 config should have evolution settings."""
        cfg = PHASE2_CONFIG
        assert cfg.evolve_sd > 0
        assert cfg.evolve_min >= 0
        assert cfg.evolve_max > cfg.evolve_min

    def test_phase3_config_has_grid_sizes(self):
        """Phase 3 config should have multiple grid sizes."""
        cfg = PHASE3_CONFIG
        assert len(cfg.grid_sizes) > 1
        assert cfg.collect_pcf is True

    def test_phase4_no_directed_hunting(self):
        """Phase 4 should have directed_hunting=False."""
        cfg = PHASE4_CONFIG
        assert cfg.directed_hunting is False

    def test_phase5_directed_hunting_enabled(self):
        """Phase 5 should have directed_hunting=True."""
        cfg = PHASE5_CONFIG
        assert cfg.directed_hunting is True


# =============================================================================
# Config Consistency Tests
# =============================================================================


class TestConfigConsistency:
    """Tests for configuration consistency and validity."""

    def test_densities_sum_valid(self):
        """Default densities should sum to <= 1.0."""
        cfg = Config()
        assert sum(cfg.densities) <= 1.0

    def test_prey_death_range_valid(self):
        """prey_death_range should have min < max."""
        cfg = Config()
        assert cfg.prey_death_range[0] < cfg.prey_death_range[1]

    def test_evolve_bounds_valid(self):
        """evolve_min should be less than evolve_max."""
        cfg = Config()
        assert cfg.evolve_min < cfg.evolve_max

    def test_pcf_bins_positive(self):
        """PCF bins should be positive."""
        cfg = Config()
        assert cfg.pcf_n_bins > 0

    def test_all_phase_configs_valid_densities(self):
        """All phase configs should have valid densities."""
        for phase, cfg in PHASE_CONFIGS.items():
            assert sum(cfg.densities) <= 1.0, f"Phase {phase} has invalid densities"

    def test_all_phase_configs_positive_steps(self):
        """All phase configs should have positive step counts."""
        for phase, cfg in PHASE_CONFIGS.items():
            assert cfg.warmup_steps > 0, f"Phase {phase} has non-positive warmup"
            assert cfg.measurement_steps > 0, f"Phase {phase} has non-positive measurement"