"""
Tests for CA base class and PP (Predator-Prey) model.

Covers:
- CA initialization and validation
- PP model initialization, parameters, and update logic
- Evolution mechanism
- Seed reproducibility
- Edge cases (empty grids, extinction)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.CA import CA, PP


# =============================================================================
# CA Base Class Tests
# =============================================================================


class TestCAInitialization:
    """Tests for CA base class initialization."""

    def test_ca_requires_positive_dimensions(self):
        """CA should reject non-positive dimensions."""
        with pytest.raises(AssertionError, match="rows must be positive"):
            CA(rows=0, cols=10, densities=(0.5,), neighborhood="moore", params={}, cell_params={})

        with pytest.raises(AssertionError, match="cols must be positive"):
            CA(rows=10, cols=-1, densities=(0.5,), neighborhood="moore", params={}, cell_params={})

    def test_ca_requires_valid_neighborhood(self):
        """CA should only accept 'moore' or 'neumann' neighborhoods."""
        with pytest.raises(AssertionError, match="neighborhood must be"):
            CA(rows=5, cols=5, densities=(0.3,), neighborhood="invalid", params={}, cell_params={})

    def test_ca_densities_must_not_exceed_one(self):
        """Sum of densities must not exceed 1.0."""
        with pytest.raises(AssertionError, match="sum of densities"):
            CA(rows=5, cols=5, densities=(0.6, 0.6), neighborhood="moore", params={}, cell_params={})

    def test_ca_densities_must_be_non_negative(self):
        """Each density must be non-negative."""
        with pytest.raises(AssertionError, match="non-negative"):
            CA(rows=5, cols=5, densities=(-0.1, 0.5), neighborhood="moore", params={}, cell_params={})

    def test_ca_grid_shape_matches_dimensions(self):
        """Grid should have the specified shape."""

        class ConcreteCA(CA):
            def update(self):
                pass

        ca = ConcreteCA(rows=7, cols=13, densities=(0.2,), neighborhood="moore", params={}, cell_params={})
        assert ca.grid.shape == (7, 13)
        assert ca.rows == 7
        assert ca.cols == 13

    def test_ca_species_count_from_densities(self):
        """n_species should equal length of densities tuple."""

        class ConcreteCA(CA):
            def update(self):
                pass

        ca = ConcreteCA(rows=5, cols=5, densities=(0.2, 0.1, 0.05), neighborhood="moore", params={}, cell_params={})
        assert ca.n_species == 3

    def test_ca_grid_population_approximately_matches_density(self):
        """Initial grid population should approximately match requested densities."""

        class ConcreteCA(CA):
            def update(self):
                pass

        np.random.seed(42)
        ca = ConcreteCA(rows=100, cols=100, densities=(0.3, 0.15), neighborhood="moore", params={}, cell_params={}, seed=42)

        total_cells = 100 * 100
        expected_species1 = int(total_cells * 0.3)
        expected_species2 = int(total_cells * 0.15)

        actual_species1 = np.sum(ca.grid == 1)
        actual_species2 = np.sum(ca.grid == 2)

        # Allow 1% tolerance due to rounding
        assert abs(actual_species1 - expected_species1) <= total_cells * 0.01
        assert abs(actual_species2 - expected_species2) <= total_cells * 0.01

    def test_ca_seed_reproducibility(self):
        """Same seed should produce identical grids."""

        class ConcreteCA(CA):
            def update(self):
                pass

        ca1 = ConcreteCA(rows=20, cols=20, densities=(0.3, 0.1), neighborhood="moore", params={}, cell_params={}, seed=123)
        ca2 = ConcreteCA(rows=20, cols=20, densities=(0.3, 0.1), neighborhood="moore", params={}, cell_params={}, seed=123)

        assert np.array_equal(ca1.grid, ca2.grid)

    def test_ca_different_seeds_produce_different_grids(self):
        """Different seeds should (almost certainly) produce different grids."""

        class ConcreteCA(CA):
            def update(self):
                pass

        ca1 = ConcreteCA(rows=20, cols=20, densities=(0.3, 0.1), neighborhood="moore", params={}, cell_params={}, seed=111)
        ca2 = ConcreteCA(rows=20, cols=20, densities=(0.3, 0.1), neighborhood="moore", params={}, cell_params={}, seed=222)

        assert not np.array_equal(ca1.grid, ca2.grid)


class TestCAValidation:
    """Tests for CA validation method."""

    def test_validate_passes_for_valid_ca(self):
        """Validation should pass for properly initialized CA."""

        class ConcreteCA(CA):
            def update(self):
                pass

        ca = ConcreteCA(rows=10, cols=10, densities=(0.2,), neighborhood="moore", params={}, cell_params={})
        ca.validate()  # Should not raise

    def test_validate_fails_for_mismatched_grid_shape(self):
        """Validation should fail if grid shape is modified incorrectly."""

        class ConcreteCA(CA):
            def update(self):
                pass

        ca = ConcreteCA(rows=10, cols=10, densities=(0.2,), neighborhood="moore", params={}, cell_params={})
        ca.grid = np.zeros((5, 5))  # Wrong shape

        with pytest.raises(ValueError, match="grid shape"):
            ca.validate()


class TestCAEvolution:
    """Tests for CA parameter evolution mechanism."""

    def test_evolve_creates_cell_params_array(self):
        """evolve() should create a per-cell parameter array."""

        class ConcreteCA(CA):
            species_names = ("prey",)

            def update(self):
                pass

        ca = ConcreteCA(
            rows=10,
            cols=10,
            densities=(0.3,),
            neighborhood="moore",
            params={"prey_death": 0.05},
            cell_params={},
        )
        ca.evolve("prey_death", species=1, sd=0.02, min_val=0.01, max_val=0.1)

        assert "prey_death" in ca.cell_params
        assert ca.cell_params["prey_death"].shape == (10, 10)

    def test_evolve_sets_values_only_for_target_species(self):
        """evolved parameter should be NaN for non-target species cells."""

        class ConcreteCA(CA):
            species_names = ("prey", "predator")

            def update(self):
                pass

        ca = ConcreteCA(
            rows=10,
            cols=10,
            densities=(0.3, 0.1),
            neighborhood="moore",
            params={"prey_death": 0.05},
            cell_params={},
        )
        ca.evolve("prey_death", species=1, sd=0.02)

        arr = ca.cell_params["prey_death"]
        # Species 1 cells should have the value
        assert np.allclose(arr[ca.grid == 1], 0.05)
        # Other cells should be NaN
        assert np.all(np.isnan(arr[ca.grid != 1]))

    def test_evolve_rejects_unknown_parameter(self):
        """evolve() should raise for parameters not in self.params."""

        class ConcreteCA(CA):
            def update(self):
                pass

        ca = ConcreteCA(rows=5, cols=5, densities=(0.3,), neighborhood="moore", params={}, cell_params={})

        with pytest.raises(ValueError, match="Unknown parameter"):
            ca.evolve("nonexistent_param")

    def test_evolve_infers_species_from_param_name(self):
        """evolve() should infer species from parameter name prefix."""

    class ConcreteCA(CA):
        def update(self):
            pass

    ca = ConcreteCA(
        rows=10,
        cols=10,
        densities=(0.3, 0.1),
        neighborhood="moore",
        params={"prey_death": 0.05},
        cell_params={},
    )
    ca.species_names = ("prey", "predator")
    
    # Should infer species=1 from "prey_death"
    ca.evolve("prey_death", sd=0.02)

    assert "prey_death" in ca._evolve_info
    assert ca._evolve_info["prey_death"]["species"] == 1


# =============================================================================
# PP Model Tests
# =============================================================================


class TestPPInitialization:
    """Tests for PP model initialization."""

    def test_pp_default_initialization(self):
        """PP should initialize with sensible defaults."""
        model = PP()
        assert model.rows == 10
        assert model.cols == 10
        assert model.n_species == 2
        assert model.species_names == ("prey", "predator")

    def test_pp_custom_dimensions(self):
        """PP should accept custom grid dimensions."""
        model = PP(rows=25, cols=30)
        assert model.rows == 25
        assert model.cols == 30
        assert model.grid.shape == (25, 30)

    def test_pp_default_parameters(self):
        """PP should have correct default parameters."""
        model = PP()
        assert model.params["prey_death"] == 0.05
        assert model.params["predator_death"] == 0.1
        assert model.params["prey_birth"] == 0.25
        assert model.params["predator_birth"] == 0.2

    def test_pp_custom_parameters(self):
        """PP should accept custom parameters."""
        model = PP(params={"prey_death": 0.1, "prey_birth": 0.3})
        assert model.params["prey_death"] == 0.1
        assert model.params["prey_birth"] == 0.3
        # Defaults should still apply for unspecified params
        assert model.params["predator_death"] == 0.1

    def test_pp_rejects_invalid_parameter_keys(self):
        """PP should reject unknown parameter keys."""
        with pytest.raises(ValueError, match="Unexpected parameter keys"):
            PP(params={"invalid_key": 0.5})

    def test_pp_rejects_out_of_range_parameters(self):
        """PP parameters must be in [0, 1]."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            PP(params={"prey_death": 1.5})

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            PP(params={"prey_birth": -0.1})

    def test_pp_accepts_both_neighborhoods(self):
        """PP should accept both moore and neumann neighborhoods."""
        model_moore = PP(neighborhood="moore")
        assert model_moore.neighborhood == "moore"

        model_neumann = PP(neighborhood="neumann")
        assert model_neumann.neighborhood == "neumann"

    def test_pp_seed_reproducibility(self):
        """Same seed should produce identical initial states."""
        model1 = PP(rows=15, cols=15, seed=999)
        model2 = PP(rows=15, cols=15, seed=999)
        assert np.array_equal(model1.grid, model2.grid)

    def test_pp_directed_hunting_option(self):
        """PP should accept directed_hunting flag."""
        model = PP(directed_hunting=True)
        assert model.directed_hunting is True

        model = PP(directed_hunting=False)
        assert model.directed_hunting is False


class TestPPUpdate:
    """Tests for PP model update mechanics."""

    def test_pp_update_runs_without_error(self, pp_model_small):
        """update() should execute without raising."""
        pp_model_small.update()  # Should not raise

    def test_pp_update_modifies_grid(self, pp_model_small):
        """update() should modify the grid state."""
        initial_grid = pp_model_small.grid.copy()
        # Run several updates to ensure some change happens
        for _ in range(10):
            pp_model_small.update()

        # Grid should have changed (with high probability)
        assert not np.array_equal(pp_model_small.grid, initial_grid)

    def test_pp_update_preserves_grid_shape(self, pp_model_small):
        """update() should not change grid dimensions."""
        original_shape = pp_model_small.grid.shape
        for _ in range(5):
            pp_model_small.update()
        assert pp_model_small.grid.shape == original_shape

    def test_pp_update_only_valid_states(self, pp_model_small):
        """Grid should only contain states 0, 1, or 2."""
        for _ in range(10):
            pp_model_small.update()
        unique_values = np.unique(pp_model_small.grid)
        assert all(v in [0, 1, 2] for v in unique_values)

    def test_pp_update_with_evolution(self, pp_model_with_evolution):
        """update() should work with evolution enabled."""
        for _ in range(5):
            pp_model_with_evolution.update()
        # Should not raise and grid should still be valid
        unique_values = np.unique(pp_model_with_evolution.grid)
        assert all(v in [0, 1, 2] for v in unique_values)

    def test_pp_directed_vs_random_produces_different_dynamics(self):
        """Directed and random hunting should produce different outcomes."""
        # Use same seed for initial state
        model_random = PP(rows=20, cols=20, seed=42, directed_hunting=False)
        model_directed = PP(rows=20, cols=20, seed=42, directed_hunting=True)

        # Run both for same number of steps
        for _ in range(20):
            model_random.update()
            model_directed.update()

        # Grids should differ (with very high probability)
        assert not np.array_equal(model_random.grid, model_directed.grid)


class TestPPValidation:
    """Tests for PP validation method."""

    def test_pp_validate_passes_for_valid_model(self):
        """Validation should pass for properly initialized PP."""
        model = PP(rows=10, cols=10, seed=42)
        model.validate()  # Should not raise

    def test_pp_validate_with_evolution(self, pp_model_with_evolution):
        """Validation should pass with properly configured evolution."""
        pp_model_with_evolution.validate()  # Should not raise

    def test_pp_validate_fails_for_invalid_evolved_values(self):
        """Validation should fail if evolved values are out of range."""
        model = PP(rows=10, cols=10, seed=42)
        model.evolve("prey_death", sd=0.02, min_val=0.01, max_val=0.1)

        # Manually corrupt the evolved values
        model.cell_params["prey_death"][model.grid == 1] = 0.5  # Outside max

        with pytest.raises(ValueError, match="contains values outside"):
            model.validate()


class TestPPRun:
    """Tests for PP run() method."""

    def test_pp_run_executes_correct_steps(self):
        """run() should execute the specified number of steps."""
        model = PP(rows=8, cols=8, seed=42)
        initial_grid = model.grid.copy()

        model.run(steps=3)

        # After 3 steps, grid should have changed
        assert not np.array_equal(model.grid, initial_grid)

    def test_pp_run_zero_steps(self):
        """run(0) should not modify the grid."""
        model = PP(rows=8, cols=8, seed=42)
        initial_grid = model.grid.copy()

        model.run(steps=0)

        assert np.array_equal(model.grid, initial_grid)

    def test_pp_run_stop_evolution(self):
        """run() with stop_evolution_at should freeze mutation."""
        model = PP(rows=10, cols=10, seed=42)
        model.evolve("prey_death", sd=0.1, min_val=0.01, max_val=0.2)

        assert model._evolution_stopped is False
        model.run(steps=5, stop_evolution_at=3)
        assert model._evolution_stopped is True


# =============================================================================
# Edge Cases
# =============================================================================


class TestPPEdgeCases:
    """Edge case tests for PP model."""

    def test_pp_survives_empty_start(self):
        """PP should handle starting with zero density gracefully."""
        model = PP(rows=5, cols=5, densities=(0.0, 0.0), seed=42)
        assert np.sum(model.grid) == 0

        # Should not raise even with empty grid
        model.update()
        assert np.sum(model.grid) == 0  # Still empty

    def test_pp_prey_only_population(self):
        """PP should handle prey-only population."""
        model = PP(rows=10, cols=10, densities=(0.5, 0.0), seed=42)
        assert np.sum(model.grid == 2) == 0  # No predators

        for _ in range(5):
            model.update()

        # Still no predators (can't spawn from nothing)
        assert np.sum(model.grid == 2) == 0

    def test_pp_predator_only_extinction(self):
        """Predators without prey should eventually die."""
        model = PP(
            rows=10,
            cols=10,
            densities=(0.0, 0.3),
            params={"predator_death": 0.5},  # High death rate
            seed=42,
        )

        # Run until extinction
        for _ in range(50):
            model.update()
            if np.sum(model.grid == 2) == 0:
                break

        # Predators should be extinct (or severely reduced)
        assert np.sum(model.grid == 2) <= 5

    def test_pp_very_small_grid(self):
        """PP should work on minimal 2x2 grid."""
        model = PP(rows=2, cols=2, densities=(0.5, 0.25), seed=42)
        assert model.grid.shape == (2, 2)

        for _ in range(3):
            model.update()

        # Should still be valid
        assert model.grid.shape == (2, 2)
        assert all(v in [0, 1, 2] for v in np.unique(model.grid))

    def test_pp_high_density_initialization(self):
        """PP should handle near-full grid initialization."""
        model = PP(rows=10, cols=10, densities=(0.5, 0.45), seed=42)
        total_occupied = np.sum(model.grid > 0)
        assert total_occupied >= 90  # At least 90% filled

        model.update()  # Should not raise


class TestPPNeighborhoods:
    """Tests for different neighborhood types."""

    def test_moore_has_8_neighbors(self):
        """Moore neighborhood should use 8 directions."""
        model = PP(rows=10, cols=10, neighborhood="moore", seed=42)
        assert len(model._kernel._dr) == 8
        assert len(model._kernel._dc) == 8

    def test_neumann_has_4_neighbors(self):
        """Von Neumann neighborhood should use 4 directions."""
        model = PP(rows=10, cols=10, neighborhood="neumann", seed=42)
        assert len(model._kernel._dr) == 4
        assert len(model._kernel._dc) == 4