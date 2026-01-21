"""Cellular Automaton tests."""

import pytest
import numpy as np
from models.CA import CA, PP


def test_initialization_and_grid_filling():
    rows, cols = 10, 10
    densities = (0.2, 0.1)
    ca = CA(rows, cols, densities, neighborhood="moore", params={}, cell_params={}, seed=42)
    assert ca.grid.shape == (rows, cols)
    assert ca.n_species == len(densities)
    total_cells = rows * cols
    # expected counts use the same rounding as CA.__init__
    expected_counts = [int(round(total_cells * d)) for d in densities]
    # verify actual counts equal expected
    for i, exp in enumerate(expected_counts, start=1):
        assert int(np.count_nonzero(ca.grid == i)) == exp


def test_invalid_parameters_raise():
    # invalid rows/cols
    with pytest.raises(AssertionError):
        CA(0, 5, (0.1,), "moore", {}, {}, seed=1)
    with pytest.raises(AssertionError):
        CA(5, -1, (0.1,), "moore", {}, {}, seed=1)
    # densities must be non-empty tuple
    with pytest.raises(AssertionError):
        CA(5, 5, (), "moore", {}, {}, seed=1)
    # densities sum > 1
    with pytest.raises(AssertionError):
        CA(5, 5, (0.8, 0.8), "moore", {}, {}, seed=1)
    # invalid neighborhood
    with pytest.raises(AssertionError):
        CA(5, 5, (0.1,), "invalid", {}, {}, seed=1)

    # PP: params must be a dict or None
    with pytest.raises(TypeError):
        PP(rows=5, cols=5, densities=(0.2, 0.1), neighborhood="moore", params="bad", cell_params=None, seed=1)


def test_neighborhood_counting():
    # set up a small grid with a single prey in the center and check neighbor counts
    ca = CA(3, 3, (0.0,), neighborhood="moore", params={}, cell_params={}, seed=1)
    ca.grid[:] = 0
    ca.grid[1, 1] = 1
    counts = ca.count_neighbors()
    # counts is a tuple with one array (state 1)
    neigh = counts[0]
    # all 8 neighbors of center should have count 1
    expected_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    for r in range(3):
        for c in range(3):
            if (r, c) in expected_positions:
                assert neigh[r, c] == 1
            else:
                # center has 0 neighbors of same state
                assert neigh[r, c] == 0


def test_validate_detects_cell_params_shape_and_nonnan_mismatch():
    # create a PP and enable evolution for a parameter
    pp = PP(rows=5, cols=5, densities=(0.2, 0.1), neighborhood="moore", params=None, cell_params=None, seed=2)
    pp.evolve("prey_death", sd=0.01, min_val=0.0, max_val=1.0)

    # wrong shape should raise informative ValueError via validate()
    pp.cell_params["prey_death"] = np.zeros((1, 1))
    with pytest.raises(ValueError) as excinfo:
        pp.validate()
    assert "shape equal to grid" in str(excinfo.value)

    # now create a same-shaped array but with non-NaN positions that don't match prey positions
    arr = np.zeros(pp.grid.shape, dtype=float)  # filled with non-NaN everywhere
    pp.cell_params["prey_death"] = arr
    with pytest.raises(ValueError) as excinfo2:
        pp.validate()
    assert "non-NaN entries must match positions" in str(excinfo2.value)


def test_extinction_when_death_one():
    # when both death rates are 1 all individuals should die in one step
    params = {
        "prey_death": 1.0,
        "predator_death": 1.0,
        "prey_birth": 0.0,
        "predator_birth": 0.0,
    }
    pp = PP(rows=10, cols=10, densities=(0.2, 0.1), neighborhood="moore", params=params, cell_params=None, seed=3)
    pp.run(1)
    # no prey or predators should remain
    assert np.count_nonzero(pp.grid != 0) == 0


def test_predators_dominate_with_high_birth_and_zero_predator_death():
    params = {
        "prey_death": 0.0,
        "predator_death": 0.0,
        "prey_birth": 1.0,
        "predator_birth": 1.0,
    }
    pp = PP(rows=10, cols=10, densities=(0.1, 0.05), neighborhood="moore", params=params, cell_params=None, seed=4)
    # run longer to allow predators to consume prey; expect prey extinction
    pp.run(200)
    after_prey = int(np.count_nonzero(pp.grid == 1))
    after_pred = int(np.count_nonzero(pp.grid == 2))
    # after sufficient time, prey should go extinct and predators remain
    assert after_prey == 0
    assert after_pred > 0
