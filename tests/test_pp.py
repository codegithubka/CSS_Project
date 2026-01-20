import pytest
import numpy as np
import sys
import os

# Ensure we can import the model from the current directory
sys.path.append(os.getcwd())

# Try importing the classes; fail gracefully if file is missing
try:
    from models.CA import PP
except ImportError:
    pytest.fail("Could not import 'PP' from 'ca_model.py'. Make sure the file exists.")

# --- FIXTURES ---

@pytest.fixture
def base_params():
    """Standard robust parameters for testing."""
    return {
        "prey_death": 0.05,
        "predator_death": 0.1,
        "prey_birth": 0.25,
        "predator_birth": 0.2,
    }

@pytest.fixture
def seed():
    """Fixed seed for reproducibility."""
    return 42

# --- TESTS ---

def test_initialization(base_params, seed):
    """Test grid setup, shapes, and density distribution."""
    rows, cols = 50, 50
    densities = (0.2, 0.1)  # 20% prey, 10% predator
    
    pp = PP(rows, cols, densities, params=base_params, seed=seed)

    # Check grid dimensions
    assert pp.grid.shape == (rows, cols)

    # Check population counts (approximate)
    total_cells = rows * cols
    prey_count = np.sum(pp.grid == 1)
    pred_count = np.sum(pp.grid == 2)

    # Allow small variance due to randomness
    tolerance = total_cells * 0.05
    assert abs(prey_count - total_cells * 0.2) < tolerance
    assert abs(pred_count - total_cells * 0.1) < tolerance

def test_async_update_changes_grid(base_params, seed):
    """Test if Asynchronous update actually modifies the grid."""
    pp = PP(20, 20, (0.5, 0.2), synchronous=False, params=base_params, seed=seed)
    initial_grid = pp.grid.copy()

    pp.update()

    # In a generic CA step with these densities, the grid MUST change
    assert not np.array_equal(pp.grid, initial_grid), "Grid did not change after Async update"

def test_sync_update_changes_grid(base_params, seed):
    """Test if Synchronous update actually modifies the grid."""
    pp = PP(20, 20, (0.5, 0.2), synchronous=True, params=base_params, seed=seed)
    initial_grid = pp.grid.copy()

    pp.update()

    assert not np.array_equal(pp.grid, initial_grid), "Grid did not change after Sync update"

def test_prey_growth_in_isolation(seed):
    """Prey should grow if there are no predators and high birth rate."""
    growth_params = {
        "prey_death": 0.0,
        "predator_death": 1.0,  # Kill any accidental predators
        "prey_birth": 1.0,      # Max birth rate
        "predator_birth": 0.0,
    }
    # Start with only prey (10%)
    pp = PP(20, 20, (0.1, 0.0), params=growth_params, synchronous=True, seed=seed)

    start_count = np.sum(pp.grid == 1)
    pp.update()
    end_count = np.sum(pp.grid == 1)

    assert end_count > start_count, "Prey did not grow in isolation"

def test_predator_starvation(seed):
    """Predators should die if there is no prey."""
    starve_params = {
        "prey_death": 0.0,
        "predator_death": 0.5,  # High death rate
        "prey_birth": 0.0,
        "predator_birth": 1.0,
    }
    # Start with only predators (50%)
    pp = PP(20, 20, (0.0, 0.5), params=starve_params, synchronous=True, seed=seed)

    start_count = np.sum(pp.grid == 2)
    pp.update()
    end_count = np.sum(pp.grid == 2)

    assert end_count < start_count, "Predators did not die from starvation"

def test_parameter_evolution(base_params, seed):
    """Test if per-cell parameters initialize and mutate correctly."""
    pp = PP(30, 30, (0.3, 0.1), params=base_params, seed=seed)

    # Enable evolution for 'prey_death'
    pp.evolve("prey_death", sd=0.05)

    # Check key existence
    assert "prey_death" in pp.cell_params
    
    # Check initialization logic
    param_grid = pp.cell_params["prey_death"]
    prey_mask = (pp.grid == 1)

    # Values should exist where prey exists
    assert np.all(~np.isnan(param_grid[prey_mask]))
    # Values should be NaN where prey does NOT exist
    assert np.all(np.isnan(param_grid[~prey_mask]))

    # Run updates to force reproduction and mutation
    for _ in range(5):
        pp.update()

    # Check for parameter drift (variance)
    current_vals = pp.cell_params["prey_death"]
    valid_vals = current_vals[~np.isnan(current_vals)]

    # If mutation is working, we expect the values to diverge from the initial scalar
    if len(valid_vals) > 5:
        assert np.std(valid_vals) > 0.0, "Parameters did not mutate/drift (variance is 0)"

def test_stability_long_run(base_params, seed):
    """Run for 100 steps to ensure no immediate crash/extinction with default params."""
    pp = PP(50, 50, (0.2, 0.1), synchronous=True, params=base_params, seed=seed)

    extinct = False
    for _ in range(100):
        pp.update()
        n_prey = np.sum(pp.grid == 1)
        n_pred = np.sum(pp.grid == 2)
        
        # We consider 'extinct' if either species drops to 0
        if n_prey == 0 or n_pred == 0:
            extinct = True
            break

    assert not extinct, "Populations went extinct within 100 steps with default parameters"

def test_viz_smoke_test():
    """Ensure visualize() can be called without error (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("Matplotlib not installed")

    try:
        pp = PP(10, 10, (0.2, 0.1))
        # Just initialize visualization, don't keep window open
        pp.visualize(interval=1, pause=0.001)
        plt.close('all')  # Cleanup figures
    except Exception as e:
        pytest.fail(f"visualize() raised an exception: {e}")