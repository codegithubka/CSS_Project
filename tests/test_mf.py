import pytest
import numpy as np
from models.mean_field import MeanFieldModel


@pytest.fixture
def model():
    """Model instance for testing."""
    return MeanFieldModel()


def test_initialization(model):
    """Test model initialization with default parameters."""
    assert model.birth == 0.2
    assert model.consumption == 0.8
    assert model.predator_death == 0.045
    assert model.conversion == 1.0
    assert model.prey_competition == 0.1
    assert model.predator_competition == 0.05
    assert model.pred_benifit == model.consumption * model.conversion


def test_prey_extinction(model):
    """Verify prey extinction logic."""
    R_eq, C_eq = model.equilibrium(prey_death=0.3)
    assert R_eq == 0.0
    assert C_eq == 0.0


def test_monotonicity(model):
    """Test monotonicity of equilibrium populations with respect to prey death rate."""
    d_r_range = np.linspace(0.01, 0.08, 10)
    sweep = model.sweep_death_rate(d_r_range)
    assert np.all(np.diff(sweep["R_eq"]) <= 0)


def test_convergence(model):
    ana_R, _ = model.equilibrium(0.05)
    num_R, _ = model.equilibrium_numerical(0.05)

    # Use approx for floating point comparisons in numerical analysis
    assert num_R == pytest.approx(ana_R, rel=1e-2)
