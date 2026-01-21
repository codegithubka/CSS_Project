import numpy as np
from scipy.ndimage import label


def find_clusters(grid: np.ndarray, state: int = 1, connectivity: int = "moore"):
    """Locate connected clusters of a given state."""
    if connectivity == "moore":
        structure = np.ones((3, 3), dtype=int)
    else:
        structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

    binary_mask = (grid == state).astype(int)
    labeled_array, num_clusters = label(binary_mask, structure=structure)

    return labeled_array, num_clusters


def get_cluster_sizes(grid: np.ndarray, state: int = 1):
    """Returns an array containing the sizes of all detected clusters."""
    labeled, n = find_clusters(grid, state)

    if n == 0:
        return np.array([])

    sizes = np.bincount(labeled.ravel())[1:]  # Exclude background count

    return sizes


def fit_power_law_mle(sizes: np.ndarray, s_min: float = 1.0):
    """
    Maximum Likelihood Estimator for the power-law exponent.
    """
    # Filter sizes below the minimum threshold
    sizes = sizes[sizes >= s_min]

    if len(sizes) < 10:
        return {"tau": np.nan, "tau_err": np.nan, "n_samples": len(sizes)}

    n = len(sizes)

    # Standard MLE formula for power-law index
    tau = 1 + n / np.sum(np.log(sizes / s_min))
    tau_err = (tau - 1) / np.sqrt(n)

    return {"tau": tau, "tau_err": tau_err, "n_samples": n}


def detect_hydra_effect(d_r_vals: np.ndarray, R_vals: np.ndarray):
    """Detects hydra effect by identifying increase in prey population with increased death rate."""

    dR = np.gradient(R_vals, d_r_vals)
    hydra_mask = dR > 0

    if not np.any(hydra_mask):
        return {"detected": False, "region": None, "max_strength": 0.0}
    hydra_indices = np.where(hydra_mask)[0]

    results = {
        "detected": True,
        "d_r_start": d_r_vals[hydra_indices[0]],
        "d_r_end": d_r_vals[hydra_indices[-1]],
        "max_slope": np.max(dR),
        "d_r_at_max_slope": d_r_vals[np.argmax(dR)],
        "peak_density": np.max(R_vals[hydra_mask]),
    }

    return results


def find_spatial_correlation(d_r_vals, ca_results):
    """ca_results: List of grids (np.ndarray) at equilibrium of each d_r value."""
    pass
