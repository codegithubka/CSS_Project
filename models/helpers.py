import numpy as np
from scipy.ndimage import label


# def find_clusters(grid: np.ndarray, state: int = 1, connectivity: int = "moore"):
#     """Locate connected clusters of a given state."""
#     if connectivity == "moore":
#         structure = np.ones((3, 3), dtype=int)
#     else:
#         structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

#     binary_mask = (grid == state).astype(int)
#     labeled_array, num_clusters = label(binary_mask, structure=structure)

#     return labeled_array, num_clusters


# def get_cluster_sizes(grid: np.ndarray, state: int = 1):
#     """Returns an array containing the sizes of all detected clusters."""
#     labeled, n = find_clusters(grid, state)

#     if n == 0:
#         return np.array([])

#     sizes = np.bincount(labeled.ravel())[1:]  # Exclude background count

#     return sizes

def get_cluster_history(ca):
	"""Retrieve cluster tracking data from a CA instance after simulation.
	
	This function extracts cluster statistics that were tracked during
	a simulation run with show_clusters=True in the visualize() method.
	
	Args:
		ca: CA instance (or subclass like PP) that was run with visualization
		    and cluster tracking enabled.
	
	Returns:
		dict or None: Dictionary containing cluster statistics over time with structure:
			{
				'iterations': list of int - timesteps when data was collected
				'prey': {
					'n_clusters': list of int - number of prey clusters at each timestep
					'largest': list of int - size of largest prey cluster at each timestep
					'mean': list of float - mean prey cluster size at each timestep
				},
				'predator': {
					'n_clusters': list of int - number of predator clusters
					'largest': list of int - size of largest predator cluster
					'mean': list of float - mean predator cluster size
				}
			}
		Returns None if cluster tracking was not enabled or no data was collected.
	
	Example:
		>>> from CA import PP
		>>> from helpers import get_cluster_history  # Or whatever your file is named
		>>> 
		>>> ca = PP(rows=100, cols=100, densities=(0.4, 0.1))
		>>> ca.visualize(interval=5, show_clusters=True)
		>>> ca.run(steps=500)
		>>> 
		>>> data = get_cluster_history(ca)
		>>> if data:
		>>>     print(f"Tracked {len(data['iterations'])} timepoints")
		>>>     print(f"Final largest prey cluster: {data['prey']['largest'][-1]}")
	
	Notes:
		- Only works if ca.visualize() was called with show_clusters=True
		- Data is collected at the visualization interval (not every timestep)
		- Both prey (state=1) and predator (state=2) statistics are tracked
	"""
	# Check if cluster tracking was enabled
	if not hasattr(ca, '_viz_cluster_stats'):
		return None
	
	# Check if any data was actually collected
	if not hasattr(ca, '_viz_time') or len(ca._viz_time) == 0:
		return None
	
	# Return structured data
	return {
		'iterations': ca._viz_time,
		'prey': {
			'n_clusters': ca._viz_cluster_stats['prey_n_clusters'],
			'largest': ca._viz_cluster_stats['prey_largest'],
			'mean': ca._viz_cluster_stats['prey_mean']
		},
		'predator': {
			'n_clusters': ca._viz_cluster_stats['pred_n_clusters'],
			'largest': ca._viz_cluster_stats['pred_largest'],
			'mean': ca._viz_cluster_stats['pred_mean']
		}
	}


def plot_cluster_evolution(cluster_data, save_path=None):
	"""Plot cluster statistics evolution over time.
	
	Creates a 2x2 grid of plots showing:
	- Number of clusters over time
	- Largest cluster size over time
	- Mean cluster size over time
	- Largest cluster as percentage of grid
	
	Args:
		cluster_data: Dictionary returned by get_cluster_history()
		save_path: Optional path to save the figure (e.g., 'clusters.png')
	
	Returns:
		matplotlib Figure object
	
	Example:
		>>> data = get_cluster_history(ca)
		>>> plot_cluster_evolution(data, save_path='cluster_evolution.png')
	"""
	import matplotlib.pyplot as plt
	import numpy as np
	
	if cluster_data is None:
		print("No cluster data available!")
		return None
	
	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	iterations = cluster_data['iterations']
	
	# 1. Number of clusters
	axes[0, 0].plot(iterations, cluster_data['prey']['n_clusters'], 
	                label='Prey', color='green', linewidth=2)
	axes[0, 0].plot(iterations, cluster_data['predator']['n_clusters'], 
	                label='Predator', color='red', linewidth=2)
	axes[0, 0].set_xlabel('Iteration', fontsize=12)
	axes[0, 0].set_ylabel('Number of clusters', fontsize=12)
	axes[0, 0].legend(fontsize=11)
	axes[0, 0].set_title('Cluster Count Evolution', fontsize=13, fontweight='bold')
	axes[0, 0].grid(True, alpha=0.3)
	
	# 2. Largest cluster size
	axes[0, 1].plot(iterations, cluster_data['prey']['largest'], 
	                label='Prey', color='green', linewidth=2)
	axes[0, 1].plot(iterations, cluster_data['predator']['largest'], 
	                label='Predator', color='red', linewidth=2)
	axes[0, 1].set_xlabel('Iteration', fontsize=12)
	axes[0, 1].set_ylabel('Largest cluster size', fontsize=12)
	axes[0, 1].legend(fontsize=11)
	axes[0, 1].set_title('Largest Cluster Size', fontsize=13, fontweight='bold')
	axes[0, 1].grid(True, alpha=0.3)
	
	# 3. Mean cluster size
	axes[1, 0].plot(iterations, cluster_data['prey']['mean'], 
	                label='Prey', color='green', linewidth=2)
	axes[1, 0].plot(iterations, cluster_data['predator']['mean'], 
	                label='Predator', color='red', linewidth=2)
	axes[1, 0].set_xlabel('Iteration', fontsize=12)
	axes[1, 0].set_ylabel('Mean cluster size', fontsize=12)
	axes[1, 0].legend(fontsize=11)
	axes[1, 0].set_title('Mean Cluster Size', fontsize=13, fontweight='bold')
	axes[1, 0].grid(True, alpha=0.3)
	
	# 4. Largest cluster as percentage (requires knowing grid size)
	# Try to get grid size from the CA if available
	prey_largest = cluster_data['prey']['largest']
	pred_largest = cluster_data['predator']['largest']
	
	axes[1, 1].plot(iterations, prey_largest, 
	                label='Prey', color='green', linewidth=2)
	axes[1, 1].plot(iterations, pred_largest, 
	                label='Predator', color='red', linewidth=2)
	axes[1, 1].set_xlabel('Iteration', fontsize=12)
	axes[1, 1].set_ylabel('Largest cluster size (absolute)', fontsize=12)
	axes[1, 1].legend(fontsize=11)
	axes[1, 1].set_title('Largest Cluster (Absolute Size)', fontsize=13, fontweight='bold')
	axes[1, 1].grid(True, alpha=0.3)
	
	plt.tight_layout()
	
	if save_path:
		plt.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f"Cluster evolution plot saved to '{save_path}'")
	
	return fig


def get_cluster_size_distribution(ca, state=1):
	"""Get the current cluster size distribution.
	
	Computes the distribution n(s) - number of clusters of size s.
	This is the empirical version of the theoretical n(s,p) distribution
	from percolation theory.
	
	Args:
		ca: CA instance
		state: State to analyze (1=prey, 2=predator)
	
	Returns:
		dict: Mapping from cluster size -> count of clusters with that size
	
	Example:
		>>> dist = get_cluster_size_distribution(ca, state=1)
		>>> print(f"Found {dist[1]} single-cell prey clusters")
		>>> print(f"Found {dist.get(100, 0)} clusters with exactly 100 prey")
	"""
	stats = ca.get_cluster_stats(state=state)
	return stats['size_distribution']


def detect_percolation_transition(cluster_data, threshold=0.1):
	"""Detect when a percolation transition occurs.
	
	Identifies the iteration when the largest cluster exceeds a threshold
	fraction of the grid size, indicating a phase transition.
	
	Args:
		cluster_data: Dictionary from get_cluster_history()
		threshold: Fraction of grid (0 to 1) to consider as transition
	
	Returns:
		dict: Information about the transition, or None if no transition detected
			{
				'iteration': int - when transition occurred
				'largest_cluster': int - cluster size at transition
				'n_clusters': int - total clusters at transition
			}
	
	Example:
		>>> data = get_cluster_history(ca)
		>>> transition = detect_percolation_transition(data, threshold=0.1)
		>>> if transition:
		>>>     print(f"Percolation at iteration {transition['iteration']}")
	"""
	import numpy as np
	
	if cluster_data is None:
		return None
	
	iterations = np.array(cluster_data['iterations'])
	largest = np.array(cluster_data['prey']['largest'])
	n_clusters = np.array(cluster_data['prey']['n_clusters'])
	
	# Estimate grid size from maximum cluster size seen
	grid_size = max(largest) * 2  # Conservative estimate
	
	fraction = largest / grid_size
	critical_idx = np.where(fraction > threshold)[0]
	
	if len(critical_idx) > 0:
		idx = critical_idx[0]
		return {
			'iteration': int(iterations[idx]),
			'largest_cluster': int(largest[idx]),
			'n_clusters': int(n_clusters[idx]),
			'fraction': float(fraction[idx])
		}
	
	return None


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
