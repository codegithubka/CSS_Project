# CA Stochastic Bifurcation Diagram:
Mutations (evolutions) parameter OFF
Control parameter: prey death rate

Possible statistical observables:
- Fraction of prey cells at equilibrium
- Measure of entropy of the generated pattern.
- Prey population count
- Predator population count

Run simulation:
- Let the system run until a steady state is observed
- For each death rate value, let the CA run for a specified number of iterations after warmp up, show distribution (scatters) for each sim run at a given prey death rate, and the average line


# Phase 1: finding the critical point
- Create bifurcation diagram of mean population count, varying prey death rate
	- Look for critical transition
- Create log-log plot of cluster size distribution, varying prey death rate
	- Look for power-law

# Experiment Phase: CA Stochastic Bifurcation Diagram:

1) Write a Config Object specific to that experiment
2) Make sure the experiment running on the cluster is running 15 reps of each runs at all sweeped values.
3) Make sure the outputs of the experiment are a 1D and 2D array (explained below)

# Bifurcation Diagram Prompts:
1) Help me write a function for creating a stochastic bifurcation diagram, of the population count at equilibrium, varying the prey death rate (as the control variable). 
2) At each sweeped value of the prey death control variable, we should be measuring the population count at equilibrium for at least 15 simulation runs. 
3) Which means that the two inputs for my function should be a 1D Array for the sweep parameter, and a 2D array for the experiment results at each sweep for the rows, and the results for each iteration for the columns.
4) When running my function, using the argparse module, my command-line arguments specifies which analysis to do, in this case the analysis is the bifurcation diagram.


# Output: 
def load_bifurcation_results(results_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load bifurcation analysis results.
    
    Returns
    -------
    sweep_params : np.ndarray
        1D array of control parameter values (prey death rates).
    results : np.ndarray
        2D array of shape (n_sweep, n_replicates) with population counts
        at equilibrium.
    """
    npz_file = results_dir / "bifurcation_results.npz"
    json_file = results_dir / "bifurcation_results.json"
    
    if npz_file.exists():
        logging.info(f"Loading bifurcation results from {npz_file}")
        data = np.load(npz_file)
        return data['sweep_params'], data['results']
    elif json_file.exists():
        logging.info(f"Loading bifurcation results from {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        return np.array(data['sweep_params']), np.array(data['results'])
    else:
        raise FileNotFoundError(f"Bifurcation results not found in {results_dir}")


def plot_bifurcation_diagram(sweep_params: np.ndarray, results: np.ndarray,
                             output_dir: Path, dpi: int = 150,
                             control_label: str = "Prey Death Rate",
                             population_label: str = "Population at Equilibrium"):
    """
    Generate a stochastic bifurcation diagram.
    
    Shows the distribution of equilibrium population counts as a function of
    a control parameter (e.g., prey death rate), with scatter points for each
    replicate run overlaid on summary statistics.
    
    Parameters
    ----------
    sweep_params : np.ndarray
        1D array of control parameter values (e.g., prey death rates).
        Shape: (n_sweep,)
    results : np.ndarray
        2D array of population counts at equilibrium.
        Shape: (n_sweep, n_replicates) where rows correspond to sweep_params
        and columns are replicate simulation runs.
    output_dir : Path
        Directory to save the output figure.
    dpi : int
        Output resolution (default: 150).
    control_label : str
        Label for x-axis (control parameter).
    population_label : str
        Label for y-axis (population count).
    """
    n_sweep, n_replicates = results.shape
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Scatter all individual replicates with transparency
    for i, param in enumerate(sweep_params):
        ax.scatter(
            np.full(n_replicates, param),
            results[i, :],
            alpha=0.3, s=15, c='steelblue', edgecolors='none'
        )
    
    # Compute summary statistics
    means = np.mean(results, axis=1)
    medians = np.median(results, axis=1)
    q25 = np.percentile(results, 25, axis=1)
    q75 = np.percentile(results, 75, axis=1)
    
    # Plot median line and IQR envelope
    ax.fill_between(sweep_params, q25, q75, alpha=0.25, color='coral',
                    label='IQR (25th-75th percentile)')
    ax.plot(sweep_params, medians, 'o-', color='darkred', linewidth=2,
            markersize=5, label='Median')
    ax.plot(sweep_params, means, 's--', color='black', linewidth=1.5,
            markersize=4, alpha=0.7, label='Mean')
    
    ax.set_xlabel(control_label)
    ax.set_ylabel(population_label)
    ax.set_title(f"Stochastic Bifurcation Diagram\n({n_replicates} replicates per parameter value)")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add rug plot at bottom showing parameter sampling density
    ax.plot(sweep_params, np.zeros_like(sweep_params), '|', color='gray',
            markersize=10, alpha=0.5)
    
    plt.tight_layout()
    output_file = output_dir / "bifurcation_diagram.png"
    plt.savefig(output_file, dpi=dpi)
    plt.close()
    logging.info(f"Saved {output_file}")
    
    return output_file
