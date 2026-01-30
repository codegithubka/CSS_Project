#!/usr/bin/env python3
"""
Predator-Prey Hydra Effect Experiments - HPC Version

Experimental phases (run sequentially):
  Phase 1: Parameter sweep to find critical point (bifurcation + cluster analysis)
  Phase 2: Self-organization analysis (evolution toward criticality)
  Phase 3: Finite-size scaling at critical point
  Phase 4: Sensitivity analysis across parameter regimes
  Phase 5: Model extensions (directed hunting comparison)

Usage:
    python experiments.py --phase 1                    # Run phase 1
    python experiments.py --phase 1 --dry-run          # Estimate runtime
    python experiments.py --phase all                  # Run all phases
    python experiments.py --phase 1 --output results/  # Custom output
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Project imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.config import Config, get_phase_config, PHASE_CONFIGS

# Numba imports
try:
    from models.numba_optimized import (
        compute_all_pcfs_fast,
        get_cluster_stats_fast,
        get_percolating_cluster_fast,
        warmup_numba_kernels,
        set_numba_seed,
        NUMBA_AVAILABLE,
    )

    USE_NUMBA = NUMBA_AVAILABLE
except ImportError:
    USE_NUMBA = False

    def warmup_numba_kernels(size, **kwargs):
        pass

    def set_numba_seed(seed):
        pass


# =============================================================================
# Utility Functions
# =============================================================================

def generate_unique_seed(params: dict, rep: int) -> int:
    """
    Create a deterministic seed from a dictionary of parameters and a repetition index.

    This function serializes the input dictionary into a sorted JSON string, 
    appends the repetition count, and hashes the resulting string using SHA-256. 
    The first 8 characters of the hex digest are then converted to an integer 
    to provide a stable, unique seed for random number generators.

    Parameters
    ----------
    params : dict
    	A dictionary of configuration parameters. Keys are sorted to ensure 
    	determinism regardless of insertion order.
    rep : int
    	The repetition or iteration index, used to ensure different seeds 
    	are generated for the same parameter set across multiple runs.

    Returns
    -------
    int
    	A unique integer seed derived from the input parameters.

    Examples
    --------
    >>> params = {'learning_rate': 0.01, 'batch_size': 32}
    >>> generate_unique_seed(params, 1)
    3432571217
    >>> generate_unique_seed(params, 2)
    3960013583
    """
    identifier = json.dumps(params, sort_keys=True) + f"_{rep}"
    return int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)


def count_populations(grid: np.ndarray) -> Tuple[int, int, int]:
    """
    Count the number of empty, prey, and predator cells in the simulation grid.

    Parameters
    ----------
    grid : np.ndarray
        A 2D NumPy array representing the simulation environment, where:
        - 0: Empty cell
        - 1: Prey
        - 2: Predator

    Returns
    -------
    empty_count : int
        Total number of cells with a value of 0.
    prey_count : int
        Total number of cells with a value of 1.
    predator_count : int
        Total number of cells with a value of 2.

    Examples
    --------
    >>> grid = np.array([[0, 1], [2, 1]])
    >>> count_populations(grid)
    (1, 2, 1)
    """
    return int(np.sum(grid == 0)), int(np.sum(grid == 1)), int(np.sum(grid == 2))


def get_evolved_stats(model, param: str) -> Dict:
    """
    Get statistics of an evolved parameter from the model.

    This function retrieves parameter values from the model's internal storage,
    filters out NaN values, and calculates basic descriptive statistics.

    Parameters
    ----------
    model : object
        The simulation model instance containing a `cell_params` attribute 
        with a `.get()` method.
    param : str
        The name of the parameter to calculate statistics for.

    Returns
    -------
    stats : dict
        A dictionary containing the following keys:
        - 'mean': Arithmetic mean of valid values.
        - 'std': Standard deviation of valid values.
        - 'min': Minimum valid value.
        - 'max': Maximum valid value.
        - 'n': Count of non-NaN values.
        If no valid data is found, all stats return NaN and n returns 0.

    Examples
    --------
    >>> stats = get_evolved_stats(my_model, "speed")
    >>> print(stats['mean'])
    1.25
    """
    arr = model.cell_params.get(param)
    if arr is None:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "n": len(valid),
    }


def average_pcfs(
    pcf_list: List[Tuple[np.ndarray, np.ndarray, int]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average multiple Pair Correlation Function (PCF) measurements and calculate standard error.

    Parameters
    ----------
    pcf_list : list of tuple
        A list where each element is a tuple containing:
        - distances (np.ndarray): The radial distances (r).
        - pcf_values (np.ndarray): The correlation values g(r).
        - count (int): Metadata or weight (not used in current calculation).

    Returns
    -------
    distances : np.ndarray
        The radial distances from the first entry in the list.
    pcf_mean : np.ndarray
        The element-wise mean of the PCF values across all measurements.
    pcf_se : np.ndarray
        The standard error of the mean for the PCF values.

    Examples
    --------
    >>> data = [(np.array([0, 1]), np.array([1.0, 2.0]), 10), 
    ...         (np.array([0, 1]), np.array([1.2, 1.8]), 12)]
    >>> dist, mean, se = average_pcfs(data)
    >>> mean
    array([1.1, 1.9])
    """
    if len(pcf_list) == 0:
        return np.array([]), np.array([]), np.array([])

    distances = pcf_list[0][0]
    pcfs = np.array([p[1] for p in pcf_list])

    pcf_mean = np.mean(pcfs, axis=0)
    pcf_se = np.std(pcfs, axis=0) / np.sqrt(len(pcfs))

    return distances, pcf_mean, pcf_se


def save_results_jsonl(results: List[Dict], output_path: Path):
    """
    Save a list of dictionaries to a file in JSON Lines (JSONL) format.

    Each dictionary in the list is serialized into a single JSON string and
    written as a new line. Non-serializable objects are converted to strings
    using the default string representation.

    Parameters
    ----------
    results : list of dict
        The collection of result dictionaries to be saved.
    output_path : Path
        The file system path (pathlib.Path) where the JSONL file will be created.

    Returns
    -------
    None

    Notes
    -----
    The file is opened in 'w' (write) mode, which will overwrite any existing 
    content at the specified path.

    Examples
    --------
    >>> data = [{"id": 1, "score": 0.95}, {"id": 2, "score": 0.88}]
    >>> save_results_jsonl(data, Path("results.jsonl"))
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, default=str) + "\n")


def save_results_npz(results: List[Dict], output_path: Path):
    """
    Save simulation results to a compressed NumPy (.npz) binary file.

    This function flattens a list of result dictionaries into a single 
    dictionary of NumPy arrays, prefixing keys with the run index to 
    maintain data separation. The resulting file is compressed to 
    reduce storage space.

    Parameters
    ----------
    results : list of dict
        A list where each dictionary contains key-value pairs of 
        simulation data (e.g., arrays, lists, or scalars).
    output_path : Path
        The file system path (pathlib.Path) where the compressed 
        NPZ file will be saved.

    Returns
    -------
    None

    Notes
    -----
    The keys in the saved file follow the format 'run_{index}_{original_key}'.
    Values are automatically converted to NumPy arrays if they are not 
    already.

    Examples
    --------
    >>> results = [{"energy": [1, 2]}, {"energy": [3, 4]}]
    >>> save_results_npz(results, Path("output.npz"))
    """
    data = {}
    for i, res in enumerate(results):
        for key, val in res.items():
            data[f"run_{i}_{key}"] = np.array(val)
    np.savez_compressed(output_path, **data)


def load_results_jsonl(input_path: Path) -> List[Dict]:
    """
    Load simulation results from a JSON Lines (JSONL) formatted file.

    This function reads a file line-by-line, parsing each line as an 
    independent JSON object and aggregating them into a list of dictionaries.

    Parameters
    ----------
    input_path : Path
        The file system path (pathlib.Path) to the JSONL file.

    Returns
    -------
    results : list of dict
        A list of dictionaries reconstructed from the file content.

    Raises
    ------
    FileNotFoundError
        If the specified input path does not exist.
    json.JSONDecodeError
        If a line in the file is not valid JSON.

    Examples
    --------
    >>> data = load_results_jsonl(Path("results.jsonl"))
    >>> len(data)
    2
    """
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


# =============================================================================
# Simulation Functionality
# =============================================================================


def run_single_simulation(
    prey_birth: float,
    prey_death: float,
    predator_birth: float,
    predator_death: float,
    grid_size: int,
    seed: int,
    cfg: Config,
    with_evolution: bool = False,
    compute_pcf: Optional[bool] = None,
) -> Dict:
    """
    Run a single Predator-Prey (PP) simulation and collect comprehensive metrics.

    This function initializes a Cellular Automata model, executes a warmup phase 
    to reach steady state, and then performs a measurement phase to track 
    population dynamics, spatial clustering, and evolutionary changes.

    Parameters
    ----------
    prey_birth : float
        The probability or rate of prey reproduction.
    prey_death : float
        The base probability or rate of prey mortality.
    predator_birth : float
        The probability or rate of predator reproduction upon consuming prey.
    predator_death : float
        The probability or rate of predator mortality.
    grid_size : int
        The side length of the square simulation grid.
    seed : int
        Random seed for ensuring reproducibility of the simulation run.
    cfg : Config
        A configuration object containing simulation hyperparameters (densities, 
        sampling rates, timing, etc.).
    with_evolution : bool, optional
        If True, enables the evolution of the 'prey_death' parameter within 
        the model (default is False).
    compute_pcf : bool, optional
        Explicit toggle for Pair Correlation Function calculation. If None, 
        it is determined by `cfg.pcf_sample_rate` (default is None).

    Returns
    -------
    result : dict
        A dictionary containing simulation results including:
        - Input parameters and survival flags.
        - Population mean and standard deviation for both species.
        - Cluster statistics (number of clusters, sizes, largest fractions).
        - Evolutionary statistics (mean, std, min, max, and final values).
        - PCF data and spatial indices (segregation and clustering).
        - Optional time series for populations and evolved parameters.

    Notes
    -----
    The function relies on several external utilities: `count_populations`, 
    `get_evolved_stats`, `get_cluster_stats_fast`, `compute_all_pcfs_fast`, 
    and `average_pcfs`.
    """

    from models.CA import PP

    if USE_NUMBA:
        set_numba_seed(seed)

    if compute_pcf is None:
        compute_pcf = cfg.collect_pcf and (np.random.random() < cfg.pcf_sample_rate)

    # Initialize model
    model = PP(
        rows=grid_size,
        cols=grid_size,
        densities=cfg.densities,
        neighborhood="moore",  # NOTE: Default neighborhood
        params={
            "prey_birth": prey_birth,
            "prey_death": prey_death,
            "predator_death": predator_death,
            "predator_birth": predator_birth,
        },
        seed=seed,
        synchronous=cfg.synchronous,
        directed_hunting=cfg.directed_hunting,
    )

    if with_evolution:
        model.evolve(
            "prey_death",
            sd=cfg.evolve_sd,
            min_val=cfg.evolve_min,
            max_val=cfg.evolve_max,
        )

    # Scale timing with grid size
    warmup_steps = cfg.get_warmup_steps(grid_size)
    measurement_steps = cfg.get_measurement_steps(grid_size)

    # Warmup phase
    for _ in range(warmup_steps):
        model.update()

    # Measurement phase: start collecting our mertics
    prey_pops, pred_pops = [], []  # Prey populations and predator populations
    evolved_means, evolved_stds = [], []  # Evolution stats over time
    cluster_sizes_prey, cluster_sizes_pred = [], []  # Cluster sizes
    largest_fractions_prey, largest_fractions_pred = (
        [],
        [],
    )  # Largest cluster fractions = size of largest cluster / total population
    pcf_samples = {"prey_prey": [], "pred_pred": [], "prey_pred": []}

    # Determine minimum count for analysis
    min_count = int(cfg.min_density_for_analysis * (grid_size**2))

    for step in range(measurement_steps):
        model.update()

        _, prey, pred = count_populations(model.grid)
        prey_pops.append(prey)
        pred_pops.append(pred)

        # Track evolution
        if with_evolution:
            stats = get_evolved_stats(model, "prey_death")
            evolved_means.append(stats["mean"])
            evolved_stds.append(stats["std"])

        # Cluster analysis (at end of measurement)
        if step == measurement_steps - 1:
            prey_survived = prey_pops[-1] > min_count
            pred_survived = pred_pops[-1] > (min_count // 4)

            if prey_survived:
                prey_stats = get_cluster_stats_fast(model.grid, 1)
                cluster_sizes_prey = prey_stats["sizes"].tolist()
                largest_fractions_prey.append(prey_stats["largest_fraction"])

            if pred_survived:
                pred_stats = get_cluster_stats_fast(model.grid, 2)
                cluster_sizes_pred = pred_stats["sizes"].tolist()
                largest_fractions_pred.append(pred_stats["largest_fraction"])

            # PCF requires both
            if compute_pcf and prey_survived and pred_survived:
                max_dist = min(grid_size / 2, cfg.pcf_max_distance)
                pcf_data = compute_all_pcfs_fast(model.grid, max_dist, cfg.pcf_n_bins)
                pcf_samples["prey_prey"].append(pcf_data["prey_prey"])
                pcf_samples["pred_pred"].append(pcf_data["pred_pred"])
                pcf_samples["prey_pred"].append(pcf_data["prey_pred"])

    # Compile results
    result = {
        # Parameters
        "prey_birth": prey_birth,
        "prey_death": prey_death,
        "predator_birth": predator_birth,
        "predator_death": predator_death,
        "grid_size": grid_size,
        "with_evolution": with_evolution,
        "seed": seed,
        # Population dynamics
        "prey_mean": float(np.mean(prey_pops)),
        "prey_std": float(np.std(prey_pops)),
        "pred_mean": float(np.mean(pred_pops)),
        "pred_std": float(np.std(pred_pops)),
        "prey_survived": prey_pops[-1] > min_count,
        "pred_survived": pred_pops[-1] > (min_count // 4),
        # Cluster statistics
        "prey_n_clusters": len(cluster_sizes_prey),
        "pred_n_clusters": len(cluster_sizes_pred),
        "prey_cluster_sizes": cluster_sizes_prey,
        "pred_cluster_sizes": cluster_sizes_pred,
        # Order parameters
        "prey_largest_fraction": (
            float(np.mean(largest_fractions_prey)) if largest_fractions_prey else np.nan
        ),
        "pred_largest_fraction": (
            float(np.mean(largest_fractions_pred)) if largest_fractions_pred else np.nan
        ),
    }

    # Time series (if requested)
    if cfg.save_timeseries:
        subsample = cfg.timeseries_subsample
        result["prey_timeseries"] = prey_pops[
            ::subsample
        ]  # NOTE: Sample temporal data every 'subsample' steps
        result["pred_timeseries"] = pred_pops[::subsample]

    # Evolution statistics
    if with_evolution and evolved_means:
        valid_means = [v for v in evolved_means if not np.isnan(v)]
        result["evolved_prey_death_mean"] = (
            float(np.mean(valid_means)) if valid_means else np.nan
        )
        result["evolved_prey_death_std"] = (
            float(np.mean([v for v in evolved_stds if not np.isnan(v)]))
            if evolved_stds
            else np.nan
        )
        result["evolved_prey_death_final"] = valid_means[-1] if valid_means else np.nan
        result["evolved_prey_death_min"] = (
            float(np.min(valid_means)) if valid_means else np.nan
        )
        result["evolved_prey_death_max"] = (
            float(np.max(valid_means)) if valid_means else np.nan
        )
        result["evolve_sd"] = cfg.evolve_sd

        if cfg.save_timeseries:
            result["evolved_prey_death_timeseries"] = evolved_means[
                :: cfg.timeseries_subsample
            ]

    # PCF statistics
    if pcf_samples["prey_prey"]:
        dist, pcf_rr, _ = average_pcfs(pcf_samples["prey_prey"])
        _, pcf_cc, _ = average_pcfs(pcf_samples["pred_pred"])
        _, pcf_cr, _ = average_pcfs(pcf_samples["prey_pred"])

        result["pcf_distances"] = dist.tolist()
        result["pcf_prey_prey"] = pcf_rr.tolist()
        result["pcf_pred_pred"] = pcf_cc.tolist()
        result["pcf_prey_pred"] = pcf_cr.tolist()

        # Short-range indices
        short_mask = dist < 3.0
        if np.any(short_mask):
            result["segregation_index"] = float(np.mean(pcf_cr[short_mask]))
            result["prey_clustering_index"] = float(np.mean(pcf_rr[short_mask]))
            result["pred_clustering_index"] = float(np.mean(pcf_cc[short_mask]))

    return result


# =============================================================================
# Experiment Phases
# =============================================================================


def run_phase1(cfg: Config, output_dir: Path, logger: logging.Logger) -> List[Dict]:
    """
    Execute Phase 1 of the simulation: a parameter sweep to identify critical points.

    This function performs a 1D sweep across varying prey mortality rates while 
    keeping other parameters fixed. It utilizes parallel execution via joblib 
    and saves results incrementally to a JSONL file to ensure data integrity 
    during long-running batches.

    Parameters
    ----------
    cfg : Config
        Configuration object containing simulation hyperparameters, sweep 
        ranges, and execution settings (n_jobs, grid_size, etc.).
    output_dir : Path
        Directory where result files (JSONL) and metadata (JSON) will be stored.
    logger : logging.Logger
        Logger instance for tracking simulation progress and recording 
        operational metadata.

    Returns
    -------
    all_results : list of dict
        A list of dictionaries containing the metrics collected from every 
        individual simulation run in the sweep.

    Notes
    -----
    The function performs the following steps:
    1. Pre-warms Numba kernels for performance.
    2. Generates a deterministic set of simulation jobs using unique seeds.
    3. Executes simulations in parallel using a generator for memory efficiency.
    4. Records metadata including a timestamp and a serialized snapshot of 
       the configuration.
    """
    from joblib import Parallel, delayed

    warmup_numba_kernels(cfg.grid_size, directed_hunting=cfg.directed_hunting)

    prey_deaths = cfg.get_prey_deaths()

    # Build job list
    jobs = []
    # Sweep through prey_death only (prey_birth is fixed)
    for pd in prey_deaths:
        for rep in range(cfg.n_replicates):
            params = {"pd": pd}

            seed = generate_unique_seed(params, rep)
            jobs.append(
                (
                    cfg.prey_birth,
                    pd,
                    cfg.predator_birth,
                    cfg.predator_death,
                    cfg.grid_size,
                    seed,
                    cfg,
                    False,
                )
            )

    logger.info(f"Phase 1: {len(jobs):,} simulations")
    logger.info(
        f"  Grid: {cfg.n_prey_death} prey_death values × {cfg.n_replicates} reps (prey_birth={cfg.prey_birth})"
    )
    # Run with incremental saving
    output_jsonl = output_dir / "phase1_results.jsonl"
    all_results = []

    with open(output_jsonl, "w", encoding="utf-8") as f:
        executor = Parallel(n_jobs=cfg.n_jobs, return_as="generator")
        tasks = (delayed(run_single_simulation)(*job) for job in jobs)

        for result in tqdm(executor(tasks), total=len(jobs), desc="Phase 1"):
            f.write(json.dumps(result, default=str) + "\n")
            f.flush()
            all_results.append(result)

    # Save metadata
    meta = {
        "phase": 1,
        "description": "Parameter sweep for critical point",
        "n_sims": len(all_results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(cfg),
    }
    with open(output_dir / "phase1_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Phase 1 complete. Results: {output_jsonl}")
    return all_results


def run_phase2(cfg: Config, output_dir: Path, logger: logging.Logger) -> List[Dict]:
    """
    Execute Phase 2 of the simulation: self-organization and criticality analysis.

    This phase tests the Self-Organized Criticality (SOC) hypothesis by 
    initializing simulations at different points in the parameter space and 
    observing whether evolutionary pressure drives the system toward a 
    common critical point, regardless of initial prey mortality rates.

    Parameters
    ----------
    cfg : Config
        Configuration object containing simulation hyperparameters, evolution 
        settings, and execution constraints.
    output_dir : Path
        Directory where result files (JSONL) and metadata (JSON) will be stored.
    logger : logging.Logger
        Logger instance for tracking progress and evolutionary convergence.

    Returns
    -------
    all_results : list of dict
        A list of dictionaries containing metrics from the evolutionary 
        simulation runs.

    Notes
    -----
    The function captures:
    1. Convergence of 'prey_death' across multiple replicates.
    2. Final steady-state population distributions.
    3. Incremental saving of results to prevent data loss.
    """
    from joblib import Parallel, delayed

    warmup_numba_kernels(cfg.grid_size, directed_hunting=cfg.directed_hunting)

    # Test at multiple prey_birth values
    pb = 0.2
    # Vary intial prey_death
    initial_prey_deaths = np.linspace(
        cfg.prey_death_range[0], cfg.prey_death_range[1], cfg.n_prey_death
    )

    jobs = []
    for initial_pd in initial_prey_deaths:
        for rep in range(cfg.n_replicates):
            params = {"pb": pb, "initial_pd": initial_pd, "phase": 2}
            seed = generate_unique_seed(params, rep)
            jobs.append(
                (
                    pb,
                    initial_pd,
                    cfg.predator_birth,
                    cfg.predator_death,
                    cfg.grid_size,
                    seed,
                    cfg,
                    True,
                )
            )

    logger.info(f"Phase 2: {len(jobs):,} simulations")
    logger.info(f"  prey_birth value: {pb}")
    logger.info(f"  initial prey_death values: {len(initial_prey_deaths)}")
    logger.info(f"  Replicates: {cfg.n_replicates}")

    output_jsonl = output_dir / "phase2_results.jsonl"
    all_results = []

    with open(output_jsonl, "w", encoding="utf-8") as f:
        executor = Parallel(n_jobs=cfg.n_jobs, return_as="generator")
        tasks = (delayed(run_single_simulation)(*job) for job in jobs)

        for result in tqdm(executor(tasks), total=len(jobs), desc="Phase 2"):
            f.write(json.dumps(result, default=str) + "\n")
            f.flush()
            all_results.append(result)

    meta = {
        "phase": 2,
        "description": "Self-organization toward criticality",
        "n_sims": len(all_results),
        "initial_prey_deaths": initial_prey_deaths.tolist(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "phase2_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Phase 2 complete. Results: {output_jsonl}")
    return all_results


def run_phase3(cfg: Config, output_dir: Path, logger: logging.Logger) -> List[Dict]:
    """
    Phase 3: Finite-size scaling at critical point.

    - Multiple grid sizes at (critical_prey_birth, critical_prey_death)
    - Analyze cluster size cutoffs vs L
    """
    from joblib import Parallel, delayed

    # NOTE: Tuned to critical points from phase 1
    pb = cfg.critical_prey_birth
    pd = cfg.critical_prey_death

    logger.info(f"Phase 3: FSS at critical point (pb={pb}, pd={pd})")

    for L in cfg.grid_sizes:
        warmup_numba_kernels(L, directed_hunting=cfg.directed_hunting)

    jobs = []
    for L in cfg.grid_sizes:  # Sweep through grid sizes
        for rep in range(cfg.n_replicates):
            params = {"L": L, "phase": 3}
            seed = generate_unique_seed(params, rep)
            jobs.append(
                (pb, pd, cfg.predator_birth, cfg.predator_death, L, seed, cfg, False)
            )

    logger.info(f"  Grid sizes: {cfg.grid_sizes}")
    logger.info(f"  Total simulations: {len(jobs):,}")

    output_jsonl = output_dir / "phase3_results.jsonl"
    all_results = []

    with open(output_jsonl, "w", encoding="utf-8") as f:
        executor = Parallel(n_jobs=cfg.n_jobs, return_as="generator")
        tasks = (delayed(run_single_simulation)(*job) for job in jobs)

        for result in tqdm(executor(tasks), total=len(jobs), desc="Phase 3"):
            f.write(json.dumps(result, default=str) + "\n")
            f.flush()
            all_results.append(result)

    # Post-run metadata: postprocessing will fit cluster cutoffs vs L
    meta = {
        "phase": 3,
        "description": "Finite-size scaling",
        "critical_point": {"prey_birth": pb, "prey_death": pd},
        "grid_sizes": cfg.grid_sizes,
        "n_sims": len(all_results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "phase3_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Phase 3 complete. Results: {output_jsonl}")
    return all_results


def run_phase4(cfg: Config, output_dir: Path, logger: logging.Logger) -> List[Dict]:
    """
    Execute Phase 3 of the simulation: Finite-Size Scaling (FSS) analysis.

    This phase investigates how spatial structures, specifically cluster size 
    cutoffs, scale with the system size (L) at the critical point identified 
    in Phase 1. This is essential for determining the universality class of 
    the phase transition.

    Parameters
    ----------
    cfg : Config
        Configuration object containing critical point parameters, the list of 
        grid sizes to test, and execution settings.
    output_dir : Path
        Directory where result files (JSONL) and FSS metadata (JSON) will be 
        stored.
    logger : logging.Logger
        Logger instance for tracking progress across different grid sizes.

    Returns
    -------
    all_results : list of dict
        A list of dictionaries containing metrics and cluster statistics for 
        each grid size and replicate.

    Notes
    -----
    The function performs the following:
    1. Iterates through multiple grid sizes defined in `cfg.grid_sizes`.
    2. Generates parallel jobs for each size using critical birth/death rates.
    3. Saves results incrementally to allow for post-simulation analysis of 
       power-law exponents.
    """
    from joblib import Parallel, delayed
    import itertools

    warmup_numba_kernels(cfg.grid_size, directed_hunting=cfg.directed_hunting)

    # Define sweep values
    prey_death_values = np.linspace(0.05, 0.95, 10)  # 10 values for prey_death
    other_param_values = np.linspace(0.0, 1.0, 11)  # 11 values for the rest

    # Logging
    logger.info(f"Phase 4: Full 4D Parameter Sweep")
    logger.info(f"  prey_death: 10 values from 0.05 to 0.95")
    logger.info(f"  prey_birth, pred_birth, pred_death: 11 values each from 0 to 1")
    logger.info(f"  Grid Size: {cfg.grid_size}")
    logger.info(f"  Replicates: {cfg.n_replicates}")

    # Build parameter grid
    param_grid = itertools.product(
        other_param_values,  # prey_birth (11 values)
        prey_death_values,  # prey_death (10 values)
        other_param_values,  # predator_birth (11 values)
        other_param_values,  # predator_death (11 values)
    )

    jobs = []

    for pb, pd, pred_b, pred_d in param_grid:
        for rep in range(cfg.n_replicates):
            params_id = {
                "pb": pb,
                "pd": pd,
                "pred_b": pred_b,
                "pred_d": pred_d,
                "rep": rep,
            }
            seed = generate_unique_seed(params_id, rep)

            jobs.append(
                (
                    pb,  # prey_birth
                    pd,  # prey_death
                    pred_b,  # predator_birth
                    pred_d,  # predator_death
                    cfg.grid_size,
                    seed,
                    cfg,
                    False,
                )
            )

    logger.info(
        f"  Total simulations: {len(jobs):,}"
    )  # 11 * 10 * 11 * 11 * n_reps = 13,310 * n_reps

    output_jsonl = output_dir / "phase4_results.jsonl"
    all_results = []

    with open(output_jsonl, "w", encoding="utf-8") as f:
        executor = Parallel(n_jobs=cfg.n_jobs, return_as="generator")
        tasks = (delayed(run_single_simulation)(*job) for job in jobs)

        for result in tqdm(executor(tasks), total=len(jobs), desc="Phase 4 (4D Sweep)"):
            f.write(json.dumps(result, default=str) + "\n")
            f.flush()
            all_results.append(result)

    # Save Metadata
    meta = {
        "phase": 4,
        "description": "Global 4D Sensitivity Analysis",
        "prey_death_values": prey_death_values.tolist(),
        "other_param_values": other_param_values.tolist(),
        "parameters_varied": [
            "prey_birth",
            "prey_death",
            "predator_birth",
            "predator_death",
        ],
        "n_sims": len(all_results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(cfg),
    }
    with open(output_dir / "phase4_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Phase 4 complete. Results: {output_jsonl}")
    return all_results


def run_phase5(cfg: Config, output_dir: Path, logger: logging.Logger) -> List[Dict]:
    """
    Phase 5: Perturbation analysis (critical slowing down).

    - Points around critical point
    - Full time series for autocorrelation analysis
    - Measure relaxation times
    """
    from joblib import Parallel, delayed

    warmup_numba_kernels(cfg.grid_size, directed_hunting=cfg.directed_hunting)

    pb = cfg.critical_prey_birth
    base_pd = cfg.critical_prey_death

    jobs = []
    for offset in cfg.prey_death_offsets:
        pd = base_pd + offset
        if pd <= 0:
            continue

        for rep in range(cfg.n_replicates):
            params = {"offset": offset, "phase": 5}
            seed = generate_unique_seed(params, rep)
            jobs.append(
                (
                    pb,
                    pd,
                    cfg.predator_birth,
                    cfg.predator_death,
                    cfg.grid_size,
                    seed,
                    cfg,
                    False,
                )
            )

    logger.info(f"Phase 5: {len(jobs):,} simulations")
    logger.info(f"  prey_death offsets: {cfg.prey_death_offsets}")
    logger.info(f"  Base critical point: pb={pb}, pd={base_pd}")

    output_jsonl = output_dir / "phase5_results.jsonl"
    all_results = []

    with open(output_jsonl, "w", encoding="utf-8") as f:
        executor = Parallel(n_jobs=cfg.n_jobs, return_as="generator")
        tasks = (delayed(run_single_simulation)(*job) for job in jobs)

        for result in tqdm(executor(tasks), total=len(jobs), desc="Phase 5"):
            f.write(json.dumps(result, default=str) + "\n")
            f.flush()
            all_results.append(result)

    meta = {
        "phase": 5,
        "description": "Perturbation analysis / critical slowing down",
        "n_sims": len(all_results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "phase5_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Phase 5 complete. Results: {output_jsonl}")
    return all_results


def run_phase6(cfg: Config, output_dir: Path, logger: logging.Logger) -> List[Dict]:
    """
    Execute Phase 6 of the simulation: Global 4D parameter sweep with directed hunting.

    This phase performs a comprehensive sensitivity analysis by varying four key 
    parameters (prey birth/death and predator birth/death) while directed 
    hunting is enabled. The results allow for a direct comparison with Phase 4 
    to determine how predator search behavior shifts the system's critical 
    thresholds and stability.

    Parameters
    ----------
    cfg : Config
        Configuration object containing simulation hyperparameters, parallel 
        execution settings, and the fixed grid size for this phase.
    output_dir : Path
        Directory where the result JSONL file and execution metadata will 
        be stored.
    logger : logging.Logger
        Logger instance for tracking the progress of the high-volume 
        simulation batch.

    Returns
    -------
    all_results : list of dict
        A list of dictionaries containing metrics for every simulation in 
        the 4D parameter grid.

    Notes
    -----
    The function utilizes a Cartesian product of parameter ranges to build a 
    job list of over 13,000 unique parameter sets (multiplied by replicates). 
    Seeds are uniquely generated to distinguish these runs from other phases 
    even if parameter values overlap.
    """
    from joblib import Parallel, delayed
    import itertools

    warmup_numba_kernels(cfg.grid_size, directed_hunting=cfg.directed_hunting)

    # Define sweep values (same as Phase 4)
    prey_death_values = np.linspace(0.05, 0.95, 10)  # 10 values for prey_death
    other_param_values = np.linspace(0.0, 1.0, 11)  # 11 values for the rest

    # Logging
    logger.info(f"Phase 6: Full 4D Parameter Sweep (Directed Hunting)")
    logger.info(f"  prey_death: 10 values from 0.05 to 0.95")
    logger.info(f"  prey_birth, pred_birth, pred_death: 11 values each from 0 to 1")
    logger.info(f"  Grid Size: {cfg.grid_size}")
    logger.info(f"  Replicates: {cfg.n_replicates}")
    logger.info(f"  Directed Hunting: {cfg.directed_hunting}")

    # Build parameter grid
    param_grid = itertools.product(
        other_param_values,  # prey_birth (11 values)
        prey_death_values,  # prey_death (10 values)
        other_param_values,  # predator_birth (11 values)
        other_param_values,  # predator_death (11 values)
    )

    jobs = []

    for pb, pd, pred_b, pred_d in param_grid:
        for rep in range(cfg.n_replicates):
            # Include phase identifier to ensure different seeds from Phase 4
            params_id = {
                "pb": pb,
                "pd": pd,
                "pred_b": pred_b,
                "pred_d": pred_d,
                "phase": 6,
                "rep": rep,
            }
            seed = generate_unique_seed(params_id, rep)

            jobs.append(
                (
                    pb,  # prey_birth
                    pd,  # prey_death
                    pred_b,  # predator_birth
                    pred_d,  # predator_death
                    cfg.grid_size,
                    seed,
                    cfg,
                    False,
                )
            )

    logger.info(
        f"  Total simulations: {len(jobs):,}"
    )  # 11 * 10 * 11 * 11 * n_reps = 13,310 * n_reps

    output_jsonl = output_dir / "phase6_results.jsonl"
    all_results = []

    with open(output_jsonl, "w", encoding="utf-8") as f:
        executor = Parallel(n_jobs=cfg.n_jobs, return_as="generator")
        tasks = (delayed(run_single_simulation)(*job) for job in jobs)

        for result in tqdm(
            executor(tasks), total=len(jobs), desc="Phase 6 (4D Sweep + Directed)"
        ):
            f.write(json.dumps(result, default=str) + "\n")
            f.flush()
            all_results.append(result)

    # Save Metadata
    meta = {
        "phase": 6,
        "description": "Global 4D Sensitivity Analysis with Directed Hunting",
        "prey_death_values": prey_death_values.tolist(),
        "other_param_values": other_param_values.tolist(),
        "parameters_varied": [
            "prey_birth",
            "prey_death",
            "predator_birth",
            "predator_death",
        ],
        "directed_hunting": cfg.directed_hunting,
        "n_sims": len(all_results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(cfg),
    }
    with open(output_dir / "phase6_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Phase 6 complete. Results: {output_jsonl}")
    return all_results


# =============================================================================
# Main:
# =============================================================================

PHASE_RUNNERS = {
    1: run_phase1,
    2: run_phase2,
    3: run_phase3,
    4: run_phase4,
    5: run_phase5,
    6: run_phase6,
}


def main():
    """
    Orchestrate the predator-prey experimental suite across multiple phases.

    This entry point handles command-line arguments, sets up logging and output 
    directories, and executes the requested simulation phases (1-6). It 
    supports parallel execution, dry runs for runtime estimation, and 
    automated configuration persistence.

    Notes
    -----
    The script dynamically retrieves phase-specific configurations using 
    `get_phase_config` and dispatches execution to the corresponding runner 
    in the `PHASE_RUNNERS` mapping.
    """
    parser = argparse.ArgumentParser(
        description="Predator-Prey Hydra Effect Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  1  Parameter sweep to find critical point
  2  Self-organization (evolution toward criticality)
  3  Finite-size scaling at critical point
  4  Sensitivity analysis across parameter regimes
  5  Model extensions (directed hunting comparison)
        """,
    )
    parser.add_argument(
        "--phase", type=str, required=True, help="Phase to run: 1-6 or 'all'"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--cores", type=int, default=-1, help="Number of cores (-1 for all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Estimate runtime without running"
    )
    args = parser.parse_args()

    # Parse phase argument
    if args.phase.lower() == "all":
        phases = list(PHASE_RUNNERS.keys())
    else:
        try:
            phases = [int(args.phase)]
        except ValueError:
            print(f"Invalid phase: {args.phase}. Use 1-6 or 'all'")
            sys.exit(1)

    # Setup output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.output / "experiments.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Header
    logger.info("=" * 60)
    logger.info("PREDATOR-PREY HYDRA EFFECT EXPERIMENTS")
    logger.info("=" * 60)
    logger.info(f"Phases: {phases}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Cores: {args.cores}")
    logger.info(f"Numba: {'ENABLED' if USE_NUMBA else 'DISABLED'}")

    # Process each phase
    for phase in phases:
        cfg = get_phase_config(phase)
        cfg.n_jobs = (
            args.cores
            if args.cores > 0
            else int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
        )

        logger.info("")
        logger.info(f"{'='*60}")
        logger.info(f"PHASE {phase}")
        logger.info(f"{'='*60}")

        n_cores = cfg.n_jobs if cfg.n_jobs > 0 else os.cpu_count()
        logger.info(f"Estimated: {cfg.estimate_runtime(n_cores)}")

        if args.dry_run:
            logger.info("Dry run - skipping execution")
            continue

        # Save config
        with open(args.output / f"phase{phase}_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)

        # Run phase
        start_time = time.time()
        runner = PHASE_RUNNERS[phase]
        runner(cfg, args.output, logger)
        elapsed = time.time() - start_time

        logger.info(f"Phase {phase} runtime: {elapsed/60:.1f} minutes")

    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPERIMENTS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
