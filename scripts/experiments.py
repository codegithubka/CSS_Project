#!/usr/bin/env python3
"""
Predator-Prey Hydra Effect Experiments - HPC Version

Experimental phases (run sequentially):
  Phase 1: Parameter sweep to find critical point (bifurcation + cluster analysis)
  Phase 2: Self-organization analysis (evolution toward criticality)
  Phase 3: Finite-size scaling at critical point
  Phase 4: Sensitivity analysis across parameter regimes
  Phase 5: Perturbation analysis (critical slowing down)
  Phase 6: Model extensions (directed hunting comparison)

Usage:
    python experiments.py --phase 1                    # Run phase 1
    python experiments.py --phase 1 --dry-run          # Estimate runtime
    python experiments.py --phase all                  # Run all phases
    python experiments.py --phase 1 --output results/  # Custom output
"""


# NOTE (1): The soc_analysis script used temporal avalache data to assess SOC.
# This functionality is not yet implemented here. We can still derive that data
# from the full time series using np.diff(prey_timeseries)

# NOTE (2): Post-processing utilities and plotting are in scripts/analysis.py. This script should
# solely focus on running the experiments and saving raw results.


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
    def warmup_numba_kernels(size, **kwargs): pass
    def set_numba_seed(seed): pass


# =============================================================================
# Utility Functions
# =============================================================================

def generate_unique_seed(params: dict, rep: int) -> int:
    """Create deterministic seed from parameters."""
    identifier = json.dumps(params, sort_keys=True) + f"_{rep}"
    return int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)


def count_populations(grid: np.ndarray) -> Tuple[int, int, int]:
    """Count empty, prey, predator cells."""
    return int(np.sum(grid == 0)), int(np.sum(grid == 1)), int(np.sum(grid == 2))


def get_evolved_stats(model, param: str) -> Dict:
    """Get statistics of evolved parameter from model."""
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

def average_pcfs(pcf_list: List[Tuple[np.ndarray, np.ndarray, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average multiple PCF measurements with standard error."""
    if len(pcf_list) == 0:
        return np.array([]), np.array([]), np.array([])
    
    distances = pcf_list[0][0]
    pcfs = np.array([p[1] for p in pcf_list])
    
    pcf_mean = np.mean(pcfs, axis=0)
    pcf_se = np.std(pcfs, axis=0) / np.sqrt(len(pcfs))
    
    return distances, pcf_mean, pcf_se


def save_results_jsonl(results: List[Dict], output_path: Path):
    """Save results incrementally to JSONL format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, default=str) + "\n")


def save_results_npz(results: List[Dict], output_path: Path):
    """Save results to compressed NPZ format."""
    data = {}
    for i, res in enumerate(results):
        for key, val in res.items():
            data[f"run_{i}_{key}"] = np.array(val)
    np.savez_compressed(output_path, **data)


def load_results_jsonl(input_path: Path) -> List[Dict]:
    """Load results from JSONL format."""
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
    Run a single PP simulation and collect metrics.
    
    Returns dict with population, cluster, PCF, and evolution metrics.
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
        neighborhood="moore", #NOTE: Default neighborhood
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
        model.evolve("prey_death", sd=cfg.evolve_sd, min_val=cfg.evolve_min, max_val=cfg.evolve_max)
    
    # Scale timing with grid size 
    warmup_steps = cfg.get_warmup_steps(grid_size)
    measurement_steps = cfg.get_measurement_steps(grid_size)
    
    
    # Warmup phase
    for _ in range(warmup_steps):
        model.update()

        
    # Measurement phase: start collecting our mertics
    prey_pops, pred_pops = [], [] # Prey populations and predator populations
    evolved_means, evolved_stds = [], [] # Evolution stats over time
    cluster_sizes_prey, cluster_sizes_pred = [], [] # Cluster sizes
    largest_fractions_prey, largest_fractions_pred = [], [] # Largest cluster fractions = size of largest cluster / total population
    pcf_samples = {'prey_prey': [], 'pred_pred': [], 'prey_pred': []}
    
    
    # Determine minimum count for analysis
    min_count = int(cfg.min_density_for_analysis * (grid_size ** 2))
    
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
                cluster_sizes_prey = prey_stats['sizes'].tolist()
                largest_fractions_prey.append(prey_stats['largest_fraction'])
            
            if pred_survived:
                pred_stats = get_cluster_stats_fast(model.grid, 2)
                cluster_sizes_pred = pred_stats['sizes'].tolist()
                largest_fractions_pred.append(pred_stats['largest_fraction'])
            
            # PCF requires both
            if compute_pcf and prey_survived and pred_survived:
                max_dist = min(grid_size / 2, cfg.pcf_max_distance)
                pcf_data = compute_all_pcfs_fast(model.grid, max_dist, cfg.pcf_n_bins)
                pcf_samples['prey_prey'].append(pcf_data['prey_prey'])
                pcf_samples['pred_pred'].append(pcf_data['pred_pred'])
                pcf_samples['prey_pred'].append(pcf_data['prey_pred'])  
                
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
        "prey_largest_fraction": float(np.mean(largest_fractions_prey)) if largest_fractions_prey else np.nan,
        "pred_largest_fraction": float(np.mean(largest_fractions_pred)) if largest_fractions_pred else np.nan,
    }
    
    # Time series (if requested)
    if cfg.save_timeseries:
        subsample = cfg.timeseries_subsample
        result["prey_timeseries"] = prey_pops[::subsample] #NOTE: Sample temporal data every 'subsample' steps
        result["pred_timeseries"] = pred_pops[::subsample]
        
        
    # Evolution statistics
    if with_evolution and evolved_means:
        valid_means = [v for v in evolved_means if not np.isnan(v)]
        result["evolved_prey_death_mean"] = float(np.mean(valid_means)) if valid_means else np.nan
        result["evolved_prey_death_std"] = float(np.mean([v for v in evolved_stds if not np.isnan(v)])) if evolved_stds else np.nan
        result["evolved_prey_death_final"] = valid_means[-1] if valid_means else np.nan
        result["evolved_prey_death_min"] = float(np.min(valid_means)) if valid_means else np.nan
        result["evolved_prey_death_max"] = float(np.max(valid_means)) if valid_means else np.nan
        result["evolve_sd"] = cfg.evolve_sd
        
        if cfg.save_timeseries:
            result["evolved_prey_death_timeseries"] = evolved_means[::cfg.timeseries_subsample]
    
     # PCF statistics
    if pcf_samples['prey_prey']:
        dist, pcf_rr, _ = average_pcfs(pcf_samples['prey_prey'])
        _, pcf_cc, _ = average_pcfs(pcf_samples['pred_pred'])
        _, pcf_cr, _ = average_pcfs(pcf_samples['prey_pred'])
        
        result["pcf_distances"] = dist.tolist()
        result["pcf_prey_prey"] = pcf_rr.tolist()
        result["pcf_pred_pred"] = pcf_cc.tolist()
        result["pcf_prey_pred"] = pcf_cr.tolist()
        
        # Short-range indices
        """
        NOTE: The Pair Correlation function measures spatial correlation at distance r.
            g(r) = 1: random (poisson distribution)
            g(r) > 1: clustering (more pairs than random)
            g(r) < 1: segregation (fewer pairs than random)
        
            prey_clustering_index: Do prey clump together?
            pred_clustering_index: Do predators clump together?
            segregation_index: Are prey and predators segregated?
        
        For the Hydra effect model:
            segregation_index < 1: Prey and predators are spatially separated
            prey_clustering_index > 1: Prey form clusters
            pred_clustering_index > 1: Predators form clusters
            
        High segregation (low segregation index): prey can reproduce in predator-free zones
        High prey clustering: prey form groups that can survive predation
        At criticality: expect sepcific balance where clusters are large enough to sustain but
            fragmented enough to avoid total predation.
            
        If segregation_index = 1 approx, no Hydra effect -> follow mean field dynamics.
        """
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
    Phase 1: Parameter sweep to find critical point.
    
    - 2D sweep of prey_birth prey_death
    - Both with and without evolution
    - Outputs: bifurcation data, cluster distributions
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
            jobs.append((cfg.prey_birth, pd, cfg.predator_birth, cfg.predator_death, 
                        cfg.grid_size, seed, cfg, False))
                
    
    logger.info(f"Phase 1: {len(jobs):,} simulations")
    logger.info(f"  Grid: {cfg.n_prey_death} prey_death values × {cfg.n_replicates} reps (prey_birth={cfg.prey_birth})")
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
    Phase 2: Self-organization analysis.
    
    SOC Hypothesis: Prey evolve toward critical critical point regardless of initial conditions.
    
    NOTE: Test is currently start evo from different intial prey_death values (?)
    If SOC holds, then all runs converge to the same final prey_death near critical point.
    
    FIXME: This run script needs to be adjusted
    """
    from joblib import Parallel, delayed
    
    warmup_numba_kernels(cfg.grid_size, directed_hunting=cfg.directed_hunting)
    
    # Test at multiple prey_birth values
    pb = 0.2
    # Vary intial prey_death
    initial_prey_deaths = np.linspace(cfg.prey_death_range[0], cfg.prey_death_range[1], cfg.n_prey_death)
    
    jobs = []
    for initial_pd in initial_prey_deaths:
        for rep in range(cfg.n_replicates):
            params = {"pb": pb, "initial_pd": initial_pd, "phase": 2}
            seed = generate_unique_seed(params, rep)
            jobs.append((pb, initial_pd, cfg.predator_birth, cfg.predator_death,
                        cfg.grid_size, seed, cfg, True))
    
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
    for L in cfg.grid_sizes: # Sweep through grid sizes
        for rep in range(cfg.n_replicates):
            params = {"L": L, "phase": 3}
            seed = generate_unique_seed(params, rep)
            jobs.append((pb, pd, cfg.predator_birth, cfg.predator_death,
                        L, seed, cfg, False))
    
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
    Phase 4: Global Sensitivity Analysis.
    Vary: prey_birth, prey_death, predator_birth, predator_death
    - prey_death: 10 values from 0.05 to 0.95
    - prey_birth, predator_birth, predator_death: 11 values each from 0 to 1
    Reps: 10
    Grid size: 250
    """
    from joblib import Parallel, delayed
    import itertools
    
    warmup_numba_kernels(cfg.grid_size, directed_hunting=cfg.directed_hunting)
    
    # Define sweep values
    prey_death_values = np.linspace(0.05, 0.95, 10)  # 10 values for prey_death
    other_param_values = np.linspace(0.0, 1.0, 11)   # 11 values for the rest
    
    # Logging
    logger.info(f"Phase 4: Full 4D Parameter Sweep")
    logger.info(f"  prey_death: 10 values from 0.05 to 0.95")
    logger.info(f"  prey_birth, pred_birth, pred_death: 11 values each from 0 to 1")
    logger.info(f"  Grid Size: {cfg.grid_size}")
    logger.info(f"  Replicates: {cfg.n_replicates}")
    
    # Build parameter grid
    param_grid = itertools.product(
        other_param_values,   # prey_birth (11 values)
        prey_death_values,    # prey_death (10 values)
        other_param_values,   # predator_birth (11 values)
        other_param_values    # predator_death (11 values)
    )
    
    jobs = []
    
    for pb, pd, pred_b, pred_d in param_grid:
        for rep in range(cfg.n_replicates):
            params_id = {
                "pb": pb, 
                "pd": pd, 
                "pred_b": pred_b, 
                "pred_d": pred_d, 
                "rep": rep
            }
            seed = generate_unique_seed(params_id, rep)
            
            jobs.append((
                pb,        # prey_birth
                pd,        # prey_death
                pred_b,    # predator_birth
                pred_d,    # predator_death
                cfg.grid_size, 
                seed, 
                cfg, 
                False 
            ))
    
    logger.info(f"  Total simulations: {len(jobs):,}")  # 11 * 10 * 11 * 11 * n_reps = 13,310 * n_reps
    
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
        "parameters_varied": ["prey_birth", "prey_death", "predator_birth", "predator_death"],
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
            jobs.append((pb, pd, cfg.predator_birth, cfg.predator_death,
                        cfg.grid_size, seed, cfg, False))
    
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
    Phase 6: Model Extensions - Directed Hunting Comparison.
    Same 4D sweep as Phase 4, but with directed_hunting=True.
    Vary: prey_birth, prey_death, predator_birth, predator_death
    - prey_death: 10 values from 0.05 to 0.95
    - prey_birth, predator_birth, predator_death: 11 values each from 0 to 1
    Reps: 10
    Grid size: 250
    Compare results with Phase 4 to assess impact of directed hunting on:
    - Critical point location
    - Hydra effect persistence
    - SOC signatures
    """
    from joblib import Parallel, delayed
    import itertools
    
    warmup_numba_kernels(cfg.grid_size, directed_hunting=cfg.directed_hunting)
    
    # Define sweep values (same as Phase 4)
    prey_death_values = np.linspace(0.05, 0.95, 10)  # 10 values for prey_death
    other_param_values = np.linspace(0.0, 1.0, 11)   # 11 values for the rest
    
    # Logging
    logger.info(f"Phase 6: Full 4D Parameter Sweep (Directed Hunting)")
    logger.info(f"  prey_death: 10 values from 0.05 to 0.95")
    logger.info(f"  prey_birth, pred_birth, pred_death: 11 values each from 0 to 1")
    logger.info(f"  Grid Size: {cfg.grid_size}")
    logger.info(f"  Replicates: {cfg.n_replicates}")
    logger.info(f"  Directed Hunting: {cfg.directed_hunting}")
    
    # Build parameter grid
    param_grid = itertools.product(
        other_param_values,   # prey_birth (11 values)
        prey_death_values,    # prey_death (10 values)
        other_param_values,   # predator_birth (11 values)
        other_param_values    # predator_death (11 values)
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
                "rep": rep
            }
            seed = generate_unique_seed(params_id, rep)
            
            jobs.append((
                pb,        # prey_birth
                pd,        # prey_death
                pred_b,    # predator_birth
                pred_d,    # predator_death
                cfg.grid_size, 
                seed, 
                cfg, 
                False 
            ))
    
    logger.info(f"  Total simulations: {len(jobs):,}")  # 11 * 10 * 11 * 11 * n_reps = 13,310 * n_reps
    
    output_jsonl = output_dir / "phase6_results.jsonl"
    all_results = []
    
    with open(output_jsonl, "w", encoding="utf-8") as f:
        executor = Parallel(n_jobs=cfg.n_jobs, return_as="generator")
        tasks = (delayed(run_single_simulation)(*job) for job in jobs)
        
        for result in tqdm(executor(tasks), total=len(jobs), desc="Phase 6 (4D Sweep + Directed)"):
            f.write(json.dumps(result, default=str) + "\n")
            f.flush()
            all_results.append(result)
    
    # Save Metadata
    meta = {
        "phase": 6,
        "description": "Global 4D Sensitivity Analysis with Directed Hunting",
        "prey_death_values": prey_death_values.tolist(),
        "other_param_values": other_param_values.tolist(),
        "parameters_varied": ["prey_birth", "prey_death", "predator_birth", "predator_death"],
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
    parser = argparse.ArgumentParser(
        description="Predator-Prey Hydra Effect Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  1  Parameter sweep to find critical point
  2  Self-organization (evolution toward criticality)
  3  Finite-size scaling at critical point
  4  Sensitivity analysis across parameter regimes
  5  Perturbation analysis (critical slowing down)
  6  Model extensions (directed hunting comparison)
        """
    )
    parser.add_argument("--phase", type=str, required=True,
                       help="Phase to run: 1-6 or 'all'")
    parser.add_argument("--output", type=Path, default=Path("results"),
                       help="Output directory (default: results)")
    parser.add_argument("--cores", type=int, default=-1,
                       help="Number of cores (-1 for all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Estimate runtime without running")
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
        cfg.n_jobs = args.cores if args.cores > 0 else int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
        
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