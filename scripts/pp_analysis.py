#!/usr/bin/env python3
"""
Prey-predator evolutionary analysis - Snellius HPC Version (Optimized)

Focus: Prey Hydra effect - high prey death rates leading to higher prey density.

Optimizations applied:
- Cell-list PCF (O(N) instead of O(N²))
- Pre-allocated kernel buffers
- PCF sampling (compute for subset of runs)
- Consistent dtypes throughout
- Removed redundant code

Usage:
    python pp_analysis.py --mode full          # Run everything
    python pp_analysis.py --mode sweep         # Only 2D sweep
    python pp_analysis.py --mode sensitivity   # Only evolution sensitivity
    python pp_analysis.py --mode fss           # Only finite-size scaling
    python pp_analysis.py --mode plot          # Only generate plots from saved data
    python pp_analysis.py --mode debug         # Interactive visualization (local only)
    python pp_analysis.py --dry-run            # Estimate runtime without running
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

project_root = str(Path(__file__).parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# Import optimized Numba functions
try:
    from scripts.numba_optimized import (
        compute_pcf_periodic_fast,
        compute_all_pcfs_fast,
        measure_cluster_sizes_fast,
        warmup_numba_kernels,
        set_numba_seed,
        NUMBA_AVAILABLE,
    )
    USE_NUMBA = NUMBA_AVAILABLE
except ImportError:
    USE_NUMBA = False
    def warmup_numba_kernels(size): pass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Central configuration for analysis."""
    
    # Grid settings
    default_grid: int = 100
    densities: Tuple[float, float] = (0.30, 0.15)
    
    # 2D sweep resolution
    n_prey_birth: int = 15
    n_prey_death: int = 15
    prey_birth_min: float = 0.10
    prey_birth_max: float = 0.35
    prey_death_min: float = 0.001
    prey_death_max: float = 0.10
    
    # Fixed predator parameters
    predator_death: float = 0.1
    predator_birth: float = 0.2
    
    # Replicates
    n_replicates: int = 50
    
    # Simulation timing
    warmup_steps: int = 200
    measurement_steps: int = 300
    
    # Cluster/PCF sampling
    cluster_samples: int = 1  # Reduced from 3 - PCF is expensive
    cluster_interval: int = 299  # Sample near end of measurement
    
    # PCF settings
    collect_pcf: bool = True
    pcf_sample_rate: float = 0.2  # Only compute PCF for 20% of runs
    pcf_max_distance: float = 20.0
    pcf_n_bins: int = 20
    
    # Evolution parameters
    evolve_sd: float = 0.10
    evolve_min: float = 0.001
    evolve_max: float = 0.10
    
    # Finite size scaling
    fss_grid_sizes: Tuple[int, ...] = (50, 75, 100, 150)
    fss_replicates: int = 100
    
    # Evolution sensitivity analysis
    sensitivity_sd_values: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20)
    sensitivity_replicates: int = 20
    
    # Update mode
    synchronous: bool = False
    
    # Diagnostic snapshots
    save_diagnostic_plots: bool = False
    diagnostic_param_sets: int = 5
    
    # Parallelization
    n_jobs: int = -1
    
    def get_prey_deaths(self) -> np.ndarray:
        return np.linspace(self.prey_death_min, self.prey_death_max, self.n_prey_death)
    
    def get_prey_births(self) -> np.ndarray:
        return np.linspace(self.prey_birth_min, self.prey_birth_max, self.n_prey_birth)
    
    def estimate_runtime(self, n_cores: int = 32) -> str:
        """Estimate total runtime."""
        n_sweep = self.n_prey_birth * self.n_prey_death * self.n_replicates * 2
        n_sens = len(self.sensitivity_sd_values) * self.sensitivity_replicates
        
        # Base time per simulation (with optimized PCF)
        base_time_s = 0.8  # ~800ms with cell-list PCF
        pcf_overhead = 0.1 if self.collect_pcf else 0  # Much smaller with cell-list
        
        # FSS with size scaling
        fss_time = 0
        for L in self.fss_grid_sizes:
            scale = (L / self.default_grid) ** 2
            warmup_scale = L / self.default_grid
            fss_time += self.fss_replicates * base_time_s * scale * warmup_scale
        
        sweep_time = n_sweep * (base_time_s + pcf_overhead * self.pcf_sample_rate)
        sens_time = n_sens * base_time_s
        
        total_seconds = (sweep_time + sens_time + fss_time) / n_cores
        hours = total_seconds / 3600
        core_hours = (sweep_time + sens_time + fss_time) / 3600
        
        n_total = n_sweep + n_sens + sum(self.fss_replicates for _ in self.fss_grid_sizes)
        return f"{n_total:,} sims, ~{hours:.1f}h on {n_cores} cores (~{core_hours:.0f} core-hours)"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_populations(grid: np.ndarray) -> Tuple[int, int, int]:
    """Count empty, prey, predator cells."""
    return int(np.sum(grid == 0)), int(np.sum(grid == 1)), int(np.sum(grid == 2))


def get_evolved_stats(model, param: str) -> Dict:
    """Get statistics of evolved parameter from model."""
    arr = model.cell_params.get(param)
    if arr is None:
        return {"mean": np.nan, "std": np.nan, "n": 0}
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"mean": np.nan, "std": np.nan, "n": 0}
    return {"mean": float(np.mean(valid)), "std": float(np.std(valid)), "n": len(valid)}


def truncated_power_law(s: np.ndarray, tau: float, s_c: float, A: float) -> np.ndarray:
    """Truncated power law: P(s) = A * s^(-tau) * exp(-s/s_c)."""
    return A * np.power(s, -tau) * np.exp(-s / s_c)


def fit_truncated_power_law(sizes: np.ndarray, s_min: int = 2) -> Dict:
    """Fit truncated power law to cluster size distribution."""
    sizes = sizes[sizes >= s_min]
    if len(sizes) < 100:
        return {"tau": np.nan, "s_c": np.nan, "valid": False, "n": len(sizes)}
    
    bins = np.logspace(np.log10(s_min), np.log10(sizes.max() + 1), 25)
    hist, edges = np.histogram(sizes, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    
    mask = hist > 0
    x, y = centers[mask], hist[mask]
    
    if len(x) < 5:
        return {"tau": np.nan, "s_c": np.nan, "valid": False, "n": len(sizes)}
    
    try:
        popt, pcov = curve_fit(
            lambda s, tau, s_c, A: np.log(truncated_power_law(s, tau, s_c, A) + 1e-20),
            x, np.log(y + 1e-20),
            p0=[2.0, 1000.0, y[0] * x[0] ** 2],
            bounds=([1.0, 10, 1e-15], [4.0, 50000, 1e10]),
            maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "tau": popt[0],
            "tau_se": perr[0],
            "s_c": popt[1],
            "valid": True,
            "n": len(sizes),
        }
    except Exception:
        return {"tau": np.nan, "s_c": np.nan, "valid": False, "n": len(sizes)}


def average_pcfs(pcf_list: List[Tuple[np.ndarray, np.ndarray, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average multiple PCF measurements with standard error."""
    if len(pcf_list) == 0:
        return np.array([]), np.array([]), np.array([])
    
    distances = pcf_list[0][0]
    pcfs = np.array([p[1] for p in pcf_list])
    
    pcf_mean = np.mean(pcfs, axis=0)
    pcf_se = np.std(pcfs, axis=0) / np.sqrt(len(pcfs))
    
    return distances, pcf_mean, pcf_se


def save_sweep_binary(results: List[Dict], output_path: Path):
    """Save sweep results to compressed .npz format."""
    data_to_save = {}
    for i, res in enumerate(results):
        prefix = f"run_{i}_"
        for key, val in res.items():
            data_to_save[f"{prefix}{key}"] = np.array(val)
    np.savez_compressed(output_path, **data_to_save)


def load_sweep_binary(input_path: Path) -> List[Dict]:
    """Load sweep results from .npz format."""
    data = np.load(input_path, allow_pickle=True)
    
    # Reconstruct results list
    results = {}
    for key in data.keys():
        parts = key.split("_", 2)
        run_idx = int(parts[1])
        field = parts[2]
        
        if run_idx not in results:
            results[run_idx] = {}
        
        val = data[key]
        # Convert 0-d arrays back to scalars
        if val.ndim == 0:
            val = val.item()
        else:
            val = val.tolist()
        results[run_idx][field] = val
    
    return [results[i] for i in sorted(results.keys())]


# =============================================================================
# SIMULATION FUNCTION
# =============================================================================

def run_single_simulation(
    prey_birth: float,
    prey_death: float,
    grid_size: int,
    seed: int,
    with_evolution: bool,
    cfg: Config,
    evolve_sd: Optional[float] = None,
    evolve_min: Optional[float] = None,
    evolve_max: Optional[float] = None,
    compute_pcf: Optional[bool] = None,
) -> Dict:
    """
    Run a single PP simulation and collect metrics.
    
    Args:
        prey_birth: Prey birth rate
        prey_death: Prey death rate (initial if evolving)
        grid_size: Grid dimension (L x L)
        seed: Random seed
        with_evolution: Whether to evolve prey_death
        cfg: Configuration object
        evolve_sd: Evolution SD (overrides cfg if provided)
        evolve_min: Evolution min (overrides cfg if provided)
        evolve_max: Evolution max (overrides cfg if provided)
        compute_pcf: Whether to compute PCF (overrides cfg sampling if provided)
    
    Returns:
        Dictionary with simulation results
    """
    from models.CA import PP
    # Seed both RNGs
    np.random.seed(seed)
    if NUMBA_AVAILABLE:
        set_numba_seed(seed)
    
    # Set evolution parameters
    if evolve_sd is None:
        evolve_sd = cfg.evolve_sd
    if evolve_min is None:
        evolve_min = cfg.evolve_min
    if evolve_max is None:
        evolve_max = cfg.evolve_max
    
    # Determine if we compute PCF this run
    if compute_pcf is None:
        compute_pcf = cfg.collect_pcf and (np.random.random() < cfg.pcf_sample_rate)
    
    # Initialize model
    params = {
        "prey_birth": prey_birth,
        "prey_death": prey_death,
        "predator_death": cfg.predator_death,
        "predator_birth": cfg.predator_birth,
    }
    
    model = PP(
        rows=grid_size,
        cols=grid_size,
        densities=cfg.densities,
        neighborhood="moore",
        params=params,
        seed=seed,
        synchronous=cfg.synchronous,
    )
    
    if with_evolution:
        model.evolve("prey_death", sd=evolve_sd, min_val=evolve_min, max_val=evolve_max)
    
    # Warmup
    model.run(cfg.warmup_steps)
    
    # Measurement phase
    prey_pops, pred_pops, evolved_vals = [], [], []
    prey_clusters, pred_clusters = [], []
    pcf_samples = {'prey_prey': [], 'pred_pred': [], 'prey_pred': []}
    
    sample_counter = 0
    
    for step in range(cfg.measurement_steps):
        model.update()
        _, prey, pred = count_populations(model.grid)
        prey_pops.append(prey)
        pred_pops.append(pred)
        
        # Track evolved parameter
        if with_evolution:
            stats = get_evolved_stats(model, "prey_death")
            evolved_vals.append(stats["mean"])
        
        # Cluster and PCF sampling
        if step >= cfg.cluster_interval and sample_counter < cfg.cluster_samples:
            if prey > 10:
                prey_clusters.extend(measure_cluster_sizes_fast(model.grid, 1))
                pred_clusters.extend(measure_cluster_sizes_fast(model.grid, 2))
                
                # Compute PCFs if enabled for this run
                if compute_pcf and prey > 20 and pred > 5:
                    max_dist = min(grid_size / 2, cfg.pcf_max_distance)
                    pcf_data = compute_all_pcfs_fast(model.grid, max_dist, cfg.pcf_n_bins)
                    pcf_samples['prey_prey'].append(pcf_data['prey_prey'])
                    pcf_samples['pred_pred'].append(pcf_data['pred_pred'])
                    pcf_samples['prey_pred'].append(pcf_data['prey_pred'])
            
            sample_counter += 1
    
    # Compile results
    result = {
        "prey_birth": prey_birth,
        "prey_death": prey_death,
        "grid_size": grid_size,
        "with_evolution": with_evolution,
        "seed": seed,
        "prey_mean": float(np.mean(prey_pops)),
        "prey_std": float(np.std(prey_pops)),
        "pred_mean": float(np.mean(pred_pops)),
        "pred_std": float(np.std(pred_pops)),
        "prey_survived": bool(np.mean(prey_pops) > 10),
        "pred_survived": bool(np.mean(pred_pops) > 10),
        "prey_n_clusters": len(prey_clusters),
        "pred_n_clusters": len(pred_clusters),
    }
    
    # Evolved parameter statistics
    if with_evolution and evolved_vals:
        valid_evolved = [v for v in evolved_vals if not np.isnan(v)]
        result["evolved_prey_death_mean"] = float(np.mean(valid_evolved)) if valid_evolved else np.nan
        result["evolved_prey_death_std"] = float(np.std(valid_evolved)) if valid_evolved else np.nan
        result["evolve_sd"] = evolve_sd
        
        # Final state
        if valid_evolved:
            result["evolved_prey_death_final"] = valid_evolved[-1]
    
    # Cluster fits
    if len(prey_clusters) > 50:
        fit = fit_truncated_power_law(np.array(prey_clusters))
        result["prey_tau"] = fit["tau"]
        result["prey_s_c"] = fit["s_c"]
    else:
        result["prey_tau"] = np.nan
        result["prey_s_c"] = np.nan
    
    if len(pred_clusters) > 50:
        fit = fit_truncated_power_law(np.array(pred_clusters))
        result["pred_tau"] = fit["tau"]
        result["pred_s_c"] = fit["s_c"]
    else:
        result["pred_tau"] = np.nan
        result["pred_s_c"] = np.nan
    
    # PCF statistics
    if len(pcf_samples['prey_prey']) > 0:
        dist, pcf_rr_mean, pcf_rr_se = average_pcfs(pcf_samples['prey_prey'])
        dist, pcf_cc_mean, pcf_cc_se = average_pcfs(pcf_samples['pred_pred'])
        dist, pcf_cr_mean, pcf_cr_se = average_pcfs(pcf_samples['prey_pred'])
        
        result["pcf_distances"] = dist.tolist()
        result["pcf_prey_prey_mean"] = pcf_rr_mean.tolist()
        result["pcf_pred_pred_mean"] = pcf_cc_mean.tolist()
        result["pcf_prey_pred_mean"] = pcf_cr_mean.tolist()
        
        # Summary indices (short-range structure)
        short_dist_mask = dist < 3.0
        if np.any(short_dist_mask):
            result["segregation_index"] = float(np.mean(pcf_cr_mean[short_dist_mask]))
            result["prey_clustering_index"] = float(np.mean(pcf_rr_mean[short_dist_mask]))
            result["pred_clustering_index"] = float(np.mean(pcf_cc_mean[short_dist_mask]))
    else:
        result["segregation_index"] = np.nan
        result["prey_clustering_index"] = np.nan
        result["pred_clustering_index"] = np.nan
    
    return result


def run_single_simulation_fss(
    prey_birth: float,
    prey_death: float,
    grid_size: int,
    seed: int,
    cfg: Config,
    warmup_steps: int,
    measurement_steps: int,
) -> Dict:
    """FSS-specific simulation with size-scaled equilibration time."""
    from models.CA import PP
    
    np.random.seed(seed)
    if NUMBA_AVAILABLE:
        set_numba_seed(seed)
    
    params = {
        "prey_birth": prey_birth,
        "prey_death": prey_death,
        "predator_death": cfg.predator_death,
        "predator_birth": cfg.predator_birth,
    }
    
    model = PP(
        rows=grid_size,
        cols=grid_size,
        densities=cfg.densities,
        neighborhood="moore",
        params=params,
        seed=seed,
        synchronous=cfg.synchronous,
    )
    
    # No evolution for FSS - studying baseline spatial structure
    model.run(warmup_steps)
    
    # Measurement
    prey_pops, pred_pops = [], []
    prey_clusters, pred_clusters = [], []
    
    cluster_interval = max(1, int(cfg.cluster_interval * grid_size / cfg.default_grid))
    sample_counter = 0
    
    for step in range(measurement_steps):
        model.update()
        _, prey, pred = count_populations(model.grid)
        prey_pops.append(prey)
        pred_pops.append(pred)
        
        if step % cluster_interval == 0 and sample_counter < cfg.cluster_samples:
            if prey > 10:
                prey_clusters.extend(measure_cluster_sizes_fast(model.grid, 1))
                pred_clusters.extend(measure_cluster_sizes_fast(model.grid, 2))
            sample_counter += 1
    
    result = {
        "prey_birth": prey_birth,
        "prey_death": prey_death,
        "grid_size": grid_size,
        "seed": seed,
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "prey_mean": float(np.mean(prey_pops)),
        "prey_std": float(np.std(prey_pops)),
        "pred_mean": float(np.mean(pred_pops)),
        "pred_std": float(np.std(pred_pops)),
        "prey_survived": bool(np.mean(prey_pops) > 10),
        "pred_survived": bool(np.mean(pred_pops) > 10),
    }
    
    # Cluster fits
    if len(prey_clusters) > 50:
        fit = fit_truncated_power_law(np.array(prey_clusters))
        result["prey_tau"] = fit["tau"]
        result["prey_tau_se"] = fit.get("tau_se", np.nan)
        result["prey_s_c"] = fit["s_c"]
    else:
        result["prey_tau"] = np.nan
        result["prey_tau_se"] = np.nan
        result["prey_s_c"] = np.nan
    
    if len(pred_clusters) > 50:
        fit = fit_truncated_power_law(np.array(pred_clusters))
        result["pred_tau"] = fit["tau"]
        result["pred_tau_se"] = fit.get("tau_se", np.nan)
        result["pred_s_c"] = fit["s_c"]
    else:
        result["pred_tau"] = np.nan
        result["pred_tau_se"] = np.nan
        result["pred_s_c"] = np.nan
    
    return result

# =============================================================================
# ANALYSIS RUNNERS
# =============================================================================

def run_2d_sweep(cfg: Config, output_dir: Path, logger: logging.Logger) -> List[Dict]:
    """Run full 2D parameter sweep."""
    from joblib import Parallel, delayed
    
    if USE_NUMBA:
        warmup_numba_kernels(cfg.default_grid)
    
    prey_births = cfg.get_prey_births()
    prey_deaths = cfg.get_prey_deaths()
    
    # Build job list
    jobs = []
    for pb in prey_births:
        for pd in prey_deaths:
            for rep in range(cfg.n_replicates):
                seed_base = int(pb * 1000) + int(pd * 10000) + rep
                # Both with and without evolution
                jobs.append((pb, pd, cfg.default_grid, seed_base, False))
                jobs.append((pb, pd, cfg.default_grid, seed_base, True))
    
    logger.info(f"2D Sweep: {len(jobs):,} simulations")
    logger.info(f"  Grid: {len(prey_births)}×{len(prey_deaths)} parameters")
    logger.info(f"  prey_birth: [{cfg.prey_birth_min:.3f}, {cfg.prey_birth_max:.3f}]")
    logger.info(f"  prey_death: [{cfg.prey_death_min:.3f}, {cfg.prey_death_max:.3f}]")
    logger.info(f"  Replicates: {cfg.n_replicates}")
    logger.info(f"  PCF sample rate: {cfg.pcf_sample_rate:.0%}")
    
    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(run_single_simulation)(pb, pd, gs, seed, evo, cfg)
        for pb, pd, gs, seed, evo in jobs
    )
    
    # Save results
    output_file = output_dir / "sweep_results.npz"
    save_sweep_binary(results, output_file)
    
    meta = {
        "n_sims": len(results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "grid_size": cfg.default_grid,
        "pcf_sample_rate": cfg.pcf_sample_rate,
    }
    with open(output_dir / "sweep_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"Saved sweep results to {output_file}")
    return results


def run_sensitivity(cfg: Config, output_dir: Path, logger: logging.Logger) -> List[Dict]:
    """Run evolution parameter sensitivity analysis."""
    from joblib import Parallel, delayed
    
    # Fixed parameters in transition zone
    pb_test = 0.20
    pd_test = 0.05
    
    jobs = []
    for sd in cfg.sensitivity_sd_values:
        for rep in range(cfg.sensitivity_replicates):
            seed = int(sd * 100000) + rep
            jobs.append((pb_test, pd_test, cfg.default_grid, seed, True, sd))
    
    logger.info(f"Sensitivity: {len(jobs)} simulations")
    logger.info(f"  SD values: {cfg.sensitivity_sd_values}")
    
    results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(run_single_simulation)(pb, pd, gs, seed, evo, cfg, evolve_sd=sd, compute_pcf=True)
        for pb, pd, gs, seed, evo, sd in jobs
    )
    
    output_file = output_dir / "sensitivity_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f)
    
    logger.info(f"Saved to {output_file}")
    return results


def run_fss(cfg: Config, output_dir: Path, logger: logging.Logger) -> List[Dict]:
    """Run finite-size scaling analysis."""
    from joblib import Parallel, delayed
    
    # Fixed parameters near critical point
    pb_test = 0.20
    pd_test = 0.03
    
    # Validation
    logger.info("=" * 60)
    logger.info("FSS PARAMETER VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Testing: prey_birth={pb_test}, prey_death={pd_test}")
    
    test_results = []
    for rep in range(5):
        result = run_single_simulation(
            pb_test, pd_test, cfg.default_grid, 10000 + rep, False, cfg, compute_pcf=False
        )
        test_results.append(result)
    
    tau_vals = [r["prey_tau"] for r in test_results if not np.isnan(r.get("prey_tau", np.nan))]
    if tau_vals:
        tau_test = np.mean(tau_vals)
        logger.info(f"  Validation τ = {tau_test:.3f} (target: ~2.05)")
        if abs(tau_test - 2.05) > 0.3:
            logger.warning("  Parameters may not be near critical point!")
    
    # Generate jobs with size-scaled equilibration
    jobs = []
    for L in cfg.fss_grid_sizes:
        warmup_factor = L / cfg.default_grid
        warmup_steps = int(cfg.warmup_steps * warmup_factor)
        measurement_steps = int(cfg.measurement_steps * warmup_factor)
        
        for rep in range(cfg.fss_replicates):
            seed = L * 1000 + rep
            jobs.append((pb_test, pd_test, L, seed, warmup_steps, measurement_steps))
    
    logger.info(f"FSS: {len(jobs)} simulations")
    logger.info(f"  Grid sizes: {cfg.fss_grid_sizes}")
    
    results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(run_single_simulation_fss)(pb, pd, gs, seed, cfg, ws, ms)
        for pb, pd, gs, seed, ws, ms in jobs
    )
    
    output_file = output_dir / "fss_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f)
    
    logger.info(f"Saved to {output_file}")
    return results


def run_debug_mode(cfg: Config, logger: logging.Logger):
    """Run interactive visualization for debugging."""
    logger.info("=" * 60)
    logger.info("DEBUG MODE - Interactive Visualization")
    logger.info("=" * 60)
    
    try:
        from models.CA import PP
    except ImportError:
        logger.error("Cannot import PP model.")
        return
    
    grid_size = 50
    params = {
        "prey_birth": 0.20,
        "prey_death": 0.05,
        "predator_death": cfg.predator_death,
        "predator_birth": cfg.predator_birth,
    }
    
    logger.info(f"Parameters: {params}")
    logger.info(f"Grid: {grid_size}×{grid_size}")
    
    model = PP(
        rows=grid_size,
        cols=grid_size,
        densities=cfg.densities,
        neighborhood="moore",
        params=params,
        seed=42,
        synchronous=cfg.synchronous,
    )
    
    model.evolve("prey_death", sd=cfg.evolve_sd, min_val=cfg.evolve_min, max_val=cfg.evolve_max)
    
    logger.info("Starting visualization...")
    model.visualize(
        interval=5,
        figsize=(16, 10),
        pause=0.01,
        show_cell_params=True,
        show_neighbors=True,
        downsample=1,
    )
    
    model.run(500)
    logger.info("Simulation complete.")
    input("Press Enter to exit...")

# =============================================================================
# PLOTTING
# =============================================================================

def generate_plots(cfg: Config, output_dir: Path, logger: logging.Logger):
    """Generate all analysis plots from saved data."""
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    plt.rcParams["figure.figsize"] = (14, 10)
    plt.rcParams["font.size"] = 11
    
    prey_births = cfg.get_prey_births()
    prey_deaths = cfg.get_prey_deaths()
    n_pb, n_pd = len(prey_births), len(prey_deaths)
    extent = [prey_births[0], prey_births[-1], prey_deaths[0], prey_deaths[-1]]
    
    # Load sweep results
    sweep_file = output_dir / "sweep_results.npz"
    if not sweep_file.exists():
        # Try JSON fallback
        sweep_file = output_dir / "sweep_results.json"
        if not sweep_file.exists():
            logger.error(f"Sweep results not found")
            return
        with open(sweep_file, "r") as f:
            results = json.load(f)
    else:
        results = load_sweep_binary(sweep_file)
    
    logger.info(f"Loaded {len(results)} results")
    
    # Initialize grids
    grids = {
        "prey_pop_no_evo": np.full((n_pd, n_pb), np.nan),
        "prey_pop_evo": np.full((n_pd, n_pb), np.nan),
        "pred_pop_no_evo": np.full((n_pd, n_pb), np.nan),
        "pred_pop_evo": np.full((n_pd, n_pb), np.nan),
        "survival_prey_no_evo": np.full((n_pd, n_pb), np.nan),
        "survival_prey_evo": np.full((n_pd, n_pb), np.nan),
        "tau_prey": np.full((n_pd, n_pb), np.nan),
        "evolved_prey_death": np.full((n_pd, n_pb), np.nan),
        "segregation_index": np.full((n_pd, n_pb), np.nan),
        "prey_clustering_index": np.full((n_pd, n_pb), np.nan),
    }
    
    # Group by parameters
    grouped = defaultdict(list)
    for r in results:
        key = (round(r["prey_birth"], 4), round(r["prey_death"], 4), r["with_evolution"])
        grouped[key].append(r)
    
    # Aggregate into grids
    for i, pd in enumerate(prey_deaths):
        for j, pb in enumerate(prey_births):
            pd_r, pb_r = round(pd, 4), round(pb, 4)
            
            # No evolution
            no_evo = grouped.get((pb_r, pd_r, False), [])
            if no_evo:
                grids["prey_pop_no_evo"][i, j] = np.mean([r["prey_mean"] for r in no_evo])
                grids["pred_pop_no_evo"][i, j] = np.mean([r["pred_mean"] for r in no_evo])
                grids["survival_prey_no_evo"][i, j] = np.mean([r["prey_survived"] for r in no_evo]) * 100
                
                taus = [r["prey_tau"] for r in no_evo if not np.isnan(r.get("prey_tau", np.nan))]
                if taus:
                    grids["tau_prey"][i, j] = np.mean(taus)
                
                seg = [r.get("segregation_index", np.nan) for r in no_evo]
                seg = [s for s in seg if not np.isnan(s)]
                if seg:
                    grids["segregation_index"][i, j] = np.mean(seg)
                
                clust = [r.get("prey_clustering_index", np.nan) for r in no_evo]
                clust = [c for c in clust if not np.isnan(c)]
                if clust:
                    grids["prey_clustering_index"][i, j] = np.mean(clust)
            
            # With evolution
            evo = grouped.get((pb_r, pd_r, True), [])
            if evo:
                grids["prey_pop_evo"][i, j] = np.mean([r["prey_mean"] for r in evo])
                grids["pred_pop_evo"][i, j] = np.mean([r["pred_mean"] for r in evo])
                grids["survival_prey_evo"][i, j] = np.mean([r["prey_survived"] for r in evo]) * 100
                
                evolved = [r.get("evolved_prey_death_mean", np.nan) for r in evo]
                evolved = [e for e in evolved if not np.isnan(e)]
                if evolved:
                    grids["evolved_prey_death"][i, j] = np.mean(evolved)
    
    # Compute Hydra derivative
    dd = prey_deaths[1] - prey_deaths[0]
    dN_dd_no_evo = np.zeros_like(grids["prey_pop_no_evo"])
    dN_dd_evo = np.zeros_like(grids["prey_pop_evo"])
    
    for j in range(n_pb):
        pop_smooth = gaussian_filter1d(grids["prey_pop_no_evo"][:, j], sigma=0.8)
        dN_dd_no_evo[:, j] = np.gradient(pop_smooth, dd)
        pop_smooth = gaussian_filter1d(grids["prey_pop_evo"][:, j], sigma=0.8)
        dN_dd_evo[:, j] = np.gradient(pop_smooth, dd)
    
    # =========================================================================
    # PLOT 1: Phase Diagrams
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    ax = axes[0, 0]
    im = ax.imshow(grids["prey_pop_no_evo"], origin="lower", aspect="auto",
                   extent=extent, cmap="YlGn")
    ax.contour(prey_births, prey_deaths, grids["survival_prey_no_evo"],
               levels=[50], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="Population")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Prey Pop (No Evolution)")
    
    ax = axes[0, 1]
    im = ax.imshow(grids["prey_pop_evo"], origin="lower", aspect="auto",
                   extent=extent, cmap="YlGn")
    ax.contour(prey_births, prey_deaths, grids["survival_prey_evo"],
               levels=[50], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="Population")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Prey Pop (With Evolution)")
    
    ax = axes[0, 2]
    advantage = np.where(
        grids["prey_pop_no_evo"] > 10,
        (grids["prey_pop_evo"] - grids["prey_pop_no_evo"]) / grids["prey_pop_no_evo"] * 100,
        np.where(grids["prey_pop_evo"] > 10, 500, 0),
    )
    im = ax.imshow(np.clip(advantage, -50, 200), origin="lower", aspect="auto",
                   extent=extent, cmap="RdYlGn", vmin=-50, vmax=200)
    plt.colorbar(im, ax=ax, label="Advantage (%)")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Evolution Advantage")
    
    ax = axes[1, 0]
    im = ax.imshow(grids["tau_prey"], origin="lower", aspect="auto",
                   extent=extent, cmap="coolwarm", vmin=1.5, vmax=2.5)
    ax.contour(prey_births, prey_deaths, grids["tau_prey"],
               levels=[2.05], colors="green", linewidths=2)
    plt.colorbar(im, ax=ax, label="τ")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Prey τ (Green: τ=2.05)")
    
    ax = axes[1, 1]
    im = ax.imshow(grids["evolved_prey_death"], origin="lower", aspect="auto",
                   extent=extent, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Evolved d")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Initial Prey Death Rate")
    ax.set_title("Evolved Prey Death Rate")
    
    ax = axes[1, 2]
    im = ax.imshow(dN_dd_no_evo, origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu_r", vmin=-5000, vmax=5000)
    ax.contour(prey_births, prey_deaths, dN_dd_no_evo,
               levels=[0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="dN/dd")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("HYDRA: dN/dd (Red: Prey ↑ with mortality)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "phase_diagrams.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved phase_diagrams.png")
    
    # =========================================================================
    # PLOT 2: Hydra Analysis
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    im = ax.imshow(dN_dd_no_evo, origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu_r", vmin=-5000, vmax=5000)
    ax.contour(prey_births, prey_deaths, dN_dd_no_evo,
               levels=[0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="dN/dd")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Hydra (No Evolution)")
    
    ax = axes[1]
    im = ax.imshow(dN_dd_evo, origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu_r", vmin=-5000, vmax=5000)
    ax.contour(prey_births, prey_deaths, dN_dd_evo,
               levels=[0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="dN/dd")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Hydra (With Evolution)")
    
    ax = axes[2]
    mid_pb_idx = n_pb // 2
    ax.plot(prey_deaths, grids["prey_pop_no_evo"][:, mid_pb_idx], 'b-o',
            label=f'No Evo (pb={prey_births[mid_pb_idx]:.2f})', markersize=4)
    ax.plot(prey_deaths, grids["prey_pop_evo"][:, mid_pb_idx], 'g-s',
            label=f'With Evo (pb={prey_births[mid_pb_idx]:.2f})', markersize=4)
    ax.set_xlabel("Prey Death Rate")
    ax.set_ylabel("Prey Population")
    ax.set_title("Prey Pop vs Death Rate Slice")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "hydra_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved hydra_analysis.png")
    
    # =========================================================================
    # PLOT 3: PCF Analysis
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    im = ax.imshow(grids["segregation_index"], origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu", vmin=0.5, vmax=1.5)
    ax.contour(prey_births, prey_deaths, grids["segregation_index"],
               levels=[1.0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="C_cr")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Segregation Index")
    
    ax = axes[1]
    im = ax.imshow(grids["prey_clustering_index"], origin="lower", aspect="auto",
                   extent=extent, cmap="Greens", vmin=1.0, vmax=3.0)
    plt.colorbar(im, ax=ax, label="C_rr")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Prey Clustering")
    
    ax = axes[2]
    im = ax.imshow(grids["segregation_index"], origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu", vmin=0.5, vmax=1.5, alpha=0.8)
    ax.contour(prey_births, prey_deaths, dN_dd_no_evo,
               levels=[0], colors="lime", linewidths=3)
    ax.contour(prey_births, prey_deaths, grids["survival_prey_no_evo"],
               levels=[50], colors="black", linewidths=2, linestyles='--')
    plt.colorbar(im, ax=ax, label="C_cr")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Segregation + Hydra Boundary")
    
    plt.tight_layout()
    plt.savefig(output_dir / "pcf_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved pcf_analysis.png")
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    summary = {
        "coexistence_no_evo": int(np.sum((grids["survival_prey_no_evo"] > 80))),
        "hydra_region_size": int(np.sum((dN_dd_no_evo > 0) & (grids["prey_pop_no_evo"] > 50))),
        "max_hydra_strength": float(np.nanmax(dN_dd_no_evo)),
        "hydra_region_size_evo": int(np.sum((dN_dd_evo > 0) & (grids["prey_pop_evo"] > 50))),
        "mean_segregation_index": float(np.nanmean(grids["segregation_index"])),
        "mean_prey_clustering": float(np.nanmean(grids["prey_clustering_index"])),
    }
    
    # Find critical point
    dist_crit = np.abs(grids["tau_prey"] - 2.05)
    if not np.all(np.isnan(dist_crit)):
        min_idx = np.unravel_index(np.nanargmin(dist_crit), dist_crit.shape)
        summary["critical_prey_birth"] = float(prey_births[min_idx[1]])
        summary["critical_prey_death"] = float(prey_deaths[min_idx[0]])
        summary["critical_tau_prey"] = float(grids["tau_prey"][min_idx])
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Hydra region: {summary['hydra_region_size']} combinations")
    logger.info(f"Max Hydra strength: {summary['max_hydra_strength']:.1f}")
    logger.info(f"Mean segregation index: {summary['mean_segregation_index']:.3f}")
    if "critical_prey_birth" in summary:
        logger.info(f"Critical point: pb={summary['critical_prey_birth']:.3f}, pd={summary['critical_prey_death']:.3f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PP Evolutionary Analysis - Optimized")
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "sweep", "sensitivity", "fss", "plot", "debug"])
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--cores", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync", action="store_true", dest="synchronous")
    args = parser.parse_args()
    
    # Setup
    cfg = Config()
    cfg.synchronous = args.synchronous
    cfg.n_jobs = args.cores if args.cores > 0 else int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "analysis.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    
    # Header
    logger.info("=" * 60)
    logger.info("PP Evolutionary Analysis - OPTIMIZED VERSION")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Cores: {cfg.n_jobs}")
    logger.info(f"Numba: {'ENABLED' if USE_NUMBA else 'DISABLED'}")
    
    if args.mode == "debug":
        run_debug_mode(cfg, logger)
        return
    
    n_cores = cfg.n_jobs if cfg.n_jobs > 0 else os.cpu_count()
    logger.info(f"Estimated: {cfg.estimate_runtime(n_cores)}")
    
    if args.dry_run:
        logger.info("Dry run - exiting")
        return
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    
    start_time = time.time()
    
    # Run analyses
    if args.mode in ["full", "sweep"]:
        run_2d_sweep(cfg, output_dir, logger)
    
    if args.mode in ["full", "sensitivity"]:
        run_sensitivity(cfg, output_dir, logger)
    
    if args.mode in ["full", "fss"]:
        run_fss(cfg, output_dir, logger)
    
    if args.mode in ["full", "plot"]:
        generate_plots(cfg, output_dir, logger)
    
    elapsed = time.time() - start_time
    logger.info(f"Total runtime: {elapsed/3600:.2f} hours")
    logger.info("Done!")


if __name__ == "__main__":
    main()