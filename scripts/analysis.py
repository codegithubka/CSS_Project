#!/usr/bin/env python3
"""
Post-analysis plotting for predator-prey evolutionary simulations.

Reads saved results from pp_analysis.py and generate figures.
Designed to run locally (not on HPC) for fast iteration.

Usage:
    python plot_pp_results.py results/                    # All plots
    python plot_pp_results.py results/ --phase-only       # Just phase diagrams
    python plot_pp_results.py results/ --hydra-only       # Just Hydra analysis
    python plot_pp_results.py results/ --pcf-only         # Just PCF analysis
    python plot_pp_results.py results/ --fss-only         # Just FSS plots
    python plot_pp_results.py results/ --bifurcation-only # Just bifurcation diagram
    python plot_pp_results.py results/ --phase2-only      # Just Phase 2 SOC analysis
    python plot_pp_results.py results/ --phase3-only      # Just Phase 3 FSS analysis
    python plot_pp_results.py results/ --phase4-only      # Just Phase 4 sensitivity analysis
    python plot_pp_results.py results/ --dpi 300          # High-res for publication
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress

# Configure matplotlib for publication-quality output
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sweep_results(results_dir: Path) -> List[Dict]:
    """Load sweep results from NPZ (preferred) or JSON fallback."""
    npz_file = results_dir / "sweep_results.npz"
    json_file = results_dir / "sweep_results.json"
    jsonl_file = results_dir / "sweep_results.jsonl"
    
    if npz_file.exists():
        logging.info(f"Loading binary results from {npz_file}")
        return load_sweep_binary(npz_file)
    elif json_file.exists():
        logging.info(f"Loading JSON results from {json_file}")
        with open(json_file, 'r') as f:
            return json.load(f)
    elif jsonl_file.exists():
        logging.info(f"Loading JSONL results from {jsonl_file}")
        results = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        return results
    else:
        raise FileNotFoundError(f"No sweep results found in {results_dir}")


def load_sweep_binary(input_path: Path) -> List[Dict]:
    """Load sweep results from .npz format."""
    data = np.load(input_path, allow_pickle=True)
    
    results = {}
    for key in data.keys():
        parts = key.split("_", 2)
        run_idx = int(parts[1])
        field = parts[2]
        
        if run_idx not in results:
            results[run_idx] = {}
        
        val = data[key]
        if val.ndim == 0:
            val = val.item()
        else:
            val = val.tolist()
        results[run_idx][field] = val
    
    return [results[i] for i in sorted(results.keys())]


def load_config(results_dir: Path) -> Dict:
    """Load configuration from saved config.json."""
    config_file = results_dir / "config.json"
    if not config_file.exists():
        logging.warning(f"Config file not found: {config_file}")
        return {}
    
    with open(config_file, 'r') as f:
        return json.load(f)


def load_fss_results(results_dir: Path) -> List[Dict]:
    """Load finite-size scaling results."""
    fss_file = results_dir / "fss_results.json"
    if not fss_file.exists():
        raise FileNotFoundError(f"FSS results not found: {fss_file}")
    
    with open(fss_file, 'r') as f:
        return json.load(f)


def load_sensitivity_results(results_dir: Path) -> List[Dict]:
    """Load evolution sensitivity results."""
    sens_file = results_dir / "sensitivity_results.json"
    if not sens_file.exists():
        raise FileNotFoundError(f"Sensitivity results not found: {sens_file}")
    
    with open(sens_file, 'r') as f:
        return json.load(f)


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


def load_phase2_results(results_dir: Path) -> List[Dict]:
    """
    Load Phase 2 (SOC test) results.
    
    Phase 2 tests self-organized criticality by running simulations with evolution
    from different initial prey_death values and checking if they converge.
    
    Returns
    -------
    results : List[Dict]
        List of simulation results, each containing:
        - prey_death: initial prey death rate
        - evolved_prey_death_final: final evolved prey death rate
        - evolved_prey_death_mean: mean evolved prey death during measurement
        - prey_mean, pred_mean: equilibrium populations
    """
    jsonl_file = results_dir / "phase2_results.jsonl"
    json_file = results_dir / "phase2_results.json"
    
    if jsonl_file.exists():
        logging.info(f"Loading Phase 2 results from {jsonl_file}")
        results = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results
    elif json_file.exists():
        logging.info(f"Loading Phase 2 results from {json_file}")
        with open(json_file, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Phase 2 results not found in {results_dir}")


def load_phase3_results(results_dir: Path) -> List[Dict]:
    """
    Load Phase 3 (Finite-Size Scaling) results.
    
    Phase 3 runs simulations at the critical point across multiple grid sizes
    to analyze how cluster size distributions scale with system size L.
    
    Returns
    -------
    results : List[Dict]
        List of simulation results, each containing:
        - grid_size: system size L
        - prey_cluster_sizes, pred_cluster_sizes: cluster size lists
        - prey_largest_fraction: largest cluster / total population
        - prey_mean, pred_mean: equilibrium populations
    """
    jsonl_file = results_dir / "phase3_results.jsonl"
    json_file = results_dir / "phase3_results.json"
    
    if jsonl_file.exists():
        logging.info(f"Loading Phase 3 results from {jsonl_file}")
        results = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results
    elif json_file.exists():
        logging.info(f"Loading Phase 3 results from {json_file}")
        with open(json_file, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Phase 3 results not found in {results_dir}")


def load_phase4_results(results_dir: Path, filename: str = "phase4_results.jsonl") -> List[Dict]:
    """
    Load Phase 4 (Global Sensitivity Analysis) results.
    
    Phase 4 runs a full 4D parameter sweep varying:
    - prey_birth, prey_death, predator_birth, predator_death
    
    This tests the sensitivity of the hydra effect and critical point
    across different parameter regimes.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing the results file
    filename : str, optional
        Name of the JSONL file to load. Default is "phase4_results.jsonl"
    
    Returns
    -------
    results : List[Dict]
        List of simulation results, each containing:
        - prey_birth, prey_death, predator_birth, predator_death: parameters
        - prey_mean, pred_mean: equilibrium populations
        - prey_survived, pred_survived: survival indicators
        - evolved_prey_death_final (if evolution enabled): final evolved trait
    """
    jsonl_file = results_dir / filename
    json_file = results_dir / filename.replace(".jsonl", ".json")
    
    if jsonl_file.exists():
        logging.info(f"Loading Phase 4 results from {jsonl_file}")
        results = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results
    elif json_file.exists():
        logging.info(f"Loading Phase 4 results from {json_file}")
        with open(json_file, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Phase 4 results not found: {jsonl_file} or {json_file}")


# =============================================================================
# DATA PROCESSING
# =============================================================================

def extract_parameter_grid(results: List[Dict], config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract prey_birth and prey_death ranges from results or config."""
    if config and 'prey_birth_min' in config:
        prey_births = np.linspace(
            config['prey_birth_min'],
            config['prey_birth_max'],
            config['n_prey_birth']
        )
        prey_deaths = np.linspace(
            config['prey_death_min'],
            config['prey_death_max'],
            config['n_prey_death']
        )
    else:
        # Infer from data
        prey_births = sorted(set(r['prey_birth'] for r in results))
        prey_deaths = sorted(set(r['prey_death'] for r in results))
        prey_births = np.array(prey_births)
        prey_deaths = np.array(prey_deaths)
    
    return prey_births, prey_deaths


def aggregate_to_grids(results: List[Dict], prey_births: np.ndarray, 
                       prey_deaths: np.ndarray) -> Dict[str, np.ndarray]:
    """Aggregate simulation results into 2D grids for plotting."""
    n_pb, n_pd = len(prey_births), len(prey_deaths)
    
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
        "prey_largest_fraction": np.full((n_pd, n_pb), np.nan),
        "prey_percolation_prob": np.full((n_pd, n_pb), np.nan),
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
                
                # Order parameter
                lf = [r.get("prey_largest_fraction_mean", np.nan) for r in no_evo]
                lf = [x for x in lf if not np.isnan(x)]
                if lf:
                    grids["prey_largest_fraction"][i, j] = np.mean(lf)
                
                # Percolation
                pp = [r.get("prey_percolation_prob", np.nan) for r in no_evo]
                pp = [x for x in pp if not np.isnan(x)]
                if pp:
                    grids["prey_percolation_prob"][i, j] = np.mean(pp)
            
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
    
    return grids


def compute_hydra_derivative(grids: Dict[str, np.ndarray], 
                             prey_deaths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ∂N/∂d (Hydra indicator) for both conditions."""
    dd = prey_deaths[1] - prey_deaths[0]
    n_pb = grids["prey_pop_no_evo"].shape[1]
    
    dN_dd_no_evo = np.zeros_like(grids["prey_pop_no_evo"])
    dN_dd_evo = np.zeros_like(grids["prey_pop_evo"])
    
    for j in range(n_pb):
        pop_smooth = gaussian_filter1d(grids["prey_pop_no_evo"][:, j], sigma=0.8)
        dN_dd_no_evo[:, j] = np.gradient(pop_smooth, dd)
        
        pop_smooth = gaussian_filter1d(grids["prey_pop_evo"][:, j], sigma=0.8)
        dN_dd_evo[:, j] = np.gradient(pop_smooth, dd)
    
    return dN_dd_no_evo, dN_dd_evo


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_phase_diagrams(grids: Dict, prey_births: np.ndarray, prey_deaths: np.ndarray,
                        dN_dd_no_evo: np.ndarray, output_dir: Path, dpi: int = 150):
    """Generate 6-panel phase diagram figure."""
    extent = [prey_births[0], prey_births[-1], prey_deaths[0], prey_deaths[-1]]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Panel 1: Prey population (no evolution)
    ax = axes[0, 0]
    im = ax.imshow(grids["prey_pop_no_evo"], origin="lower", aspect="auto",
                   extent=extent, cmap="YlGn")
    ax.contour(prey_births, prey_deaths, grids["survival_prey_no_evo"],
               levels=[50], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="Population")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Prey Population (No Evolution)")
    
    # Panel 2: Prey population (with evolution)
    ax = axes[0, 1]
    im = ax.imshow(grids["prey_pop_evo"], origin="lower", aspect="auto",
                   extent=extent, cmap="YlGn")
    ax.contour(prey_births, prey_deaths, grids["survival_prey_evo"],
               levels=[50], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="Population")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Prey Population (With Evolution)")
    
    # Panel 3: Evolution advantage
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
    ax.set_title("Evolutionary Advantage")
    
    # Panel 4: Critical exponent τ
    ax = axes[1, 0]
    im = ax.imshow(grids["tau_prey"], origin="lower", aspect="auto",
                   extent=extent, cmap="coolwarm", vmin=1.5, vmax=2.5)
    ax.contour(prey_births, prey_deaths, grids["tau_prey"],
               levels=[2.05], colors="green", linewidths=2)
    plt.colorbar(im, ax=ax, label="τ")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Cluster Exponent τ (Green: Critical Point)")
    
    # Panel 5: Evolved mortality rate
    ax = axes[1, 1]
    im = ax.imshow(grids["evolved_prey_death"], origin="lower", aspect="auto",
                   extent=extent, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Evolved d")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Initial Prey Death Rate")
    ax.set_title("Evolved Prey Death Rate")
    
    # Panel 6: Hydra effect
    ax = axes[1, 2]
    im = ax.imshow(dN_dd_no_evo, origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu_r", vmin=-5000, vmax=5000)
    ax.contour(prey_births, prey_deaths, dN_dd_no_evo,
               levels=[0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="∂N/∂d")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("HYDRA EFFECT (Red: N↑ with d↑)")
    
    plt.tight_layout()
    output_file = output_dir / "phase_diagrams.png"
    plt.savefig(output_file, dpi=dpi)
    plt.close()
    logging.info(f"Saved {output_file}")


def plot_hydra_analysis(grids: Dict, prey_births: np.ndarray, prey_deaths: np.ndarray,
                        dN_dd_no_evo: np.ndarray, dN_dd_evo: np.ndarray,
                        output_dir: Path, dpi: int = 150):
    """Generate 3-panel Hydra analysis figure."""
    extent = [prey_births[0], prey_births[-1], prey_deaths[0], prey_deaths[-1]]
    n_pb = len(prey_births)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Hydra (no evolution)
    ax = axes[0]
    im = ax.imshow(dN_dd_no_evo, origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu_r", vmin=-5000, vmax=5000)
    ax.contour(prey_births, prey_deaths, dN_dd_no_evo,
               levels=[0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="∂N/∂d")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Hydra Effect (No Evolution)")
    
    # Panel 2: Hydra (with evolution)
    ax = axes[1]
    im = ax.imshow(dN_dd_evo, origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu_r", vmin=-5000, vmax=5000)
    ax.contour(prey_births, prey_deaths, dN_dd_evo,
               levels=[0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="∂N/∂d")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Hydra Effect (With Evolution)")
    
    # Panel 3: 1D slice showing Hydra
    ax = axes[2]
    mid_pb_idx = n_pb // 2
    target_pb = prey_births[mid_pb_idx]
    no_evo_slice = grids["prey_pop_no_evo"][:, mid_pb_idx]
    evo_slice = grids["prey_pop_evo"][:, mid_pb_idx]
    
    ax.plot(prey_deaths, no_evo_slice, 'b-o',
            label=f'No Evolution', markersize=4, linewidth=2)
    ax.plot(prey_deaths, evo_slice, 'g-s',
            label=f'With Evolution', markersize=4, linewidth=2)
    
    ax.set_xlabel("Prey Death Rate")
    ax.set_ylabel("Prey Population")
    ax.set_title(f"Prey Density vs. Mortality (b={target_pb:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "hydra_analysis.png"
    plt.savefig(output_file, dpi=dpi)
    plt.close()
    logging.info(f"Saved {output_file}")


def plot_pcf_analysis(grids: Dict, prey_births: np.ndarray, prey_deaths: np.ndarray,
                      dN_dd_no_evo: np.ndarray, output_dir: Path, dpi: int = 150):
    """Generate 3-panel PCF spatial correlation figure."""
    extent = [prey_births[0], prey_births[-1], prey_deaths[0], prey_deaths[-1]]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Segregation index
    ax = axes[0]
    im = ax.imshow(grids["segregation_index"], origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu", vmin=0.5, vmax=1.5)
    ax.contour(prey_births, prey_deaths, grids["segregation_index"],
               levels=[1.0], colors="black", linewidths=2)
    plt.colorbar(im, ax=ax, label="C_cr(r<3)")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Segregation Index (C_cr)")
    
    # Panel 2: Prey clustering
    ax = axes[1]
    im = ax.imshow(grids["prey_clustering_index"], origin="lower", aspect="auto",
                   extent=extent, cmap="Greens", vmin=1.0, vmax=3.0)
    plt.colorbar(im, ax=ax, label="C_rr(r<3)")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Prey Clustering Index (C_rr)")
    
    # Panel 3: Overlay with boundaries
    ax = axes[2]
    im = ax.imshow(grids["segregation_index"], origin="lower", aspect="auto",
                   extent=extent, cmap="RdBu", vmin=0.5, vmax=1.5, alpha=0.8)
    ax.contour(prey_births, prey_deaths, dN_dd_no_evo,
               levels=[0], colors="lime", linewidths=3, label="Hydra Boundary")
    ax.contour(prey_births, prey_deaths, grids["survival_prey_no_evo"],
               levels=[50], colors="black", linewidths=2, linestyles='--',
               label="Coexistence Boundary")
    plt.colorbar(im, ax=ax, label="C_cr(r<3)")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Prey Death Rate")
    ax.set_title("Segregation + Phase Boundaries")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_file = output_dir / "pcf_analysis.png"
    plt.savefig(output_file, dpi=dpi)
    plt.close()
    logging.info(f"Saved {output_file}")


def plot_fss_analysis(fss_results: List[Dict], output_dir: Path, dpi: int = 150):
    """Generate finite-size scaling analysis plots."""
    # Group by grid size
    by_size = defaultdict(list)
    for r in fss_results:
        by_size[r['grid_size']].append(r)
    
    sizes = sorted(by_size.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: τ vs. L
    ax = axes[0, 0]
    taus, tau_ses = [], []
    for L in sizes:
        tau_vals = [r['prey_tau'] for r in by_size[L] if not np.isnan(r.get('prey_tau', np.nan))]
        if tau_vals:
            taus.append(np.mean(tau_vals))
            tau_ses.append(np.std(tau_vals) / np.sqrt(len(tau_vals)))
        else:
            taus.append(np.nan)
            tau_ses.append(np.nan)
    
    ax.errorbar(sizes, taus, yerr=tau_ses, fmt='o-', capsize=5, linewidth=2)
    ax.axhline(2.05, color='red', linestyle='--', label='Critical τ = 2.05')
    ax.set_xlabel("System Size L")
    ax.set_ylabel("Cluster Exponent τ")
    ax.set_title("Critical Exponent vs. System Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: s_c vs. L (log-log)
    ax = axes[0, 1]
    s_cs = []
    for L in sizes:
        sc_vals = [r['prey_s_c'] for r in by_size[L] if not np.isnan(r.get('prey_s_c', np.nan))]
        if sc_vals:
            s_cs.append(np.mean(sc_vals))
        else:
            s_cs.append(np.nan)
    
    valid = ~np.isnan(s_cs)
    if np.sum(valid) >= 2:
        ax.plot(np.array(sizes)[valid], np.array(s_cs)[valid], 'o-', linewidth=2)
        
        # Fit power law
        log_L = np.log(np.array(sizes)[valid])
        log_sc = np.log(np.array(s_cs)[valid])
        slope, intercept, r_val, _, _ = linregress(log_L, log_sc)
        
        ax.plot(sizes, np.exp(intercept) * np.array(sizes)**slope, 'r--',
                label=f'Fit: s_c ∼ L^{slope:.2f} (R²={r_val**2:.3f})')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("System Size L")
    ax.set_ylabel("Cutoff Scale s_c")
    ax.set_title("Correlation Length Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Order parameter (largest cluster fraction)
    ax = axes[1, 0]
    lf_means, lf_ses = [], []
    for L in sizes:
        lf_vals = [r['prey_largest_fraction'] for r in by_size[L] 
                  if not np.isnan(r.get('prey_largest_fraction', np.nan))]
        if lf_vals:
            lf_means.append(np.mean(lf_vals))
            lf_ses.append(np.std(lf_vals) / np.sqrt(len(lf_vals)))
        else:
            lf_means.append(np.nan)
            lf_ses.append(np.nan)
    
    ax.errorbar(sizes, lf_means, yerr=lf_ses, fmt='o-', capsize=5, linewidth=2)
    ax.set_xlabel("System Size L")
    ax.set_ylabel("Largest Cluster Fraction")
    ax.set_title("Order Parameter Φ(L)")
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Percolation probability
    ax = axes[1, 1]
    perc_probs = []
    for L in sizes:
        pp_vals = [r['prey_percolation_prob'] for r in by_size[L] 
                  if not np.isnan(r.get('prey_percolation_prob', np.nan))]
        if pp_vals:
            perc_probs.append(np.mean(pp_vals))
        else:
            perc_probs.append(np.nan)
    
    ax.plot(sizes, perc_probs, 'o-', linewidth=2)
    ax.axhline(0.5, color='red', linestyle='--', label='Critical P = 0.5')
    ax.set_xlabel("System Size L")
    ax.set_ylabel("Percolation Probability")
    ax.set_title("Phase Transition Indicator")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "fss_analysis.png"
    plt.savefig(output_file, dpi=dpi)
    plt.close()
    logging.info(f"Saved {output_file}")


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


def plot_sensitivity_analysis(sens_results: List[Dict], output_dir: Path, dpi: int = 150):
    """Generate evolution sensitivity analysis plots."""
    # Group by evolve_sd
    by_sd = defaultdict(list)
    for r in sens_results:
        sd = r.get('evolve_sd', np.nan)
        if not np.isnan(sd):
            by_sd[sd].append(r)
    
    sd_values = sorted(by_sd.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Prey population vs. σ
    ax = axes[0, 0]
    prey_means, prey_ses = [], []
    for sd in sd_values:
        pops = [r['prey_mean'] for r in by_sd[sd]]
        prey_means.append(np.mean(pops))
        prey_ses.append(np.std(pops) / np.sqrt(len(pops)))
    
    ax.errorbar(sd_values, prey_means, yerr=prey_ses, fmt='o-', capsize=5, linewidth=2)
    ax.set_xlabel("Mutation Strength σ")
    ax.set_ylabel("Prey Population")
    ax.set_title("Population Response to Evolution Strength")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Evolved trait vs. σ
    ax = axes[0, 1]
    trait_means, trait_ses = [], []
    for sd in sd_values:
        traits = [r['evolved_prey_death_mean'] for r in by_sd[sd] 
                 if not np.isnan(r.get('evolved_prey_death_mean', np.nan))]
        if traits:
            trait_means.append(np.mean(traits))
            trait_ses.append(np.std(traits) / np.sqrt(len(traits)))
        else:
            trait_means.append(np.nan)
            trait_ses.append(np.nan)
    
    ax.errorbar(sd_values, trait_means, yerr=trait_ses, fmt='o-', capsize=5, linewidth=2)
    ax.set_xlabel("Mutation Strength σ")
    ax.set_ylabel("Evolved Prey Death Rate")
    ax.set_title("Selection Response")
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Segregation vs. σ
    ax = axes[1, 0]
    seg_means = []
    for sd in sd_values:
        seg_vals = [r['segregation_index'] for r in by_sd[sd] 
                   if not np.isnan(r.get('segregation_index', np.nan))]
        if seg_vals:
            seg_means.append(np.mean(seg_vals))
        else:
            seg_means.append(np.nan)
    
    ax.plot(sd_values, seg_means, 'o-', linewidth=2)
    ax.axhline(1.0, color='red', linestyle='--', label='Random Mixing')
    ax.set_xlabel("Mutation Strength σ")
    ax.set_ylabel("Segregation Index C_cr")
    ax.set_title("Spatial Structure vs. Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Critical exponent vs. σ
    ax = axes[1, 1]
    tau_means = []
    for sd in sd_values:
        tau_vals = [r['prey_tau'] for r in by_sd[sd] 
                   if not np.isnan(r.get('prey_tau', np.nan))]
        if tau_vals:
            tau_means.append(np.mean(tau_vals))
        else:
            tau_means.append(np.nan)
    
    ax.plot(sd_values, tau_means, 'o-', linewidth=2)
    ax.axhline(2.05, color='red', linestyle='--', label='Critical Point')
    ax.set_xlabel("Mutation Strength σ")
    ax.set_ylabel("Cluster Exponent τ")
    ax.set_title("Evolution Effect on Critical Point")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "sensitivity_analysis.png"
    plt.savefig(output_file, dpi=dpi)
    plt.close()
    logging.info(f"Saved {output_file}")


def plot_phase2_soc_analysis(results: List[Dict], output_dir: Path, dpi: int = 150,
                             critical_prey_death: float = 0.0963):
    """
    Generate Phase 2 self-organized criticality (SOC) analysis plots.
    
    Tests whether prey populations evolve toward a critical point regardless
    of their starting prey_death value.
    
    Parameters
    ----------
    results : List[Dict]
        Phase 2 simulation results with evolved prey_death values.
    output_dir : Path
        Directory to save output figures.
    dpi : int
        Output resolution.
    critical_prey_death : float
        Expected critical prey death rate for reference line.
    """
    # Extract data
    initial_pd = []
    final_pd = []
    mean_pd = []
    prey_pops = []
    pred_pops = []
    
    for r in results:
        if r.get('evolved_prey_death_final') is not None:
            initial_pd.append(r.get('prey_death', np.nan))
            final_pd.append(r.get('evolved_prey_death_final', np.nan))
            mean_pd.append(r.get('evolved_prey_death_mean', np.nan))
            prey_pops.append(r.get('prey_mean', np.nan))
            pred_pops.append(r.get('pred_mean', np.nan))
    
    initial_pd = np.array(initial_pd)
    final_pd = np.array(final_pd)
    mean_pd = np.array(mean_pd)
    prey_pops = np.array(prey_pops)
    pred_pops = np.array(pred_pops)
    
    # Remove NaN values
    valid = ~(np.isnan(initial_pd) | np.isnan(final_pd))
    initial_pd = initial_pd[valid]
    final_pd = final_pd[valid]
    mean_pd = mean_pd[valid]
    prey_pops = prey_pops[valid]
    pred_pops = pred_pops[valid]
    
    if len(initial_pd) == 0:
        logging.warning("No valid Phase 2 results to plot")
        return None
    
    # Get unique initial values for grouping
    unique_initial = np.unique(initial_pd)
    
    # Create figure with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Phase 2: Self-Organized Criticality Test", fontsize=14, fontweight='bold')
    
    # ==========================================================================
    # Panel 1: Convergence Plot (Main SOC Test)
    # ==========================================================================
    ax = axes[0, 0]
    
    # Plot individual replicates as scatter
    ax.scatter(initial_pd, final_pd, alpha=0.4, s=30, c='steelblue', label='Individual runs')
    
    # Plot mean ± std for each initial value
    means = []
    stds = []
    for init_val in unique_initial:
        mask = initial_pd == init_val
        means.append(np.mean(final_pd[mask]))
        stds.append(np.std(final_pd[mask]))
    
    ax.errorbar(unique_initial, means, yerr=stds, fmt='o-', color='darkred', 
                linewidth=2, markersize=8, capsize=5, label='Mean ± SD')
    
    # Reference lines
    ax.axhline(critical_prey_death, color='green', linestyle='--', linewidth=2,
               label=f'Critical point ({critical_prey_death})')
    ax.plot([0, 0.2], [0, 0.2], 'k:', alpha=0.5, label='No evolution (y=x)')
    
    ax.set_xlabel("Initial Prey Death Rate", fontsize=12)
    ax.set_ylabel("Final Evolved Prey Death Rate", fontsize=12)
    ax.set_title("SOC Test: Do All Initial Conditions Converge?", fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 0.21)
    ax.set_ylim(-0.01, 0.21)
    
    # ==========================================================================
    # Panel 2: Distribution of Final Evolved Values
    # ==========================================================================
    ax = axes[0, 1]
    
    # Histogram of final evolved prey_death
    ax.hist(final_pd, bins=30, density=True, alpha=0.7, color='steelblue', 
            edgecolor='black', label='Final evolved values')
    
    # Add vertical line for critical point
    ax.axvline(critical_prey_death, color='green', linestyle='--', linewidth=2,
               label=f'Critical point ({critical_prey_death})')
    
    # Add vertical line for mean of final values
    final_mean = np.mean(final_pd)
    ax.axvline(final_mean, color='darkred', linestyle='-', linewidth=2,
               label=f'Mean ({final_mean:.4f})')
    
    ax.set_xlabel("Final Evolved Prey Death Rate", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Final Evolved Values", fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Panel 3: Final Prey Death vs Population
    # ==========================================================================
    ax = axes[1, 0]
    
    # Color by initial prey_death
    scatter = ax.scatter(final_pd, prey_pops, c=initial_pd, cmap='viridis', 
                         alpha=0.6, s=40)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Initial Prey Death Rate', fontsize=10)
    
    ax.axvline(critical_prey_death, color='green', linestyle='--', linewidth=2,
               label=f'Critical point')
    
    ax.set_xlabel("Final Evolved Prey Death Rate", fontsize=12)
    ax.set_ylabel("Equilibrium Prey Population", fontsize=12)
    ax.set_title("Population vs Evolved Trait", fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Panel 4: Boxplot of Final Values by Initial Condition
    # ==========================================================================
    ax = axes[1, 1]
    
    # Create boxplot data
    boxplot_data = []
    boxplot_labels = []
    for init_val in unique_initial:
        mask = initial_pd == init_val
        boxplot_data.append(final_pd[mask])
        boxplot_labels.append(f'{init_val:.2f}')
    
    bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_initial)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(critical_prey_death, color='green', linestyle='--', linewidth=2,
               label=f'Critical point ({critical_prey_death})')
    
    ax.set_xlabel("Initial Prey Death Rate", fontsize=12)
    ax.set_ylabel("Final Evolved Prey Death Rate", fontsize=12)
    ax.set_title("Convergence by Initial Condition", fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "phase2_soc_analysis.png"
    plt.savefig(output_file, dpi=dpi)
    plt.close()
    logging.info(f"Saved {output_file}")
    
    # ==========================================================================
    # Generate Summary Statistics
    # ==========================================================================
    summary = {
        'n_simulations': len(final_pd),
        'initial_prey_death_values': unique_initial.tolist(),
        'n_replicates_per_condition': len(final_pd) // len(unique_initial) if len(unique_initial) > 0 else 0,
        'final_prey_death_mean': float(np.mean(final_pd)),
        'final_prey_death_std': float(np.std(final_pd)),
        'final_prey_death_median': float(np.median(final_pd)),
        'critical_prey_death_reference': critical_prey_death,
        'distance_from_critical': float(np.abs(np.mean(final_pd) - critical_prey_death)),
        'convergence_achieved': float(np.std(final_pd)) < 0.02,  # Low variance = convergence
    }
    
    # Per-condition statistics
    summary['per_condition'] = {}
    for init_val in unique_initial:
        mask = initial_pd == init_val
        summary['per_condition'][f'init_{init_val:.3f}'] = {
            'mean': float(np.mean(final_pd[mask])),
            'std': float(np.std(final_pd[mask])),
            'n': int(np.sum(mask)),
        }
    
    summary_file = output_dir / "phase2_soc_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved {summary_file}")
    
    # Log key findings
    logging.info("Phase 2 SOC Analysis Summary:")
    logging.info(f"  Final prey_death mean: {summary['final_prey_death_mean']:.4f} ± {summary['final_prey_death_std']:.4f}")
    logging.info(f"  Critical reference: {critical_prey_death}")
    logging.info(f"  Distance from critical: {summary['distance_from_critical']:.4f}")
    logging.info(f"  Convergence achieved: {summary['convergence_achieved']}")
    
    return output_file


def plot_phase3_fss_analysis(results: List[Dict], output_dir: Path, dpi: int = 150):
    """
    Generate Phase 3 finite-size scaling (FSS) analysis plots.
    
    Analyzes how cluster size distributions scale with system size L at the
    critical point. Key predictions for critical systems:
    - Cluster size distribution: P(s) ~ s^(-tau) with cutoff at s_max ~ L^D
    - Largest cluster fraction scales with L
    
    Parameters
    ----------
    results : List[Dict]
        Phase 3 simulation results with cluster sizes at different grid sizes.
    output_dir : Path
        Directory to save output figures.
    dpi : int
        Output resolution.
    """
    import ast
    
    # Parse cluster sizes helper
    def parse_clusters(x):
        if isinstance(x, list):
            return x
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return []
        return []
    
    # Extract data by grid size
    grid_sizes = sorted(set(r['grid_size'] for r in results))
    
    # Data structures for analysis
    data_by_L = {L: {
        'prey_clusters': [],
        'pred_clusters': [],
        'prey_largest_frac': [],
        'pred_largest_frac': [],
        'prey_max_cluster': [],
        'prey_n_clusters': [],
        'prey_mean_cluster': [],
    } for L in grid_sizes}
    
    # Collect data
    for r in results:
        L = r['grid_size']
        prey_clusters = parse_clusters(r.get('prey_cluster_sizes', []))
        pred_clusters = parse_clusters(r.get('pred_cluster_sizes', []))
        
        data_by_L[L]['prey_clusters'].extend(prey_clusters)
        data_by_L[L]['pred_clusters'].extend(pred_clusters)
        
        if r.get('prey_largest_fraction') is not None and not np.isnan(r.get('prey_largest_fraction', np.nan)):
            data_by_L[L]['prey_largest_frac'].append(r['prey_largest_fraction'])
        
        if prey_clusters:
            data_by_L[L]['prey_max_cluster'].append(max(prey_clusters))
            data_by_L[L]['prey_n_clusters'].append(len(prey_clusters))
            data_by_L[L]['prey_mean_cluster'].append(np.mean(prey_clusters))
    
    if len(grid_sizes) == 0:
        logging.warning("No valid Phase 3 results to plot")
        return None
    
    # Create figure with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Phase 3: Finite-Size Scaling at Critical Point", fontsize=14, fontweight='bold')
    
    # Color map for different grid sizes
    colors = plt.cm.viridis(np.linspace(0, 1, len(grid_sizes)))
    
    # ==========================================================================
    # Panel 1: Cluster Size Distributions by Grid Size
    # ==========================================================================
    ax = axes[0, 0]
    
    for L, color in zip(grid_sizes, colors):
        clusters = np.array(data_by_L[L]['prey_clusters'])
        if len(clusters) == 0:
            continue
        clusters = clusters[clusters > 0]
        
        # Compute histogram
        sizes, counts = np.unique(clusters, return_counts=True)
        # Normalize by number of replicates for fair comparison
        n_replicates = len([r for r in results if r['grid_size'] == L])
        counts = counts / n_replicates
        
        ax.scatter(sizes, counts, alpha=0.5, s=15, color=color, label=f'L={L}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Cluster Size s', fontsize=12)
    ax.set_ylabel('P(s) (normalized)', fontsize=12)
    ax.set_title('Cluster Size Distribution vs System Size', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Panel 2: Maximum Cluster Size vs L (Power-law scaling)
    # ==========================================================================
    ax = axes[0, 1]
    
    L_vals = []
    s_max_mean = []
    s_max_std = []
    
    for L in grid_sizes:
        max_clusters = data_by_L[L]['prey_max_cluster']
        if len(max_clusters) > 0:
            L_vals.append(L)
            s_max_mean.append(np.mean(max_clusters))
            s_max_std.append(np.std(max_clusters))
    
    L_vals = np.array(L_vals)
    s_max_mean = np.array(s_max_mean)
    s_max_std = np.array(s_max_std)
    
    ax.errorbar(L_vals, s_max_mean, yerr=s_max_std, fmt='o-', color='steelblue',
                markersize=8, capsize=5, linewidth=2, label='Data')
    
    # Fit power law: s_max ~ L^D
    if len(L_vals) >= 3:
        log_L = np.log10(L_vals)
        log_s = np.log10(s_max_mean)
        slope, intercept, r_value, _, _ = linregress(log_L, log_s)
        
        fit_L = np.logspace(np.log10(L_vals.min()), np.log10(L_vals.max()), 50)
        fit_s = 10**intercept * fit_L**slope
        ax.plot(fit_L, fit_s, 'r--', linewidth=2, 
                label=f'Fit: $s_{{max}} \\sim L^{{{slope:.2f}}}$ (R²={r_value**2:.3f})')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('System Size L', fontsize=12)
    ax.set_ylabel('Maximum Cluster Size $s_{max}$', fontsize=12)
    ax.set_title('Finite-Size Scaling of Maximum Cluster', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Panel 3: Largest Cluster Fraction vs L
    # ==========================================================================
    ax = axes[1, 0]
    
    L_vals_frac = []
    frac_mean = []
    frac_std = []
    
    for L in grid_sizes:
        fracs = data_by_L[L]['prey_largest_frac']
        if len(fracs) > 0:
            L_vals_frac.append(L)
            frac_mean.append(np.mean(fracs))
            frac_std.append(np.std(fracs))
    
    L_vals_frac = np.array(L_vals_frac)
    frac_mean = np.array(frac_mean)
    frac_std = np.array(frac_std)
    
    ax.errorbar(L_vals_frac, frac_mean, yerr=frac_std, fmt='s-', color='forestgreen',
                markersize=8, capsize=5, linewidth=2, label='Data')
    
    # Fit power law for largest fraction scaling
    if len(L_vals_frac) >= 3:
        log_L = np.log10(L_vals_frac)
        log_frac = np.log10(frac_mean)
        slope_frac, intercept_frac, r_value_frac, _, _ = linregress(log_L, log_frac)
        
        fit_L = np.logspace(np.log10(L_vals_frac.min()), np.log10(L_vals_frac.max()), 50)
        fit_frac = 10**intercept_frac * fit_L**slope_frac
        ax.plot(fit_L, fit_frac, 'r--', linewidth=2,
                label=f'Fit: $P_{{\\infty}} \\sim L^{{{slope_frac:.2f}}}$ (R²={r_value_frac**2:.3f})')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('System Size L', fontsize=12)
    ax.set_ylabel('Largest Cluster Fraction $P_\\infty$', fontsize=12)
    ax.set_title('Order Parameter Scaling', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Panel 4: Number of Clusters vs L
    # ==========================================================================
    ax = axes[1, 1]
    
    L_vals_n = []
    n_clusters_mean = []
    n_clusters_std = []
    
    for L in grid_sizes:
        n_clusters = data_by_L[L]['prey_n_clusters']
        if len(n_clusters) > 0:
            L_vals_n.append(L)
            n_clusters_mean.append(np.mean(n_clusters))
            n_clusters_std.append(np.std(n_clusters))
    
    L_vals_n = np.array(L_vals_n)
    n_clusters_mean = np.array(n_clusters_mean)
    n_clusters_std = np.array(n_clusters_std)
    
    ax.errorbar(L_vals_n, n_clusters_mean, yerr=n_clusters_std, fmt='d-', color='darkorange',
                markersize=8, capsize=5, linewidth=2, label='Data')
    
    # Fit power law
    if len(L_vals_n) >= 3:
        log_L = np.log10(L_vals_n)
        log_n = np.log10(n_clusters_mean)
        slope_n, intercept_n, r_value_n, _, _ = linregress(log_L, log_n)
        
        fit_L = np.logspace(np.log10(L_vals_n.min()), np.log10(L_vals_n.max()), 50)
        fit_n = 10**intercept_n * fit_L**slope_n
        ax.plot(fit_L, fit_n, 'r--', linewidth=2,
                label=f'Fit: $N_c \\sim L^{{{slope_n:.2f}}}$ (R²={r_value_n**2:.3f})')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('System Size L', fontsize=12)
    ax.set_ylabel('Number of Clusters $N_c$', fontsize=12)
    ax.set_title('Cluster Count Scaling', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "phase3_fss_analysis.png"
    plt.savefig(output_file, dpi=dpi)
    plt.close()
    logging.info(f"Saved {output_file}")
    
    # ==========================================================================
    # Generate Summary Statistics
    # ==========================================================================
    summary = {
        'n_simulations': len(results),
        'grid_sizes': grid_sizes,
        'n_replicates_per_size': len(results) // len(grid_sizes) if len(grid_sizes) > 0 else 0,
    }
    
    # Scaling exponents
    if len(L_vals) >= 3:
        summary['s_max_exponent'] = float(slope)
        summary['s_max_exponent_r2'] = float(r_value**2)
    
    if len(L_vals_frac) >= 3:
        summary['largest_frac_exponent'] = float(slope_frac)
        summary['largest_frac_exponent_r2'] = float(r_value_frac**2)
    
    if len(L_vals_n) >= 3:
        summary['n_clusters_exponent'] = float(slope_n)
        summary['n_clusters_exponent_r2'] = float(r_value_n**2)
    
    # Per-size statistics
    summary['per_size'] = {}
    for L in grid_sizes:
        summary['per_size'][f'L_{L}'] = {
            's_max_mean': float(np.mean(data_by_L[L]['prey_max_cluster'])) if data_by_L[L]['prey_max_cluster'] else None,
            's_max_std': float(np.std(data_by_L[L]['prey_max_cluster'])) if data_by_L[L]['prey_max_cluster'] else None,
            'largest_frac_mean': float(np.mean(data_by_L[L]['prey_largest_frac'])) if data_by_L[L]['prey_largest_frac'] else None,
            'n_clusters_mean': float(np.mean(data_by_L[L]['prey_n_clusters'])) if data_by_L[L]['prey_n_clusters'] else None,
        }
    
    summary_file = output_dir / "phase3_fss_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved {summary_file}")
    
    # Log key findings
    logging.info("Phase 3 FSS Analysis Summary:")
    logging.info(f"  Grid sizes tested: {grid_sizes}")
    if 's_max_exponent' in summary:
        logging.info(f"  s_max scaling exponent: {summary['s_max_exponent']:.3f} (R²={summary['s_max_exponent_r2']:.3f})")
    if 'largest_frac_exponent' in summary:
        logging.info(f"  Largest frac exponent: {summary['largest_frac_exponent']:.3f}")
    
    return output_file


def plot_phase4_sensitivity_analysis(results: List[Dict], output_dir: Path, dpi: int = 150,
                                      critical_prey_death: float = 0.0963):
    """
    Generate Phase 4 global sensitivity analysis plots.
    
    Phase 4 tests the sensitivity of the hydra effect and critical point
    across a full 4D parameter sweep (prey_birth, prey_death, pred_birth, pred_death).
    
    Key analyses:
    1. Where does the hydra effect occur across parameter space?
    2. How does the critical point vary with other parameters?
    3. Is there evidence for self-organized criticality across regimes?
    
    Parameters
    ----------
    results : List[Dict]
        Phase 4 simulation results with 4D parameter sweep
    output_dir : Path
        Directory to save plots
    dpi : int
        Plot resolution
    critical_prey_death : float
        Reference critical point for comparison
    """
    import pandas as pd
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Check if evolution data is present
    has_evolution = 'evolved_prey_death_final' in df.columns and df['evolved_prey_death_final'].notna().any()
    
    logging.info(f"Phase 4 Analysis: {len(df)} simulations")
    logging.info(f"  Evolution data present: {has_evolution}")
    
    # Get unique parameter values
    prey_births = sorted(df['prey_birth'].unique())
    prey_deaths = sorted(df['prey_death'].unique())
    pred_births = sorted(df['predator_birth'].unique())
    pred_deaths = sorted(df['predator_death'].unique())
    
    logging.info(f"  prey_birth values: {len(prey_births)} ({min(prey_births):.2f} to {max(prey_births):.2f})")
    logging.info(f"  prey_death values: {len(prey_deaths)} ({min(prey_deaths):.2f} to {max(prey_deaths):.2f})")
    logging.info(f"  pred_birth values: {len(pred_births)} ({min(pred_births):.2f} to {max(pred_births):.2f})")
    logging.info(f"  pred_death values: {len(pred_deaths)} ({min(pred_deaths):.2f} to {max(pred_deaths):.2f})")
    
    # ==========================================================================
    # Figure 1: Coexistence Regions (2D slices)
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Phase 4: Coexistence Regions Across Parameter Space", fontsize=14, fontweight='bold')
    
    # Panel 1: prey_birth vs prey_death (averaged over pred params)
    ax = axes[0, 0]
    coex_grid = np.zeros((len(prey_deaths), len(prey_births)))
    
    for i, pd_val in enumerate(prey_deaths):
        for j, pb_val in enumerate(prey_births):
            subset = df[(df['prey_death'] == pd_val) & (df['prey_birth'] == pb_val)]
            if len(subset) > 0:
                # Coexistence = both prey and predator survive
                coex_rate = ((subset['prey_survived'] == True) & (subset['pred_survived'] == True)).mean()
                coex_grid[i, j] = coex_rate * 100
    
    im = ax.imshow(coex_grid, origin='lower', aspect='auto', cmap='RdYlGn',
                   extent=[min(prey_births), max(prey_births), min(prey_deaths), max(prey_deaths)],
                   vmin=0, vmax=100)
    ax.set_xlabel('Prey Birth Rate', fontsize=12)
    ax.set_ylabel('Prey Death Rate', fontsize=12)
    ax.set_title('Coexistence Rate (%) - Prey vs Predator Params Averaged', fontsize=11)
    ax.axhline(critical_prey_death, color='white', linestyle='--', linewidth=2, label=f'd*={critical_prey_death}')
    ax.legend(loc='upper right', fontsize=9)
    plt.colorbar(im, ax=ax, label='Coexistence %')
    
    # Panel 2: pred_birth vs pred_death (averaged over prey params)
    ax = axes[0, 1]
    coex_grid2 = np.zeros((len(pred_deaths), len(pred_births)))
    
    for i, pred_d in enumerate(pred_deaths):
        for j, pred_b in enumerate(pred_births):
            subset = df[(df['predator_death'] == pred_d) & (df['predator_birth'] == pred_b)]
            if len(subset) > 0:
                coex_rate = ((subset['prey_survived'] == True) & (subset['pred_survived'] == True)).mean()
                coex_grid2[i, j] = coex_rate * 100
    
    im2 = ax.imshow(coex_grid2, origin='lower', aspect='auto', cmap='RdYlGn',
                    extent=[min(pred_births), max(pred_births), min(pred_deaths), max(pred_deaths)],
                    vmin=0, vmax=100)
    ax.set_xlabel('Predator Birth Rate', fontsize=12)
    ax.set_ylabel('Predator Death Rate', fontsize=12)
    ax.set_title('Coexistence Rate (%) - Pred Params (Prey Averaged)', fontsize=11)
    plt.colorbar(im2, ax=ax, label='Coexistence %')
    
    # Panel 3: Mean prey population across prey_death (sensitivity)
    ax = axes[1, 0]
    for pb in prey_births[::2]:  # Every other prey_birth for clarity
        means = []
        stds = []
        for pd_val in prey_deaths:
            subset = df[(df['prey_birth'] == pb) & (df['prey_death'] == pd_val)]
            if len(subset) > 0:
                means.append(subset['prey_mean'].mean())
                stds.append(subset['prey_mean'].std())
            else:
                means.append(np.nan)
                stds.append(np.nan)
        ax.plot(prey_deaths, means, 'o-', label=f'pb={pb:.1f}', alpha=0.7)
    
    ax.axvline(critical_prey_death, color='red', linestyle='--', linewidth=2, label=f'd*={critical_prey_death}')
    ax.set_xlabel('Prey Death Rate', fontsize=12)
    ax.set_ylabel('Mean Prey Population', fontsize=12)
    ax.set_title('Prey Population Sensitivity to prey_death', fontsize=11)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Mean predator population across prey_death
    ax = axes[1, 1]
    for pb in prey_births[::2]:
        means = []
        for pd_val in prey_deaths:
            subset = df[(df['prey_birth'] == pb) & (df['prey_death'] == pd_val)]
            if len(subset) > 0:
                means.append(subset['pred_mean'].mean())
            else:
                means.append(np.nan)
        ax.plot(prey_deaths, means, 'o-', label=f'pb={pb:.1f}', alpha=0.7)
    
    ax.axvline(critical_prey_death, color='red', linestyle='--', linewidth=2, label=f'd*={critical_prey_death}')
    ax.set_xlabel('Prey Death Rate', fontsize=12)
    ax.set_ylabel('Mean Predator Population', fontsize=12)
    ax.set_title('Predator Population Sensitivity to prey_death', fontsize=11)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file1 = output_dir / "phase4_coexistence.png"
    plt.savefig(output_file1, dpi=dpi)
    plt.close()
    logging.info(f"Saved {output_file1}")
    
    # ==========================================================================
    # Figure 2: Hydra Effect Detection
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Phase 4: Hydra Effect Detection", fontsize=14, fontweight='bold')
    
    # Calculate dN/d(prey_death) for hydra effect detection
    # Hydra effect: prey population INCREASES when prey mortality increases
    
    # Panel 1: Hydra effect heatmap
    ax = axes[0, 0]
    hydra_grid = np.zeros((len(prey_deaths)-1, len(prey_births)))
    
    for j, pb in enumerate(prey_births):
        for i in range(len(prey_deaths)-1):
            pd1, pd2 = prey_deaths[i], prey_deaths[i+1]
            subset1 = df[(df['prey_birth'] == pb) & (df['prey_death'] == pd1)]
            subset2 = df[(df['prey_birth'] == pb) & (df['prey_death'] == pd2)]
            
            if len(subset1) > 0 and len(subset2) > 0:
                N1 = subset1['prey_mean'].mean()
                N2 = subset2['prey_mean'].mean()
                dN_dd = (N2 - N1) / (pd2 - pd1)
                hydra_grid[i, j] = dN_dd
    
    # Positive values = hydra effect
    vmax = np.nanpercentile(np.abs(hydra_grid), 95)
    im = ax.imshow(hydra_grid, origin='lower', aspect='auto', cmap='RdBu_r',
                   extent=[min(prey_births), max(prey_births), min(prey_deaths), max(prey_deaths[:-1])],
                   vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Prey Birth Rate', fontsize=12)
    ax.set_ylabel('Prey Death Rate', fontsize=12)
    ax.set_title('dN/d(prey_death) - Red = Hydra Effect', fontsize=11)
    ax.axhline(critical_prey_death, color='black', linestyle='--', linewidth=2)
    plt.colorbar(im, ax=ax, label='dN/dd')
    
    # Panel 2: Hydra effect strength vs prey_birth
    ax = axes[0, 1]
    hydra_strength = []
    for pb in prey_births:
        col_idx = prey_births.index(pb)
        hydra_vals = hydra_grid[:, col_idx]
        # Max positive dN/dd = strongest hydra
        max_hydra = np.nanmax(hydra_vals)
        hydra_strength.append(max_hydra if max_hydra > 0 else 0)
    
    ax.bar(prey_births, hydra_strength, width=0.08, color='coral', edgecolor='black')
    ax.set_xlabel('Prey Birth Rate', fontsize=12)
    ax.set_ylabel('Max Hydra Effect Strength (dN/dd)', fontsize=12)
    ax.set_title('Hydra Effect Strength by Prey Birth Rate', fontsize=11)
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Where does hydra effect occur? (prey_death values)
    ax = axes[1, 0]
    hydra_locations = []
    for j, pb in enumerate(prey_births):
        for i in range(len(prey_deaths)-1):
            if hydra_grid[i, j] > 0:  # Positive = hydra effect
                hydra_locations.append(prey_deaths[i])
    
    if hydra_locations:
        ax.hist(hydra_locations, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(critical_prey_death, color='green', linestyle='--', linewidth=2,
                   label=f'Critical point ({critical_prey_death})')
        ax.axvline(np.mean(hydra_locations), color='red', linestyle='-', linewidth=2,
                   label=f'Mean hydra location ({np.mean(hydra_locations):.3f})')
    ax.set_xlabel('Prey Death Rate', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Hydra Effect Locations', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Predator effect on hydra
    ax = axes[1, 1]
    # Check if hydra effect varies with predator parameters
    for pred_b in pred_births[::2]:
        hydra_by_pd = []
        for pd_val in prey_deaths[:-1]:
            subset = df[(df['predator_birth'] == pred_b) & (df['prey_death'] == pd_val)]
            subset_next = df[(df['predator_birth'] == pred_b) & (df['prey_death'] == prey_deaths[prey_deaths.index(pd_val)+1])]
            if len(subset) > 0 and len(subset_next) > 0:
                dN = subset_next['prey_mean'].mean() - subset['prey_mean'].mean()
                dd = prey_deaths[prey_deaths.index(pd_val)+1] - pd_val
                hydra_by_pd.append(dN / dd)
            else:
                hydra_by_pd.append(np.nan)
        ax.plot(prey_deaths[:-1], hydra_by_pd, 'o-', label=f'pred_b={pred_b:.1f}', alpha=0.7)
    
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(critical_prey_death, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Prey Death Rate', fontsize=12)
    ax.set_ylabel('dN/d(prey_death)', fontsize=12)
    ax.set_title('Hydra Effect by Predator Birth Rate', fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file2 = output_dir / "phase4_hydra_effect.png"
    plt.savefig(output_file2, dpi=dpi)
    plt.close()
    logging.info(f"Saved {output_file2}")
    
    # ==========================================================================
    # Figure 3: SOC Analysis (if evolution data present)
    # ==========================================================================
    if has_evolution:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Phase 4: Self-Organized Criticality Across Parameter Regimes", 
                     fontsize=14, fontweight='bold')
        
        df_evo = df[df['evolved_prey_death_final'].notna()].copy()
        
        # Panel 1: Evolved prey_death vs initial prey_death (by pred_birth)
        ax = axes[0, 0]
        for pred_b in pred_births[::2]:
            subset = df_evo[df_evo['predator_birth'] == pred_b]
            if len(subset) > 0:
                means = subset.groupby('prey_death')['evolved_prey_death_final'].mean()
                ax.plot(means.index, means.values, 'o-', label=f'pred_b={pred_b:.1f}', alpha=0.7)
        
        ax.plot([0, max(prey_deaths)], [0, max(prey_deaths)], 'k:', label='y=x')
        ax.axhline(critical_prey_death, color='green', linestyle='--', linewidth=2, 
                   label=f'd*={critical_prey_death}')
        ax.set_xlabel('Initial Prey Death Rate', fontsize=12)
        ax.set_ylabel('Final Evolved Prey Death Rate', fontsize=12)
        ax.set_title('SOC Test: Convergence by Predator Birth Rate', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Distribution of final evolved values by pred_birth
        ax = axes[0, 1]
        for pred_b in pred_births[::2]:
            subset = df_evo[df_evo['predator_birth'] == pred_b]
            if len(subset) > 0:
                ax.hist(subset['evolved_prey_death_final'], bins=15, alpha=0.5, 
                        label=f'pred_b={pred_b:.1f}')
        
        ax.axvline(critical_prey_death, color='green', linestyle='--', linewidth=2)
        ax.set_xlabel('Final Evolved Prey Death Rate', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Evolved Values by pred_birth', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Mean evolved prey_death vs predator parameters
        ax = axes[1, 0]
        evolved_by_pred = df_evo.groupby(['predator_birth', 'predator_death'])['evolved_prey_death_final'].mean()
        evolved_grid = np.zeros((len(pred_deaths), len(pred_births)))
        
        for i, pred_d in enumerate(pred_deaths):
            for j, pred_b in enumerate(pred_births):
                if (pred_b, pred_d) in evolved_by_pred.index:
                    evolved_grid[i, j] = evolved_by_pred[(pred_b, pred_d)]
                else:
                    evolved_grid[i, j] = np.nan
        
        im = ax.imshow(evolved_grid, origin='lower', aspect='auto', cmap='viridis',
                       extent=[min(pred_births), max(pred_births), min(pred_deaths), max(pred_deaths)])
        ax.set_xlabel('Predator Birth Rate', fontsize=12)
        ax.set_ylabel('Predator Death Rate', fontsize=12)
        ax.set_title('Mean Evolved prey_death by Predator Parameters', fontsize=11)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Evolved prey_death')
        
        # Panel 4: Correlation between critical point and evolved value
        ax = axes[1, 1]
        # Group by all 4 params and compute mean evolved value
        grouped = df_evo.groupby(['prey_birth', 'predator_birth', 'predator_death']).agg({
            'evolved_prey_death_final': 'mean',
            'prey_mean': 'mean'
        }).reset_index()
        
        ax.scatter(grouped['evolved_prey_death_final'], grouped['prey_mean'], 
                   alpha=0.5, s=30, c='steelblue')
        ax.axvline(critical_prey_death, color='green', linestyle='--', linewidth=2,
                   label=f'Critical d*={critical_prey_death}')
        ax.set_xlabel('Mean Evolved Prey Death Rate', fontsize=12)
        ax.set_ylabel('Mean Prey Population', fontsize=12)
        ax.set_title('Population vs Evolved Trait Across Regimes', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file3 = output_dir / "phase4_soc_sensitivity.png"
        plt.savefig(output_file3, dpi=dpi)
        plt.close()
        logging.info(f"Saved {output_file3}")
    
    # ==========================================================================
    # Generate Summary Statistics
    # ==========================================================================
    summary = {
        'n_simulations': len(df),
        'parameters': {
            'prey_birth': {'min': min(prey_births), 'max': max(prey_births), 'n': len(prey_births)},
            'prey_death': {'min': min(prey_deaths), 'max': max(prey_deaths), 'n': len(prey_deaths)},
            'predator_birth': {'min': min(pred_births), 'max': max(pred_births), 'n': len(pred_births)},
            'predator_death': {'min': min(pred_deaths), 'max': max(pred_deaths), 'n': len(pred_deaths)},
        },
        'coexistence': {
            'overall_rate': float(((df['prey_survived'] == True) & (df['pred_survived'] == True)).mean() * 100),
            'prey_survival_rate': float((df['prey_survived'] == True).mean() * 100),
            'pred_survival_rate': float((df['pred_survived'] == True).mean() * 100),
        },
        'hydra_effect': {
            'detected': bool(len(hydra_locations) > 0) if hydra_locations else False,
            'mean_location': float(np.mean(hydra_locations)) if hydra_locations else None,
            'max_strength': float(np.nanmax(hydra_grid)) if np.any(hydra_grid > 0) else 0,
        },
        'has_evolution_data': has_evolution,
    }
    
    if has_evolution:
        summary['soc'] = {
            'mean_evolved_prey_death': float(df_evo['evolved_prey_death_final'].mean()),
            'std_evolved_prey_death': float(df_evo['evolved_prey_death_final'].std()),
            'distance_from_critical': float(abs(df_evo['evolved_prey_death_final'].mean() - critical_prey_death)),
        }
    
    summary_file = output_dir / "phase4_sensitivity_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved {summary_file}")
    
    # Log key findings
    logging.info("Phase 4 Sensitivity Analysis Summary:")
    logging.info(f"  Total simulations: {summary['n_simulations']}")
    logging.info(f"  Overall coexistence rate: {summary['coexistence']['overall_rate']:.1f}%")
    logging.info(f"  Hydra effect detected: {summary['hydra_effect']['detected']}")
    if summary['hydra_effect']['mean_location']:
        logging.info(f"  Mean hydra location: {summary['hydra_effect']['mean_location']:.4f}")
    if has_evolution:
        logging.info(f"  Mean evolved prey_death: {summary['soc']['mean_evolved_prey_death']:.4f}")
        logging.info(f"  Distance from critical: {summary['soc']['distance_from_critical']:.4f}")
    
    return output_file1


def generate_summary_report(grids: Dict, dN_dd_no_evo: np.ndarray, 
                            prey_births: np.ndarray, prey_deaths: np.ndarray,
                            output_dir: Path):
    """Generate summary statistics JSON."""
    summary = {
        "coexistence_no_evo": int(np.sum(grids["survival_prey_no_evo"] > 80)),
        "hydra_region_size": int(np.sum((dN_dd_no_evo > 0) & (grids["prey_pop_no_evo"] > 50))),
        "max_hydra_strength": float(np.nanmax(dN_dd_no_evo)),
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
    
    output_file = output_dir / "summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Saved {output_file}")
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from PP evolutionary analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results/                    # Generate all plots
  %(prog)s results/ --phase-only       # Only phase diagrams
  %(prog)s results/ --dpi 300          # High-resolution output
  %(prog)s results/ --fss-only         # Only FSS plots
        """
    )
    
    parser.add_argument('results_dir', type=Path,
                       help='Directory containing analysis results')
    parser.add_argument('--phase-only', action='store_true',
                       help='Generate only phase diagrams')
    parser.add_argument('--hydra-only', action='store_true',
                       help='Generate only Hydra analysis plots')
    parser.add_argument('--pcf-only', action='store_true',
                       help='Generate only PCF analysis plots')
    parser.add_argument('--fss-only', action='store_true',
                       help='Generate only FSS plots')
    parser.add_argument('--sensitivity-only', action='store_true',
                       help='Generate only sensitivity analysis plots')
    parser.add_argument('--bifurcation-only', action='store_true',
                       help='Generate only bifurcation diagram')
    parser.add_argument('--phase2-only', action='store_true',
                       help='Generate only Phase 2 SOC analysis plots')
    parser.add_argument('--phase3-only', action='store_true',
                       help='Generate only Phase 3 FSS analysis plots')
    parser.add_argument('--phase4-only', action='store_true',
                       help='Generate only Phase 4 sensitivity analysis plots')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Output resolution (default: 150)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output directory (default: same as results_dir)')
    
    args = parser.parse_args()
    
    # Setup
    results_dir = args.results_dir
    output_dir = args.output if args.output else results_dir
    output_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    logging.info("=" * 60)
    logging.info("PP ANALYSIS PLOTTING")
    logging.info("=" * 60)
    logging.info(f"Results: {results_dir}")
    logging.info(f"Output: {output_dir}")
    logging.info(f"DPI: {args.dpi}")
    
    # Determine what to plot
    plot_all = not any([args.phase_only, args.hydra_only, args.pcf_only,
                        args.fss_only, args.sensitivity_only, args.bifurcation_only,
                        args.phase2_only, args.phase3_only, args.phase4_only])
    
    # Main sweep plots
    if plot_all or args.phase_only or args.hydra_only or args.pcf_only:
        try:
            results = load_sweep_results(results_dir)
            config = load_config(results_dir)
            
            logging.info(f"Loaded {len(results)} sweep results")
            
            prey_births, prey_deaths = extract_parameter_grid(results, config)
            logging.info(f"Grid: {len(prey_births)} × {len(prey_deaths)}")
            
            grids = aggregate_to_grids(results, prey_births, prey_deaths)
            dN_dd_no_evo, dN_dd_evo = compute_hydra_derivative(grids, prey_deaths)
            
            if plot_all or args.phase_only:
                plot_phase_diagrams(grids, prey_births, prey_deaths, 
                                   dN_dd_no_evo, output_dir, args.dpi)
            
            if plot_all or args.hydra_only:
                plot_hydra_analysis(grids, prey_births, prey_deaths,
                                   dN_dd_no_evo, dN_dd_evo, output_dir, args.dpi)
            
            if plot_all or args.pcf_only:
                plot_pcf_analysis(grids, prey_births, prey_deaths,
                                 dN_dd_no_evo, output_dir, args.dpi)
            
            if plot_all:
                summary = generate_summary_report(grids, dN_dd_no_evo,
                                                 prey_births, prey_deaths, output_dir)
                logging.info("SUMMARY:")
                logging.info(f"  Hydra region size: {summary['hydra_region_size']}")
                logging.info(f"  Max Hydra strength: {summary['max_hydra_strength']:.1f}")
                if 'critical_prey_birth' in summary:
                    logging.info(f"  Critical point: pb={summary['critical_prey_birth']:.3f}, "
                               f"pd={summary['critical_prey_death']:.3f}")
        
        except FileNotFoundError as e:
            logging.error(f"Sweep results not found: {e}")
    
    # FSS plots
    if plot_all or args.fss_only:
        try:
            fss_results = load_fss_results(results_dir)
            logging.info(f"Loaded {len(fss_results)} FSS results")
            plot_fss_analysis(fss_results, output_dir, args.dpi)
        except FileNotFoundError as e:
            logging.warning(f"FSS results not found: {e}")
    
    # Sensitivity plots
    if plot_all or args.sensitivity_only:
        try:
            sens_results = load_sensitivity_results(results_dir)
            logging.info(f"Loaded {len(sens_results)} sensitivity results")
            plot_sensitivity_analysis(sens_results, output_dir, args.dpi)
        except FileNotFoundError as e:
            logging.warning(f"Sensitivity results not found: {e}")
    
    # Bifurcation diagram
    if plot_all or args.bifurcation_only:
        try:
            sweep_params, bifurc_results = load_bifurcation_results(results_dir)
            logging.info(f"Loaded bifurcation results: {len(sweep_params)} sweep values, "
                        f"{bifurc_results.shape[1]} replicates each")
            plot_bifurcation_diagram(sweep_params, bifurc_results, output_dir, args.dpi)
        except FileNotFoundError as e:
            logging.warning(f"Bifurcation results not found: {e}")
    
    # Phase 2: SOC analysis
    if plot_all or args.phase2_only:
        try:
            phase2_results = load_phase2_results(results_dir)
            logging.info(f"Loaded {len(phase2_results)} Phase 2 (SOC) results")
            plot_phase2_soc_analysis(phase2_results, output_dir, args.dpi)
        except FileNotFoundError as e:
            logging.warning(f"Phase 2 results not found: {e}")
    
    # Phase 3: FSS analysis
    if plot_all or args.phase3_only:
        try:
            phase3_results = load_phase3_results(results_dir)
            logging.info(f"Loaded {len(phase3_results)} Phase 3 (FSS) results")
            plot_phase3_fss_analysis(phase3_results, output_dir, args.dpi)
        except FileNotFoundError as e:
            logging.warning(f"Phase 3 results not found: {e}")
    
    # Phase 4: Sensitivity analysis
    if plot_all or args.phase4_only:
        try:
            phase4_results = load_phase4_results(results_dir)
            logging.info(f"Loaded {len(phase4_results)} Phase 4 (Sensitivity) results")
            plot_phase4_sensitivity_analysis(phase4_results, output_dir, args.dpi)
        except FileNotFoundError as e:
            logging.warning(f"Phase 4 results not found: {e}")
    
    logging.info("Done!")


if __name__ == "__main__":
    main()