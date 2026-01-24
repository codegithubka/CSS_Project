#!/usr/bin/env python3
"""
Post-analysis plotting for predator-prey evolutionary simulations.

Reads saved results from pp_analysis.py and generate figures.
Designed to run locally (not on HPC) for fast iteration.

Usage:
    python plot_pp_results.py results/               # All plots
    python plot_pp_results.py results/ --phase-only  # Just phase diagrams
    python plot_pp_results.py results/ --hydra-only  # Just Hydra analysis
    python plot_pp_results.py results/ --pcf-only    # Just PCF analysis
    python plot_pp_results.py results/ --fss-only    # Just FSS plots
    python plot_pp_results.py results/ --dpi 300     # High-res for publication
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
                        args.fss_only, args.sensitivity_only])
    
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
    
    logging.info("Done!")


if __name__ == "__main__":
    main()