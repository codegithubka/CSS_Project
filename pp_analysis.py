#!/usr/bin/env python3
"""
PP Evolutionary Analysis - Snellius HPC Version
================================================

Comprehensive analysis of predator-prey cellular automaton with evolutionary dynamics.

Analyses:
1. 2D parameter sweep (prey_birth × predator_death)
2. Hydra effect quantification (dP/dd derivative)
3. Critical point search (τ ≈ 2.05)
4. Evolution sensitivity (SD, min, max sweeps)
5. Finite-size scaling (multiple grid sizes)

Usage:
    python pp_analysis.py --mode full          # Run everything
    python pp_analysis.py --mode sweep         # Only 2D sweep
    python pp_analysis.py --mode sensitivity   # Only evolution sensitivity
    python pp_analysis.py --mode fss           # Only finite-size scaling
    python pp_analysis.py --mode plot          # Only generate plots from saved data
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

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class Config:
    """
    Central configuration - adjust these for your CPU budget.

    RESOURCE PROFILES:
    -----------------
    MINIMAL (~15 core-hours):
        n_prey_birth=10, n_pred_death=10, n_replicates=15, default_grid=100

    STANDARD (~40 core-hours):
        n_prey_birth=15, n_pred_death=15, n_replicates=25, default_grid=100

    HIGH-QUALITY (~80 core-hours):
        n_prey_birth=15, n_pred_death=15, n_replicates=25, default_grid=150

    PUBLICATION (~150 core-hours):
        n_prey_birth=20, n_pred_death=20, n_replicates=30, default_grid=150
    """

    # Grid settings
    default_grid: int = 100
    densities: Tuple[float, float] = (0.30, 0.15)

    # 2D sweep resolution
    n_prey_birth: int = 15
    n_pred_death: int = 15
    prey_birth_min: float = 0.15
    prey_birth_max: float = 0.50
    pred_death_min: float = 0.02
    pred_death_max: float = 0.22

    # Replicates per parameter combination
    n_replicates: int = 25

    # Simulation length
    # Warmup: system reaches quasi-stationary state
    # Measurement: collect population/cluster data
    warmup_steps: int = 200
    measurement_steps: int = 300
    cluster_samples: int = 40
    cluster_interval: int = 8

    # Fixed ecological parameters
    prey_death: float = 0.05
    predator_birth: float = 0.20

    # Evolution parameters (defaults)
    evolve_sd: float = 0.015
    evolve_min: float = 0.005
    evolve_max: float = 0.40

    # Finite-size scaling
    # Theory: s_c ~ L^1.9, τ should be constant (~2.05)
    fss_grid_sizes: Tuple[int, ...] = (50, 75, 100, 150)
    fss_replicates: int = 20

    # Evolution sensitivity
    # Tests how mutation rate affects adaptation
    sensitivity_sd_values: Tuple[float, ...] = (0.005, 0.010, 0.020, 0.035, 0.050)
    sensitivity_replicates: int = 20

    # Parallel settings
    n_jobs: int = -1  # -1 = all available cores
    
    synchronous: bool = True  # Use synchronous/async updates

    def get_prey_births(self) -> np.ndarray:
        return np.linspace(self.prey_birth_min, self.prey_birth_max, self.n_prey_birth)

    def get_pred_deaths(self) -> np.ndarray:
        return np.linspace(self.pred_death_min, self.pred_death_max, self.n_pred_death)

    def estimate_runtime(self, n_cores: int = 32) -> str:
        """Estimate total runtime with grid-size scaling."""
        n_sweep = self.n_prey_birth * self.n_pred_death * self.n_replicates * 2
        n_sens = len(self.sensitivity_sd_values) * self.sensitivity_replicates
        n_fss = sum(self.fss_replicates for _ in self.fss_grid_sizes)

        # Time per sim scales as (L/100)^2
        grid_factor = (self.default_grid / 100) ** 2
        base_time = 1.5  # seconds for 100×100

        # Sweep and sensitivity use default_grid
        sweep_time = n_sweep * base_time * grid_factor
        sens_time = n_sens * base_time * grid_factor

        # FSS uses variable grid sizes
        fss_time = sum(
            self.fss_replicates * base_time * (L / 100) ** 2
            for L in self.fss_grid_sizes
        )

        total_seconds = (sweep_time + sens_time + fss_time) / n_cores
        hours = total_seconds / 3600
        core_hours = (sweep_time + sens_time + fss_time) / 3600

        return f"{n_sweep + n_sens + n_fss:,} sims, ~{hours:.1f}h on {n_cores} cores (~{core_hours:.0f} core-hours)"


# Core functionality# =============================================================================


def count_populations(grid: np.ndarray) -> Tuple[int, int, int]:
    """Count empty, prey, predator cells."""
    return int(np.sum(grid == 0)), int(np.sum(grid == 1)), int(np.sum(grid == 2))


def measure_cluster_sizes(grid: np.ndarray, species: int) -> np.ndarray:
    """Extract cluster sizes using 4-connected component analysis."""
    binary_mask = (grid == species).astype(int)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, n = ndimage.label(binary_mask, structure=structure)
    if n == 0:
        return np.array([], dtype=int)
    return np.array(ndimage.sum(binary_mask, labeled, range(1, n + 1)), dtype=int)


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
    """Truncated power law: P(s) = A * s^(-tau) * exp(-s/s_c)"""
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
            x,
            np.log(y + 1e-20),
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


# Sim runner # =============================================================================


def run_single_simulation(
    prey_birth: float,
    pred_death: float,
    grid_size: int,
    seed: int,
    with_evolution: bool,
    cfg: Config,
    evolve_sd: Optional[float] = None,
    evolve_min: Optional[float] = None,
    evolve_max: Optional[float] = None,
) -> Dict:
    """
    Run a single PP simulation and collect all metrics.

    This is the unit of parallelization.
    """
    # Import here to avoid issues with multiprocessing
    from models.CA import PP

    if evolve_sd is None:
        evolve_sd = cfg.evolve_sd
    if evolve_min is None:
        evolve_min = cfg.evolve_min
    if evolve_max is None:
        evolve_max = cfg.evolve_max

    params = {
        "prey_birth": prey_birth,
        "predator_death": pred_death,
        "prey_death": cfg.prey_death,
        "predator_birth": cfg.predator_birth,
    }

    model = PP(
        rows=grid_size,
        cols=grid_size,
        densities=cfg.densities,
        neighborhood="moore",
        params=params,
        seed=seed,
        syncrhonous = cfg.synchronous # Use synchronous/async updates
    )

    if with_evolution:
        model.evolve("predator_death", sd=evolve_sd, min=evolve_min, max=evolve_max)

    # Warmup
    model.run(cfg.warmup_steps)

    # Measurement
    prey_pops, pred_pops, evolved_vals = [], [], []
    prey_clusters, pred_clusters = [], []
    sample_counter = 0

    for step in range(cfg.measurement_steps):
        model.update()
        _, prey, pred = count_populations(model.grid)
        prey_pops.append(prey)
        pred_pops.append(pred)

        if with_evolution:
            stats = get_evolved_stats(model, "predator_death")
            evolved_vals.append(stats["mean"])

        # Cluster sampling (periodic)
        if step % cfg.cluster_interval == 0 and sample_counter < cfg.cluster_samples:
            if pred > 10:
                prey_clusters.extend(measure_cluster_sizes(model.grid, 1))
                pred_clusters.extend(measure_cluster_sizes(model.grid, 2))
            sample_counter += 1

    # Compile results
    result = {
        "prey_birth": prey_birth,
        "pred_death": pred_death,
        "grid_size": grid_size,
        "with_evolution": with_evolution,
        "seed": seed,
        "prey_mean": float(np.mean(prey_pops)),
        "prey_std": float(np.std(prey_pops)),
        "pred_mean": float(np.mean(pred_pops)),
        "pred_std": float(np.std(pred_pops)),
        "pred_survived": bool(np.mean(pred_pops) > 10),
        "prey_n_clusters": len(prey_clusters),
        "pred_n_clusters": len(pred_clusters),
    }

    # Evolved parameter
    if with_evolution and evolved_vals:
        valid_evolved = [v for v in evolved_vals if not np.isnan(v)]
        result["evolved_death_mean"] = (
            float(np.mean(valid_evolved)) if valid_evolved else np.nan
        )
        result["evolved_death_std"] = (
            float(np.std(valid_evolved)) if valid_evolved else np.nan
        )
        result["evolve_sd"] = evolve_sd

    # Cluster power-law fits
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

    return result


# Analysis Runners # =============================================================================


def run_2d_sweep(cfg: Config, output_dir: Path, logger: logging.Logger) -> List[Dict]:
    """Run full 2D parameter sweep with and without evolution."""
    from joblib import Parallel, delayed

    prey_births = cfg.get_prey_births()
    pred_deaths = cfg.get_pred_deaths()

    # Generate jobs
    jobs = []
    for pb in prey_births:
        for pd in pred_deaths:
            for rep in range(cfg.n_replicates):
                seed_base = int(pb * 1000) + int(pd * 10000) + rep
                jobs.append(
                    (pb, pd, cfg.default_grid, seed_base, False)
                )  # No evolution
                jobs.append(
                    (pb, pd, cfg.default_grid, seed_base, True)
                )  # With evolution

    logger.info(f"2D Sweep: {len(jobs):,} simulations")
    logger.info(f"  Grid: {len(prey_births)}×{len(pred_deaths)} parameters")
    logger.info(f"  Replicates: {cfg.n_replicates}")

    # Run parallel
    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(run_single_simulation)(pb, pd, gs, seed, evo, cfg)
        for pb, pd, gs, seed, evo in jobs
    )

    # Save
    output_file = output_dir / "sweep_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f)
    logger.info(f"Saved to {output_file}")

    return results


def run_sensitivity(
    cfg: Config, output_dir: Path, logger: logging.Logger
) -> List[Dict]:
    """Run evolution parameter sensitivity analysis."""
    from joblib import Parallel, delayed

    # Fixed parameters in transition zone
    pb_test = 0.30
    pd_test = 0.14

    jobs = []
    for sd in cfg.sensitivity_sd_values:
        for rep in range(cfg.sensitivity_replicates):
            seed = int(sd * 100000) + rep
            jobs.append((pb_test, pd_test, cfg.default_grid, seed, True, sd))

    logger.info(f"Sensitivity: {len(jobs)} simulations")
    logger.info(f"  SD values: {cfg.sensitivity_sd_values}")

    results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(run_single_simulation)(pb, pd, gs, seed, evo, cfg, evolve_sd=sd)
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

    # Fixed parameters in coexistence region
    pb_test = 0.35
    pd_test = 0.08

    jobs = []
    for L in cfg.fss_grid_sizes:
        for rep in range(cfg.fss_replicates):
            seed = L * 1000 + rep
            jobs.append((pb_test, pd_test, L, seed, False))

    logger.info(f"FSS: {len(jobs)} simulations")
    logger.info(f"  Grid sizes: {cfg.fss_grid_sizes}")

    results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(run_single_simulation)(pb, pd, gs, seed, evo, cfg)
        for pb, pd, gs, seed, evo in jobs
    )

    output_file = output_dir / "fss_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f)
    logger.info(f"Saved to {output_file}")

    return results


# Plotting and Summary # =============================================================================


def generate_plots(cfg: Config, output_dir: Path, logger: logging.Logger):
    """Generate all analysis plots from saved data."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    plt.rcParams["figure.figsize"] = (14, 10)
    plt.rcParams["font.size"] = 11

    prey_births = cfg.get_prey_births()
    pred_deaths = cfg.get_pred_deaths()
    n_pb, n_pd = len(prey_births), len(pred_deaths)
    extent = [prey_births[0], prey_births[-1], pred_deaths[0], pred_deaths[-1]]

    # Load sweep results
    sweep_file = output_dir / "sweep_results.json"
    if not sweep_file.exists():
        logger.error(f"Sweep results not found: {sweep_file}")
        return

    with open(sweep_file, "r") as f:
        results = json.load(f)

    logger.info(f"Loaded {len(results)} results from {sweep_file}")

    # Aggregate into grids
    grids = {
        "pred_pop_no_evo": np.full((n_pd, n_pb), np.nan),
        "pred_pop_evo": np.full((n_pd, n_pb), np.nan),
        "survival_no_evo": np.full((n_pd, n_pb), np.nan),
        "survival_evo": np.full((n_pd, n_pb), np.nan),
        "tau_prey": np.full((n_pd, n_pb), np.nan),
        "tau_pred": np.full((n_pd, n_pb), np.nan),
        "evolved_death": np.full((n_pd, n_pb), np.nan),
    }

    # Group by parameters
    grouped = defaultdict(list)
    for r in results:
        key = (
            round(r["prey_birth"], 4),
            round(r["pred_death"], 4),
            r["with_evolution"],
        )
        grouped[key].append(r)

    for i, pd in enumerate(pred_deaths):
        for j, pb in enumerate(prey_births):
            pd_r, pb_r = round(pd, 4), round(pb, 4)

            no_evo = grouped.get((pb_r, pd_r, False), [])
            if no_evo:
                grids["pred_pop_no_evo"][i, j] = np.mean(
                    [r["pred_mean"] for r in no_evo]
                )
                grids["survival_no_evo"][i, j] = (
                    np.mean([r["pred_survived"] for r in no_evo]) * 100
                )
                taus = [
                    r["prey_tau"]
                    for r in no_evo
                    if not np.isnan(r.get("prey_tau", np.nan))
                ]
                if taus:
                    grids["tau_prey"][i, j] = np.mean(taus)
                taus = [
                    r["pred_tau"]
                    for r in no_evo
                    if not np.isnan(r.get("pred_tau", np.nan))
                ]
                if taus:
                    grids["tau_pred"][i, j] = np.mean(taus)

            evo = grouped.get((pb_r, pd_r, True), [])
            if evo:
                grids["pred_pop_evo"][i, j] = np.mean([r["pred_mean"] for r in evo])
                grids["survival_evo"][i, j] = (
                    np.mean([r["pred_survived"] for r in evo]) * 100
                )
                evolved = [r.get("evolved_death_mean", np.nan) for r in evo]
                evolved = [e for e in evolved if not np.isnan(e)]
                if evolved:
                    grids["evolved_death"][i, j] = np.mean(evolved)

    # Compute Hydra derivative
    dd = pred_deaths[1] - pred_deaths[0]
    dP_dd_evo = np.zeros_like(grids["pred_pop_evo"])
    for j in range(n_pb):
        pop_smooth = gaussian_filter1d(grids["pred_pop_evo"][:, j], sigma=0.8)
        dP_dd_evo[:, j] = np.gradient(pop_smooth, dd)

    # =========================================================================
    # PLOT 1: Phase Diagrams
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    im = ax.imshow(
        grids["pred_pop_no_evo"],
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="YlOrRd",
    )
    ax.contour(
        prey_births,
        pred_deaths,
        grids["survival_no_evo"],
        levels=[50],
        colors="black",
        linewidths=2,
    )
    plt.colorbar(im, ax=ax, label="Population")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Predator Death Rate")
    ax.set_title("Predator Pop (No Evolution)\nBlack: 50% survival")

    ax = axes[0, 1]
    im = ax.imshow(
        grids["pred_pop_evo"],
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="YlOrRd",
    )
    ax.contour(
        prey_births,
        pred_deaths,
        grids["survival_evo"],
        levels=[50],
        colors="black",
        linewidths=2,
    )
    plt.colorbar(im, ax=ax, label="Population")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Predator Death Rate")
    ax.set_title("Predator Pop (With Evolution)\nBlack: 50% survival")

    ax = axes[0, 2]
    advantage = np.where(
        grids["pred_pop_no_evo"] > 10,
        (grids["pred_pop_evo"] - grids["pred_pop_no_evo"])
        / grids["pred_pop_no_evo"]
        * 100,
        np.where(grids["pred_pop_evo"] > 10, 500, 0),
    )
    im = ax.imshow(
        np.clip(advantage, -50, 200),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="RdYlGn",
        vmin=-50,
        vmax=200,
    )
    plt.colorbar(im, ax=ax, label="Advantage (%)")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Predator Death Rate")
    ax.set_title("Evolution Advantage\n(Evo - NoEvo) / NoEvo")

    ax = axes[1, 0]
    im = ax.imshow(
        grids["tau_prey"],
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=1.5,
        vmax=2.5,
    )
    ax.contour(
        prey_births,
        pred_deaths,
        grids["tau_prey"],
        levels=[2.05],
        colors="green",
        linewidths=2,
    )
    plt.colorbar(im, ax=ax, label="τ")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Predator Death Rate")
    ax.set_title("Prey τ (Green: τ=2.05)")

    ax = axes[1, 1]
    im = ax.imshow(
        grids["tau_pred"],
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=1.5,
        vmax=2.5,
    )
    ax.contour(
        prey_births,
        pred_deaths,
        grids["tau_pred"],
        levels=[2.05],
        colors="green",
        linewidths=2,
    )
    plt.colorbar(im, ax=ax, label="τ")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Predator Death Rate")
    ax.set_title("Predator τ (Green: τ=2.05)")

    ax = axes[1, 2]
    im = ax.imshow(
        dP_dd_evo,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="RdBu_r",
        vmin=-5000,
        vmax=5000,
    )
    ax.contour(
        prey_births, pred_deaths, dP_dd_evo, levels=[0], colors="black", linewidths=2
    )
    plt.colorbar(im, ax=ax, label="dP/dd")
    ax.set_xlabel("Prey Birth Rate")
    ax.set_ylabel("Predator Death Rate")
    ax.set_title("Hydra Effect: dP/dd\nRed: Pop ↑ with mortality")

    plt.tight_layout()
    plt.savefig(output_dir / "phase_diagrams.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved phase_diagrams.png")

    # =========================================================================
    # PLOT 2: Sensitivity Analysis
    # =========================================================================
    sens_file = output_dir / "sensitivity_results.json"
    if sens_file.exists():
        with open(sens_file, "r") as f:
            sens_results = json.load(f)

        sens_grouped = defaultdict(list)
        for r in sens_results:
            sens_grouped[r.get("evolve_sd", cfg.evolve_sd)].append(r)

        fig, ax = plt.subplots(figsize=(10, 6))
        sd_vals = sorted(sens_grouped.keys())
        pops = [np.mean([r["pred_mean"] for r in sens_grouped[sd]]) for sd in sd_vals]
        stds = [np.std([r["pred_mean"] for r in sens_grouped[sd]]) for sd in sd_vals]

        ax.errorbar(
            sd_vals, pops, yerr=stds, fmt="o-", capsize=5, markersize=8, linewidth=2
        )
        ax.set_xlabel("Evolution SD", fontsize=12)
        ax.set_ylabel("Mean Predator Population", fontsize=12)
        ax.set_title(
            "Evolution Sensitivity Analysis\n(prey_birth=0.30, pred_death=0.14)",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "sensitivity.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved sensitivity.png")

    # =========================================================================
    # PLOT 3: FSS (if data exists)
    # =========================================================================
    fss_file = output_dir / "fss_results.json"
    if fss_file.exists():
        with open(fss_file, "r") as f:
            fss_results = json.load(f)

        fss_grouped = defaultdict(list)
        for r in fss_results:
            fss_grouped[r["grid_size"]].append(r)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        Ls = sorted(fss_grouped.keys())
        tau_prey = [np.nanmean([r["prey_tau"] for r in fss_grouped[L]]) for L in Ls]
        tau_pred = [np.nanmean([r["pred_tau"] for r in fss_grouped[L]]) for L in Ls]
        s_c_prey = [np.nanmean([r["prey_s_c"] for r in fss_grouped[L]]) for L in Ls]

        ax = axes[0]
        ax.plot(Ls, tau_prey, "go-", markersize=10, label="Prey τ")
        ax.plot(Ls, tau_pred, "rs-", markersize=10, label="Predator τ")
        ax.axhline(2.05, color="black", linestyle="--", label="Theory τ=2.05")
        ax.set_xlabel("Grid Size L")
        ax.set_ylabel("Exponent τ")
        ax.set_title("Power Law Exponent vs Grid Size")
        ax.legend()
        ax.set_ylim(1.5, 2.6)

        ax = axes[1]
        valid = [(L, sc) for L, sc in zip(Ls, s_c_prey) if not np.isnan(sc) and sc > 0]
        if len(valid) >= 2:
            Ls_v, sc_v = zip(*valid)
            ax.scatter(Ls_v, sc_v, s=100, c="green", edgecolors="black")
            slope, intercept = np.polyfit(np.log(Ls_v), np.log(sc_v), 1)
            L_line = np.linspace(min(Ls) * 0.9, max(Ls) * 1.1, 50)
            ax.plot(
                L_line,
                np.exp(intercept) * L_line**slope,
                "g--",
                label=f"Fit: $s_c \\sim L^{{{slope:.2f}}}$",
            )
            ax.plot(
                L_line,
                (L_line / 100) ** 1.896 * 1000,
                "k:",
                alpha=0.5,
                label="Theory: $L^{1.90}$",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Grid Size L")
        ax.set_ylabel("Cutoff $s_c$")
        ax.set_title("Finite-Size Scaling")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "fss.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved fss.png")

    # =========================================================================
    # Summary statistics
    # =========================================================================
    summary = {
        "coexistence_no_evo": int(np.sum(grids["survival_no_evo"] > 80)),
        "coexistence_evo": int(np.sum(grids["survival_evo"] > 80)),
        "hydra_region_size": int(
            np.sum((dP_dd_evo > 0) & (grids["pred_pop_evo"] > 50))
        ),
        "max_hydra_strength": float(np.nanmax(dP_dd_evo)),
    }

    # Find critical point
    dist_crit = np.sqrt(
        (grids["tau_prey"] - 2.05) ** 2 + (grids["tau_pred"] - 2.05) ** 2
    )
    if not np.all(np.isnan(dist_crit)):
        min_idx = np.unravel_index(np.nanargmin(dist_crit), dist_crit.shape)
        summary["critical_prey_birth"] = float(prey_births[min_idx[1]])
        summary["critical_pred_death"] = float(pred_deaths[min_idx[0]])
        summary["critical_tau_prey"] = float(grids["tau_prey"][min_idx])
        summary["critical_tau_pred"] = float(grids["tau_pred"][min_idx])

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary.json")

    # Print summary
    logger.info("=" * 60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Coexistence region (>80% survival):")
    logger.info(
        f"  Without evolution: {summary['coexistence_no_evo']} parameter combinations"
    )
    logger.info(
        f"  With evolution: {summary['coexistence_evo']} parameter combinations"
    )
    logger.info(
        f"Hydra effect region (dP/dd > 0): {summary['hydra_region_size']} combinations"
    )
    if "critical_prey_birth" in summary:
        logger.info(f"Closest to criticality:")
        logger.info(f"  prey_birth = {summary['critical_prey_birth']:.3f}")
        logger.info(f"  pred_death = {summary['critical_pred_death']:.3f}")
        logger.info(
            f"  τ_prey = {summary['critical_tau_prey']:.3f}, τ_pred = {summary['critical_tau_pred']:.3f}"
        )


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="PP Evolutionary Analysis")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "sweep", "sensitivity", "fss", "plot"],
        help="Analysis mode",
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--cores", type=int, default=-1, help="Number of cores (-1 = all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print estimated runtime and exit"
    )
    
    parser.add_argument(
        "--async", 
        action="store_false", 
        dest="synchronous", 
        help="Run simulation in asynchronous (random-sequential) mode"
    )
    
    args = parser.parse_args()

    # Setup
    cfg = Config()
    cfg.synchronous = args.synchronous
    cfg.n_jobs = (
        args.cores if args.cores > 0 else int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
    )

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
    logger.info("PP Evolutionary Analysis")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Cores: {cfg.n_jobs}")

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
