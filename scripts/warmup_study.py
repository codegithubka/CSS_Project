#!/usr/bin/env python3
"""
Study warmup period cost as a function of grid size.

Measures how equilibration time scales with system size L for the
predator-prey cellular automaton. Key metrics:
- Wall-clock time per simulation step
- Number of steps to reach equilibrium
- Total warmup cost (time × steps)

Usage:
    python warmup_study.py                           # Default grid sizes
    python warmup_study.py --sizes 50 100 150 200    # Custom sizes
    python warmup_study.py --replicates 20           # More replicates
    python warmup_study.py --output results/warmup/  # Custom output dir
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for module imports
project_root = str(Path(__file__).parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Configure matplotlib
plt.rcParams.update({
    'figure.figsize': (15, 5),
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
# CONFIGURATION
# =============================================================================

@dataclass
class WarmupStudyConfig:
    """Configuration for warmup cost study."""
    
    # Grid sizes to test
    grid_sizes: Tuple[int, ...] = (50, 75, 100, 150, 200)
    
    # Number of independent replicates per grid size
    n_replicates: int = 10
    
    # Maximum steps to run (should be enough for largest grid to equilibrate)
    max_steps: int = 2000
    
    # How often to sample population (steps)
    sample_interval: int = 10
    
    # Equilibration detection parameters
    equilibration_window: int = 50  # FFT window size (needs to capture oscillation periods)
    
    # Simulation parameters (near critical point)
    prey_birth: float = 0.25
    prey_death: float = 0.05
    predator_birth: float = 0.2
    predator_death: float = 0.1
    densities: Tuple[float, float] = (0.2, 0.1)
    
    # Update mode
    synchronous: bool = False
    directed_hunting: bool = True


# =============================================================================
# EQUILIBRATION DETECTION
# =============================================================================

def estimate_equilibration_frequency(
    time_series: np.ndarray,
    sample_interval: int,
    grid_size: int = 100,
    base_window: int = 50,
    n_stable_windows: int = 3,
    frequency_tolerance: float = 0.2,
) -> int:
    """
    Detect equilibration when a characteristic oscillation frequency dominates.
    
    Uses spectral analysis (FFT) on sliding windows to find the dominant
    frequency. Equilibrium is detected when the dominant frequency stabilizes
    (stops changing significantly between consecutive windows).
    
    Parameters
    ----------
    time_series : np.ndarray
        Population density or count over time.
    sample_interval : int
        Number of simulation steps between samples.
    grid_size : int
        Size of the grid (L). Window size scales with grid size.
    base_window : int
        Base FFT window size (number of samples) for L=100.
        Needs to be large enough to capture oscillation periods.
    n_stable_windows : int
        Number of consecutive windows with stable dominant frequency
        required to declare equilibrium.
    frequency_tolerance : float
        Maximum allowed relative change in dominant frequency between
        consecutive windows to be considered "stable".
    
    Returns
    -------
    int
        Estimated equilibration step.
    """
    # Scale window with grid size
    window = max(base_window, int(base_window * (grid_size / 100)))
    
    # Need at least 3 windows worth of data
    if len(time_series) < window * 4:
        return len(time_series) * sample_interval
    
    # Compute dominant frequency for each sliding window
    step_size = window // 4  # Overlap windows by 75%
    dominant_freqs = []
    window_centers = []
    
    for start in range(0, len(time_series) - window, step_size):
        segment = time_series[start:start + window]
        
        # Remove mean (DC component)
        segment = segment - np.mean(segment)
        
        # Compute FFT
        fft_result = np.fft.rfft(segment)
        power = np.abs(fft_result) ** 2
        freqs = np.fft.rfftfreq(window, d=sample_interval)
        
        # Skip DC (index 0) and find dominant frequency
        if len(power) > 1:
            # Find peak in power spectrum (excluding DC)
            peak_idx = np.argmax(power[1:]) + 1
            dominant_freq = freqs[peak_idx]
            dominant_freqs.append(dominant_freq)
            window_centers.append(start + window // 2)
    
    if len(dominant_freqs) < n_stable_windows + 2:
        return len(time_series) * sample_interval
    
    dominant_freqs = np.array(dominant_freqs)
    window_centers = np.array(window_centers)
    
    # Find where dominant frequency stabilizes
    # Skip first few windows (definitely transient)
    start_check = max(2, len(dominant_freqs) // 5)
    
    stable_count = 0
    
    for i in range(start_check, len(dominant_freqs) - 1):
        freq_prev = dominant_freqs[i - 1]
        freq_curr = dominant_freqs[i]
        
        # Check if frequency is stable (relative change small)
        if freq_prev > 0:
            rel_change = abs(freq_curr - freq_prev) / freq_prev
        else:
            rel_change = 1.0 if freq_curr != 0 else 0.0
        
        if rel_change < frequency_tolerance:
            stable_count += 1
            if stable_count >= n_stable_windows:
                # Found stable frequency regime
                eq_sample = window_centers[i - n_stable_windows + 1]
                return eq_sample * sample_interval
        else:
            stable_count = 0
    
    return len(time_series) * sample_interval


def get_dominant_frequency_series(
    time_series: np.ndarray,
    sample_interval: int,
    window: int,
) -> tuple:
    """
    Compute dominant frequency over sliding windows (for diagnostic plotting).
    
    Returns (window_centers, dominant_frequencies, power_concentration).
    """
    step_size = window // 4
    dominant_freqs = []
    power_concentrations = []
    window_centers = []
    
    for start in range(0, len(time_series) - window, step_size):
        segment = time_series[start:start + window]
        segment = segment - np.mean(segment)
        
        fft_result = np.fft.rfft(segment)
        power = np.abs(fft_result) ** 2
        freqs = np.fft.rfftfreq(window, d=sample_interval)
        
        if len(power) > 1:
            # Dominant frequency (excluding DC)
            peak_idx = np.argmax(power[1:]) + 1
            dominant_freq = freqs[peak_idx]
            dominant_freqs.append(dominant_freq)
            
            # Power concentration: fraction of total power in dominant frequency
            total_power = np.sum(power[1:])  # Exclude DC
            if total_power > 0:
                concentration = power[peak_idx] / total_power
            else:
                concentration = 0
            power_concentrations.append(concentration)
            
            window_centers.append((start + window // 2) * sample_interval)
    
    return (np.array(window_centers), 
            np.array(dominant_freqs), 
            np.array(power_concentrations))


# =============================================================================
# MAIN STUDY FUNCTION
# =============================================================================

def run_warmup_study(cfg: WarmupStudyConfig, logger: logging.Logger) -> Dict[int, Dict]:
    """
    Run warmup cost study across multiple grid sizes.
    
    Returns dict mapping grid_size -> results dict.
    """
    from models.CA import PP
    
    # Try to import Numba optimization
    try:
        from models.numba_optimized import warmup_numba_kernels, set_numba_seed, NUMBA_AVAILABLE
        USE_NUMBA = NUMBA_AVAILABLE
    except ImportError:
        USE_NUMBA = False
        def warmup_numba_kernels(size, **kwargs): pass
        def set_numba_seed(seed): pass
    
    logger.info(f"Numba acceleration: {'ENABLED' if USE_NUMBA else 'DISABLED'}")
    
    results = {}
    
    for L in cfg.grid_sizes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing grid size L = {L}")
        logger.info(f"{'='*50}")
        
        # Show scaled FFT window size
        scaled_window = max(cfg.equilibration_window, int(cfg.equilibration_window * (L / 100)))
        logger.info(f"  FFT window size (scaled): {scaled_window} samples")
        
        # Warmup Numba kernels for this size
        warmup_numba_kernels(L, directed_hunting=cfg.directed_hunting)
        
        size_results = {
            'time_per_step': [],
            'equilibration_steps': [],
            'final_prey_density': [],
            'final_pred_density': [],
        }
        
        for rep in range(cfg.n_replicates):
            seed = rep * 1000 + L
            np.random.seed(seed)
            if USE_NUMBA:
                set_numba_seed(seed)
            
            # Initialize model
            model = PP(
                rows=L, cols=L,
                densities=cfg.densities,
                neighborhood="moore",
                params={
                    "prey_birth": cfg.prey_birth,
                    "prey_death": cfg.prey_death,
                    "predator_death": cfg.predator_death,
                    "predator_birth": cfg.predator_birth,
                },
                seed=seed,
                synchronous=cfg.synchronous,
                directed_hunting=cfg.directed_hunting,
            )
            
            # Track population over time
            prey_densities = []
            pred_densities = []
            grid_cells = L * L
            
            t0 = time.perf_counter()
            
            for step in range(cfg.max_steps):
                if step % cfg.sample_interval == 0:
                    prey_count = np.sum(model.grid == 1)
                    pred_count = np.sum(model.grid == 2)
                    prey_densities.append(prey_count / grid_cells)
                    pred_densities.append(pred_count / grid_cells)
                model.update()
            
            total_time = time.perf_counter() - t0
            time_per_step = total_time / cfg.max_steps
            
            prey_densities = np.array(prey_densities)
            pred_densities = np.array(pred_densities)
            
            # Estimate equilibration (trend-based, robust to grid size)
            eq_steps = estimate_equilibration_frequency(
                prey_densities,
                cfg.sample_interval,
                grid_size=L,
                base_window=cfg.equilibration_window,
            )
            
            size_results['time_per_step'].append(time_per_step)
            size_results['equilibration_steps'].append(eq_steps)
            size_results['final_prey_density'].append(prey_densities[-1])
            size_results['final_pred_density'].append(pred_densities[-1])
            
            if (rep + 1) % max(1, cfg.n_replicates // 5) == 0:
                logger.info(f"  Replicate {rep+1}/{cfg.n_replicates}: "
                           f"eq_steps={eq_steps}, time/step={time_per_step*1000:.2f}ms")
        
        # Aggregate results
        results[L] = {
            'grid_size': L,
            'grid_cells': L * L,
            'mean_time_per_step': float(np.mean(size_results['time_per_step'])),
            'std_time_per_step': float(np.std(size_results['time_per_step'])),
            'mean_eq_steps': float(np.mean(size_results['equilibration_steps'])),
            'std_eq_steps': float(np.std(size_results['equilibration_steps'])),
            'mean_total_warmup_time': float(
                np.mean(size_results['equilibration_steps']) * 
                np.mean(size_results['time_per_step'])
            ),
            'mean_final_prey_density': float(np.mean(size_results['final_prey_density'])),
            'mean_final_pred_density': float(np.mean(size_results['final_pred_density'])),
            'raw_data': {k: [float(x) for x in v] for k, v in size_results.items()},
        }
        
        logger.info(f"\n  Summary for L={L}:")
        logger.info(f"    Time per step: {results[L]['mean_time_per_step']*1000:.2f} ± "
                   f"{results[L]['std_time_per_step']*1000:.2f} ms")
        logger.info(f"    Equilibration steps: {results[L]['mean_eq_steps']:.0f} ± "
                   f"{results[L]['std_eq_steps']:.0f}")
        logger.info(f"    Total warmup time: {results[L]['mean_total_warmup_time']:.2f} s")
    
    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_warmup_scaling(
    results: Dict[int, Dict],
    output_dir: Path,
    dpi: int = 150,
) -> Path:
    """Generate warmup scaling analysis plots."""
    
    sizes = sorted(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Time per step vs L²
    ax = axes[0]
    times_ms = [results[L]['mean_time_per_step'] * 1000 for L in sizes]
    times_std = [results[L]['std_time_per_step'] * 1000 for L in sizes]
    cells = [L**2 for L in sizes]
    
    ax.errorbar(cells, times_ms, yerr=times_std, fmt='o-', capsize=5, 
                linewidth=2, color='steelblue', markersize=8)
    
    # Fit linear scaling with L²
    slope, intercept, r_val, _, _ = linregress(cells, times_ms)
    fit_line = intercept + slope * np.array(cells)
    ax.plot(cells, fit_line, 'r--', alpha=0.7,
            label=f'Fit: T = {slope:.4f}·L² + {intercept:.2f}\n(R² = {r_val**2:.3f})')
    
    ax.set_xlabel("Grid cells (L²)")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title("Computational Cost per Step")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Equilibration steps vs L (log-log)
    ax = axes[1]
    
    eq_steps = [results[L]['mean_eq_steps'] for L in sizes]
    eq_stds = [results[L]['std_eq_steps'] for L in sizes]
    ax.errorbar(sizes, eq_steps, yerr=eq_stds, fmt='o-', capsize=5, 
                linewidth=2, color='forestgreen', markersize=8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Fit power law: steps ~ L^z
    valid_mask = np.array(eq_steps) > 0
    if np.sum(valid_mask) >= 2:
        log_L = np.log(np.array(sizes)[valid_mask])
        log_steps = np.log(np.array(eq_steps)[valid_mask])
        z, log_a, r_val, _, _ = linregress(log_L, log_steps)
        
        fit_sizes = np.linspace(min(sizes), max(sizes), 100)
        fit_steps = np.exp(log_a) * fit_sizes**z
        ax.plot(fit_sizes, fit_steps, 'r--', alpha=0.7,
                label=f'Fit: t_eq ∼ L^{z:.2f} (R² = {r_val**2:.3f})')
    
    ax.set_xlabel("Grid size L")
    ax.set_ylabel("Equilibration steps")
    ax.set_title("Equilibration Time Scaling")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    # Panel 3: Total equilibration time vs L
    ax = axes[2]
    total_times = [results[L]['mean_total_warmup_time'] for L in sizes]
    
    ax.plot(sizes, total_times, 'o-', linewidth=2, color='crimson', markersize=8)
    
    # Fit power law for total time
    if len(sizes) >= 2:
        log_L = np.log(sizes)
        log_T = np.log(total_times)
        exponent, log_c, r_val, _, _ = linregress(log_L, log_T)
        
        fit_sizes = np.linspace(min(sizes), max(sizes), 100)
        fit_T = np.exp(log_c) * fit_sizes**exponent
        ax.plot(fit_sizes, fit_T, 'k--', alpha=0.7,
                label=f'Fit: T_warmup ∼ L^{exponent:.2f}\n(R² = {r_val**2:.3f})')
    
    ax.set_xlabel("Grid size L")
    ax.set_ylabel("Total warmup time (s)")
    ax.set_title("Total Warmup Cost")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "warmup_scaling.png"
    plt.savefig(output_file, dpi=dpi)
    plt.close()
    
    return output_file


def plot_scaling_summary(
    results: Dict[int, Dict],
    output_dir: Path,
    dpi: int = 150,
) -> Path:
    """Generate summary plot with scaling exponents."""
    
    sizes = sorted(results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot time per step normalized by L²
    times_normalized = [results[L]['mean_time_per_step'] / (L**2) * 1e6 for L in sizes]
    ax.plot(sizes, times_normalized, 'o-', linewidth=2, markersize=8,
            label='Time/step / L² (μs/cell)')
    
    # Plot equilibration steps normalized by theoretical scaling
    # Try different z values
    for z, color, style in [(1.0, 'green', '--'), (1.5, 'orange', '-.'), (2.0, 'red', ':')]:
        eq_normalized = [results[L]['mean_eq_steps'] / (L**z) for L in sizes]
        # Normalize to first point for comparison
        if eq_normalized[0] > 0:
            eq_normalized = [x / eq_normalized[0] for x in eq_normalized]
        ax.plot(sizes, eq_normalized, style, color=color, linewidth=2, alpha=0.7,
                label=f'Eq. steps / L^{z:.1f} (normalized)')
    
    ax.set_xlabel("Grid size L")
    ax.set_ylabel("Normalized value")
    ax.set_title("Scaling Analysis: Identifying Exponents")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    output_file = output_dir / "warmup_scaling_summary.png"
    plt.savefig(output_file, dpi=dpi)
    plt.close()
    
    return output_file


# =============================================================================
# DIAGNOSTIC VISUALIZATION
# =============================================================================

def run_diagnostic(
    grid_sizes: List[int],
    cfg: WarmupStudyConfig,
    output_dir: Path,
    logger: logging.Logger,
    dpi: int = 150,
):
    """
    Run diagnostic simulations to visualize population dynamics and equilibration detection.
    
    Creates detailed plots showing:
    - Population time series for each grid size
    - Rolling means used for trend detection
    - Direction of changes (+ or -)
    - Detected equilibration point
    """
    from models.CA import PP
    
    try:
        from models.numba_optimized import warmup_numba_kernels, set_numba_seed, NUMBA_AVAILABLE
        USE_NUMBA = NUMBA_AVAILABLE
    except ImportError:
        USE_NUMBA = False
        def warmup_numba_kernels(size, **kwargs): pass
        def set_numba_seed(seed): pass
    
    n_sizes = len(grid_sizes)
    fig, axes = plt.subplots(n_sizes, 3, figsize=(15, 4 * n_sizes))
    if n_sizes == 1:
        axes = axes.reshape(1, -1)
    
    for row, L in enumerate(grid_sizes):
        logger.info(f"Diagnostic run for L={L}...")
        
        # Warmup Numba
        warmup_numba_kernels(L, directed_hunting=cfg.directed_hunting)
        
        seed = 42 + L
        np.random.seed(seed)
        if USE_NUMBA:
            set_numba_seed(seed)
        
        # Run simulation
        model = PP(
            rows=L, cols=L,
            densities=cfg.densities,
            neighborhood="moore",
            params={
                "prey_birth": cfg.prey_birth,
                "prey_death": cfg.prey_death,
                "predator_death": cfg.predator_death,
                "predator_birth": cfg.predator_birth,
            },
            seed=seed,
            synchronous=cfg.synchronous,
            directed_hunting=cfg.directed_hunting,
        )
        
        # Collect data
        prey_densities = []
        pred_densities = []
        grid_cells = L * L
        
        for step in range(cfg.max_steps):
            if step % cfg.sample_interval == 0:
                prey_densities.append(np.sum(model.grid == 1) / grid_cells)
                pred_densities.append(np.sum(model.grid == 2) / grid_cells)
            model.update()
        
        prey_densities = np.array(prey_densities)
        pred_densities = np.array(pred_densities)
        steps = np.arange(len(prey_densities)) * cfg.sample_interval
        
        # Compute frequency analysis
        base_window = cfg.equilibration_window
        window = max(base_window, int(base_window * (L / 100)))
        
        # Get frequency series for plotting
        freq_centers, dominant_freqs, power_conc = get_dominant_frequency_series(
            prey_densities, cfg.sample_interval, window
        )
        
        # Detect equilibration
        eq_steps = estimate_equilibration_frequency(
            prey_densities, cfg.sample_interval, grid_size=L, base_window=base_window
        )
        
        # Panel 1: Population time series
        ax = axes[row, 0]
        ax.plot(steps, prey_densities, 'g-', alpha=0.7, linewidth=1, label='Prey')
        ax.plot(steps, pred_densities, 'r-', alpha=0.7, linewidth=1, label='Predator')
        ax.axvline(eq_steps, color='blue', linestyle='--', linewidth=2, label=f'Equilibrium @ {eq_steps}')
        ax.set_xlabel("Simulation steps")
        ax.set_ylabel("Density")
        ax.set_title(f"L={L}: Population Dynamics (window={window})")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Dominant frequency over time
        ax = axes[row, 1]
        if len(dominant_freqs) > 0:
            ax.plot(freq_centers, dominant_freqs * 1000, 'b-', linewidth=1.5, marker='o', markersize=3)
        ax.axvline(eq_steps, color='blue', linestyle='--', linewidth=2)
        ax.set_xlabel("Simulation steps")
        ax.set_ylabel("Dominant frequency (mHz)")
        ax.set_title(f"L={L}: Dominant Oscillation Frequency")
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Power concentration (how dominant is the main frequency)
        ax = axes[row, 2]
        if len(power_conc) > 0:
            ax.plot(freq_centers, power_conc, 'purple', linewidth=1.5, marker='o', markersize=3)
            ax.fill_between(freq_centers, 0, power_conc, alpha=0.3, color='purple')
        ax.axvline(eq_steps, color='blue', linestyle='--', linewidth=2, label=f'Detected @ {eq_steps}')
        ax.set_xlabel("Simulation steps")
        ax.set_ylabel("Power concentration")
        ax.set_title(f"L={L}: Frequency Dominance")
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "warmup_diagnostic.png"
    plt.savefig(output_file, dpi=dpi)
    plt.close()
    
    logger.info(f"Saved diagnostic plot to {output_file}")
    return output_file


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Study warmup period cost vs. grid size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --diagnostic                       # Visualize dynamics first!
  %(prog)s                                    # Default settings
  %(prog)s --sizes 50 100 150 200 300         # Custom grid sizes
  %(prog)s --replicates 20                    # More replicates for statistics
  %(prog)s --max-steps 3000                   # Longer runs for large grids
  %(prog)s --output results/warmup_analysis/  # Custom output directory
        """
    )
    
    parser.add_argument('--sizes', type=int, nargs='+', default=[50, 75, 100, 150, 200],
                        help='Grid sizes to test (default: 50 75 100 150 200)')
    parser.add_argument('--replicates', type=int, default=10,
                        help='Number of replicates per grid size (default: 10)')
    parser.add_argument('--max-steps', type=int, default=2000,
                        help='Maximum simulation steps (default: 2000)')
    parser.add_argument('--sample-interval', type=int, default=10,
                        help='Steps between population samples (default: 10)')
    parser.add_argument('--output', type=Path, default=Path('results/warmup_study'),
                        help='Output directory (default: results/warmup_study)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Plot resolution (default: 150)')
    parser.add_argument('--prey-birth', type=float, default=0.22,
                        help='Prey birth rate (default: 0.22)')
    parser.add_argument('--prey-death', type=float, default=0.04,
                        help='Prey death rate (default: 0.04)')
    parser.add_argument('--diagnostic', action='store_true',
                        help='Run diagnostic mode: visualize dynamics and equilibration detection')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "warmup_study.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    
    # Create configuration
    cfg = WarmupStudyConfig(
        grid_sizes=tuple(args.sizes),
        n_replicates=args.replicates,
        max_steps=args.max_steps,
        sample_interval=args.sample_interval,
        prey_birth=args.prey_birth,
        prey_death=args.prey_death,
    )
    
    # Header
    logger.info("=" * 60)
    logger.info("WARMUP PERIOD COST STUDY")
    logger.info("=" * 60)
    logger.info(f"Grid sizes: {cfg.grid_sizes}")
    logger.info(f"Replicates: {cfg.n_replicates}")
    logger.info(f"Max steps: {cfg.max_steps}")
    logger.info(f"Parameters: prey_birth={cfg.prey_birth}, prey_death={cfg.prey_death}")
    logger.info(f"Output: {output_dir}")
    
    # Save configuration
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(asdict(cfg), f, indent=2)
    logger.info(f"Saved config to {config_file}")
    
    # Diagnostic mode: visualize dynamics without full study
    if args.diagnostic:
        logger.info("\n" + "=" * 60)
        logger.info("DIAGNOSTIC MODE")
        logger.info("=" * 60)
        logger.info("Running single simulations to visualize dynamics...")
        run_diagnostic(list(cfg.grid_sizes), cfg, output_dir, logger, args.dpi)
        logger.info("\nDiagnostic complete! Check warmup_diagnostic.png")
        logger.info("Adjust parameters based on the plots, then run without --diagnostic")
        return
    
    # Run study
    results = run_warmup_study(cfg, logger)
    
    # Save results
    results_file = output_dir / "warmup_results.json"
    # Convert keys to strings for JSON
    json_results = {str(k): v for k, v in results.items()}
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Saved results to {results_file}")
    
    # Generate plots
    logger.info("\nGenerating plots...")
    plot1 = plot_warmup_scaling(results, output_dir, args.dpi)
    logger.info(f"Saved {plot1}")
    
    plot2 = plot_scaling_summary(results, output_dir, args.dpi)
    logger.info(f"Saved {plot2}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    sizes = sorted(results.keys())
    
    # Compute scaling exponents
    if len(sizes) >= 2:
        eq_steps = [results[L]['mean_eq_steps'] for L in sizes]
        total_times = [results[L]['mean_total_warmup_time'] for L in sizes]
        
        # Filter out any zero or negative values for log
        valid_eq = [(L, eq) for L, eq in zip(sizes, eq_steps) if eq > 0]
        valid_T = [(L, T) for L, T in zip(sizes, total_times) if T > 0]
        
        if len(valid_eq) >= 2:
            log_L_eq = np.log([x[0] for x in valid_eq])
            log_eq = np.log([x[1] for x in valid_eq])
            z_eq, _, r_eq, _, _ = linregress(log_L_eq, log_eq)
        else:
            z_eq, r_eq = 0, 0
        
        if len(valid_T) >= 2:
            log_L_T = np.log([x[0] for x in valid_T])
            log_T = np.log([x[1] for x in valid_T])
            z_total, _, r_total, _, _ = linregress(log_L_T, log_T)
        else:
            z_total, r_total = 0, 0
        
        logger.info(f"Equilibration steps scaling: t_eq ~ L^{z_eq:.2f} (R² = {r_eq**2:.3f})")
        logger.info(f"Total warmup time scaling: T_warmup ~ L^{z_total:.2f} (R² = {r_total**2:.3f})")
        logger.info(f"\nInterpretation:")
        logger.info(f"  - Computational cost per step scales as L² (as expected)")
        logger.info(f"  - Equilibration steps scale as L^{z_eq:.2f}")
        logger.info(f"  - Combined effect: total warmup ~ L^{z_total:.2f}")
        
        if z_eq > 1.5:
            logger.info(f"\n  Warning: Dynamic exponent z={z_eq:.2f} > 1.5 suggests")
            logger.info(f"  critical slowing down. Consider longer warmup for large grids.")
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()

