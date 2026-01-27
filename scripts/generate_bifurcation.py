#!/usr/bin/env python3
"""
Generate bifurcation diagram data for predator-prey model.

Sweeps prey death rate (control parameter) and records equilibrium
prey population across multiple replicates.

Output is saved in format compatible with analysis.py's plot_bifurcation_diagram.

Usage:
    python scripts/generate_bifurcation.py                    # Default settings
    python scripts/generate_bifurcation.py --n-sweep 30       # More sweep points
    python scripts/generate_bifurcation.py --n-replicates 25  # More replicates
    python scripts/generate_bifurcation.py --grid-size 50     # Smaller grid (faster)
    python scripts/generate_bifurcation.py --output results/bifurcation_test
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = str(Path(__file__).parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.CA import PP
from models.numba_optimized import warmup_numba_kernels, set_numba_seed, NUMBA_AVAILABLE


def run_single_bifurcation_sim(
    prey_death: float,
    prey_birth: float,
    predator_birth: float,
    predator_death: float,
    grid_size: int,
    warmup_steps: int,
    measurement_steps: int,
    seed: int,
    densities: Tuple[float, float] = (0.30, 0.15),
    directed_hunting: bool = False,
) -> Tuple[float, float]:
    """
    Run a single simulation and return equilibrium populations.
    
    Parameters
    ----------
    prey_death : float
        Prey death rate (control parameter).
    prey_birth : float
        Prey birth rate (fixed).
    predator_birth : float
        Predator birth rate.
    predator_death : float
        Predator death rate.
    grid_size : int
        Size of the grid (grid_size x grid_size).
    warmup_steps : int
        Steps to run before measuring (equilibration).
    measurement_steps : int
        Steps to run while measuring.
    seed : int
        Random seed for reproducibility.
    densities : tuple
        Initial (prey, predator) densities.
    directed_hunting : bool
        Whether predators hunt directionally.
    
    Returns
    -------
    Tuple[float, float]
        Mean (prey, predator) populations during measurement phase.
    """
    # Set seed
    np.random.seed(seed)
    if NUMBA_AVAILABLE:
        set_numba_seed(seed)
    
    # Create model
    model = PP(
        rows=grid_size,
        cols=grid_size,
        densities=densities,
        params={
            "prey_death": prey_death,
            "prey_birth": prey_birth,
            "predator_birth": predator_birth,
            "predator_death": predator_death,
        },
        seed=seed,
        synchronous=False,
        directed_hunting=directed_hunting,
    )
    
    # Warmup phase (equilibration)
    model.run(warmup_steps)
    
    # Measurement phase - collect prey and predator counts
    prey_counts = []
    predator_counts = []
    for _ in range(measurement_steps):
        model.update()
        prey_counts.append(np.sum(model.grid == 1))
        predator_counts.append(np.sum(model.grid == 2))
    
    # Return mean populations during measurement
    return float(np.mean(prey_counts)), float(np.mean(predator_counts))


def generate_bifurcation_data(
    prey_death_min: float = 0.0,
    prey_death_max: float = 0.2,
    n_sweep: int = 25,
    n_replicates: int = 30,
    prey_birth: float = 0.20,
    predator_birth: float = 0.8,
    predator_death: float = 0.05,
    grid_size: int = 100,
    warmup_steps: int = 1000,
    measurement_steps: int = 200,
    densities: Tuple[float, float] = (0.30, 0.15),
    directed_hunting: bool = False,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate bifurcation diagram data.
    
    Parameters
    ----------
    prey_death_min : float
        Minimum prey death rate.
    prey_death_max : float
        Maximum prey death rate.
    n_sweep : int
        Number of sweep points.
    n_replicates : int
        Number of replicates per sweep point.
    prey_birth : float
        Fixed prey birth rate.
    predator_birth : float
        Fixed predator birth rate.
    predator_death : float
        Fixed predator death rate.
    grid_size : int
        Grid size (grid_size x grid_size).
    warmup_steps : int
        Equilibration steps.
    measurement_steps : int
        Measurement steps.
    densities : tuple
        Initial densities.
    directed_hunting : bool
        Use directed hunting.
    n_jobs : int
        Number of parallel jobs (-1 for all cores).
    
    Returns
    -------
    sweep_params : np.ndarray
        1D array of prey death rates.
    prey_results : np.ndarray
        2D array of shape (n_sweep, n_replicates) with prey equilibrium populations.
    predator_results : np.ndarray
        2D array of shape (n_sweep, n_replicates) with predator equilibrium populations.
    """
    # Generate sweep parameters
    sweep_params = np.linspace(prey_death_min, prey_death_max, n_sweep)
    
    # Warm up Numba kernels
    if NUMBA_AVAILABLE:
        logging.info("Warming up Numba kernels...")
        warmup_numba_kernels(grid_size, directed_hunting=directed_hunting)
    
    # Build job list
    jobs = []
    for i, pd in enumerate(sweep_params):
        for rep in range(n_replicates):
            # Generate unique seed from parameters
            seed = int(abs(hash((pd, rep, grid_size))) % (2**31))
            jobs.append((i, rep, pd, seed))
    
    logging.info(f"Running {len(jobs)} simulations...")
    logging.info(f"  Sweep: {n_sweep} points from {prey_death_min:.3f} to {prey_death_max:.3f}")
    logging.info(f"  Replicates: {n_replicates} per point")
    logging.info(f"  Grid: {grid_size}x{grid_size}")
    logging.info(f"  Steps: {warmup_steps} warmup + {measurement_steps} measurement")
    
    # Results arrays for both species
    prey_results = np.zeros((n_sweep, n_replicates))
    predator_results = np.zeros((n_sweep, n_replicates))
    
    # Try parallel execution
    try:
        from joblib import Parallel, delayed
        
        def run_job(idx, rep, pd, seed):
            prey_pop, pred_pop = run_single_bifurcation_sim(
                prey_death=pd,
                prey_birth=prey_birth,
                predator_birth=predator_birth,
                predator_death=predator_death,
                grid_size=grid_size,
                warmup_steps=warmup_steps,
                measurement_steps=measurement_steps,
                seed=seed,
                densities=densities,
                directed_hunting=directed_hunting,
            )
            return idx, rep, prey_pop, pred_pop
        
        # Run in parallel with progress bar
        parallel_results = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(run_job)(i, rep, pd, seed) for i, rep, pd, seed in jobs
        )
        
        for idx, rep, prey_pop, pred_pop in tqdm(parallel_results, total=len(jobs), desc="Bifurcation"):
            prey_results[idx, rep] = prey_pop
            predator_results[idx, rep] = pred_pop
    
    except ImportError:
        logging.warning("joblib not available, running sequentially...")
        for i, rep, pd, seed in tqdm(jobs, desc="Bifurcation"):
            prey_pop, pred_pop = run_single_bifurcation_sim(
                prey_death=pd,
                prey_birth=prey_birth,
                predator_birth=predator_birth,
                predator_death=predator_death,
                grid_size=grid_size,
                warmup_steps=warmup_steps,
                measurement_steps=measurement_steps,
                seed=seed,
                densities=densities,
                directed_hunting=directed_hunting,
            )
            prey_results[i, rep] = prey_pop
            predator_results[i, rep] = pred_pop
    
    return sweep_params, prey_results, predator_results


def main():
    parser = argparse.ArgumentParser(
        description="Generate bifurcation diagram data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Default: 100x100, 25 sweep, 15 replicates
  %(prog)s --grid-size 50               # Faster with smaller grid
  %(prog)s --n-sweep 40 --n-replicates 20  # Higher resolution
  %(prog)s --prey-death-min 0.02 --prey-death-max 0.08  # Focus on critical region
        """
    )
    
    # Sweep parameters
    parser.add_argument('--prey-death-min', type=float, default=0.0,
                       help='Minimum prey death rate (default: 0.0)')
    parser.add_argument('--prey-death-max', type=float, default=0.2,
                       help='Maximum prey death rate (default: 0.2)')
    parser.add_argument('--n-sweep', type=int, default=25,
                       help='Number of sweep points (default: 25)')
    parser.add_argument('--n-replicates', type=int, default=30,
                       help='Replicates per sweep point (default: 15)')
    
    # Fixed parameters
    parser.add_argument('--prey-birth', type=float, default=0.20,
                       help='Fixed prey birth rate (default: 0.20)')
    parser.add_argument('--predator-birth', type=float, default=0.8,
                       help='Fixed predator birth rate (default: 0.8)')
    parser.add_argument('--predator-death', type=float, default=0.05,
                       help='Fixed predator death rate (default: 0.05)')
    
    # Grid and timing
    parser.add_argument('--grid-size', type=int, default=100,
                       help='Grid size (default: 100)')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Warmup/equilibration steps (default: 1000)')
    parser.add_argument('--measurement-steps', type=int, default=200,
                       help='Measurement steps (default: 200)')
    
    # Output
    parser.add_argument('--output', type=Path, default=Path('results/bifurcation'),
                       help='Output directory (default: results/bifurcation)')
    
    # Parallelization
    parser.add_argument('--cores', type=int, default=-1,
                       help='Number of cores (-1 for all)')
    
    args = parser.parse_args()
    
    # Setup output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.output / "bifurcation.log"),
            logging.StreamHandler(),
        ]
    )
    
    logging.info("=" * 60)
    logging.info("BIFURCATION DATA GENERATION")
    logging.info("=" * 60)
    
    start_time = time.time()
    
    # Generate data
    sweep_params, prey_results, predator_results = generate_bifurcation_data(
        prey_death_min=args.prey_death_min,
        prey_death_max=args.prey_death_max,
        n_sweep=args.n_sweep,
        n_replicates=args.n_replicates,
        prey_birth=args.prey_birth,
        predator_birth=args.predator_birth,
        predator_death=args.predator_death,
        grid_size=args.grid_size,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        n_jobs=args.cores,
    )
    
    elapsed = time.time() - start_time
    
    # Save results in NPZ format (for analysis.py)
    npz_file = args.output / "bifurcation_results.npz"
    np.savez(npz_file, 
             sweep_params=sweep_params, 
             prey_results=prey_results,
             predator_results=predator_results)
    logging.info(f"Saved: {npz_file}")
    
    # Also save as JSON for inspection
    json_file = args.output / "bifurcation_results.json"
    with open(json_file, 'w') as f:
        json.dump({
            'sweep_params': sweep_params.tolist(),
            'prey_results': prey_results.tolist(),
            'predator_results': predator_results.tolist(),
            'config': {
                'prey_death_min': args.prey_death_min,
                'prey_death_max': args.prey_death_max,
                'n_sweep': args.n_sweep,
                'n_replicates': args.n_replicates,
                'prey_birth': args.prey_birth,
                'predator_birth': args.predator_birth,
                'predator_death': args.predator_death,
                'grid_size': args.grid_size,
                'warmup_steps': args.warmup_steps,
                'measurement_steps': args.measurement_steps,
            }
        }, f, indent=2)
    logging.info(f"Saved: {json_file}")
    
    # Summary statistics
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logging.info(f"Simulations: {args.n_sweep * args.n_replicates}")
    logging.info(f"Time per sim: {elapsed / (args.n_sweep * args.n_replicates):.2f}s")
    logging.info(f"")
    logging.info(f"Results shape: {prey_results.shape}")
    logging.info(f"Prey population range: {prey_results.min():.0f} - {prey_results.max():.0f}")
    logging.info(f"Prey mean population: {prey_results.mean():.1f}")
    logging.info(f"Predator population range: {predator_results.min():.0f} - {predator_results.max():.0f}")
    logging.info(f"Predator mean population: {predator_results.mean():.1f}")
    logging.info(f"")
    logging.info(f"To plot: python scripts/analysis.py {args.output} --bifurcation-only")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

