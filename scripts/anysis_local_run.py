#!/usr/bin/env python3
"""
Local test run for pp_analysis.py

Run this to verify everything works before submitting to HPC.
Uses minimal settings that should complete in ~2-5 minutes on a laptop.

Usage:
    python test_local_run.py
    python test_local_run.py --mode sweep      # Only test sweep
    python test_local_run.py --mode plot       # Only test plotting (requires prior sweep)
"""

import sys
import time
from pathlib import Path

# Import and modify config before running
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from pp_analysis import Config, main, run_2d_sweep, run_sensitivity, run_fss, generate_plots
import logging
import argparse


def create_minimal_config():
    """Create a minimal config for fast local testing."""
    cfg = Config()
    
    # Tiny grid
    cfg.default_grid = 30  # Small grid (was 100)
    
    # Minimal parameter sweep (3x3 = 9 parameter combinations)
    cfg.n_prey_birth = 3
    cfg.n_prey_death = 3
    cfg.prey_birth_min = 0.15
    cfg.prey_birth_max = 0.25
    cfg.prey_death_min = 0.02
    cfg.prey_death_max = 0.08
    
    # Very few replicates
    cfg.n_replicates = 2  # Was 50
    
    # Short simulation
    cfg.warmup_steps = 50   # Was 200
    cfg.measurement_steps = 100  # Was 300
    cfg.cluster_samples = 10  # Was 40
    cfg.cluster_interval = 10  # Was 8
    
    # Minimal FSS
    cfg.fss_grid_sizes = (30, 50)  # Was (50, 75, 100, 150)
    cfg.fss_replicates = 3  # Was 100
    
    # Minimal sensitivity
    cfg.sensitivity_sd_values = (0.05, 0.10, 0.15)  # Was 5 values
    cfg.sensitivity_replicates = 3  # Was 20
    
    # Use all available cores but cap at 4 for laptop
    cfg.n_jobs = 4
    
    # Disable optional expensive features
    cfg.collect_neighbor_stats = False
    cfg.collect_timeseries = False
    cfg.save_snapshots = False
    
    return cfg


def run_local_test(mode="full"):
    """Run a minimal local test."""
    
    # Setup
    cfg = create_minimal_config()
    output_dir = Path("results_local_test")
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "test_run.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    
    # Print config summary
    logger.info("=" * 60)
    logger.info("LOCAL TEST RUN - Minimal Configuration")
    logger.info("=" * 60)
    logger.info(f"Grid size: {cfg.default_grid}x{cfg.default_grid}")
    logger.info(f"Parameter grid: {cfg.n_prey_birth}x{cfg.n_prey_death}")
    logger.info(f"Replicates: {cfg.n_replicates}")
    logger.info(f"Warmup/Measurement: {cfg.warmup_steps}/{cfg.measurement_steps}")
    logger.info(f"Cores: {cfg.n_jobs}")
    
    # Estimate
    n_sweep = cfg.n_prey_birth * cfg.n_prey_death * cfg.n_replicates * 2
    n_sens = len(cfg.sensitivity_sd_values) * cfg.sensitivity_replicates
    n_fss = len(cfg.fss_grid_sizes) * cfg.fss_replicates
    logger.info(f"Total simulations: sweep={n_sweep}, sens={n_sens}, fss={n_fss}")
    logger.info(f"Estimated time: 2-5 minutes")
    logger.info("=" * 60)
    
    start = time.time()
    
    try:
        if mode in ["full", "sweep"]:
            logger.info("Running 2D sweep...")
            t0 = time.time()
            run_2d_sweep(cfg, output_dir, logger)
            logger.info(f"Sweep completed in {time.time()-t0:.1f}s")
        
        if mode in ["full", "sensitivity"]:
            logger.info("Running sensitivity analysis...")
            t0 = time.time()
            run_sensitivity(cfg, output_dir, logger)
            logger.info(f"Sensitivity completed in {time.time()-t0:.1f}s")
        
        if mode in ["full", "fss"]:
            logger.info("Running FSS analysis...")
            t0 = time.time()
            run_fss(cfg, output_dir, logger)
            logger.info(f"FSS completed in {time.time()-t0:.1f}s")
        
        if mode in ["full", "plot"]:
            logger.info("Generating plots...")
            t0 = time.time()
            generate_plots(cfg, output_dir, logger)
            logger.info(f"Plots completed in {time.time()-t0:.1f}s")
        
        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"LOCAL TEST COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)
        
        # List output files
        logger.info("Generated files:")
        for f in sorted(output_dir.glob("*")):
            size = f.stat().st_size / 1024
            logger.info(f"  {f.name}: {size:.1f} KB")
        
        return True
        
    except Exception as e:
        logger.exception(f"Test failed with error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local test run for pp_analysis")
    parser.add_argument(
        "--mode",
        choices=["full", "sweep", "sensitivity", "fss", "plot"],
        default="full",
        help="Which analysis to run (default: full)"
    )
    args = parser.parse_args()
    
    success = run_local_test(args.mode)
    sys.exit(0 if success else 1)