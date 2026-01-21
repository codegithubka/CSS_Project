import time
import statistics
import logging
from pathlib import Path
import sys
import numpy as np

# Ensure we can find our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.pp_analysis as ppa
from scripts.pp_analysis import Config

def run_stress_benchmark():
    cfg = Config()
    # INCREASE WORKLOAD: 150x150 grid and 1000 total steps
    # This ensures the 'race' is long enough to overcome the 'warmup'
    cfg.default_grid = 150       
    cfg.n_prey_birth = 2        # 2x2 sweep is enough to see scaling
    cfg.n_prey_death = 2
    cfg.n_replicates = 3        
    cfg.warmup_steps = 400
    cfg.measurement_steps = 600
    cfg.n_jobs = 4              
    
    output_dir = Path("results_stress")
    output_dir.mkdir(exist_ok=True)
    
    logging.getLogger('scripts.pp_analysis').setLevel(logging.WARNING)
    
    print("="*60)
    print("STRESS BENCHMARK: SCALING TO PRODUCTION SIZES")
    print("="*60)

    # --- MODE A: NUMBA ENABLED ---
    print("[*] Running WITH Numba (Compiling + High-Speed Execution)...")
    ppa.USE_NUMBA = True
    t0 = time.perf_counter()
    ppa.run_2d_sweep(cfg, output_dir, logging.getLogger("test"))
    numba_total = time.perf_counter() - t0
    print(f"    Total Time (Numba): {numba_total:.2f}s")

    # --- MODE B: NUMBA DISABLED ---
    print("[*] Running WITHOUT Numba (Pure Python)...")
    ppa.USE_NUMBA = False
    t0 = time.perf_counter()
    ppa.run_2d_sweep(cfg, output_dir, logging.getLogger("test"))
    python_total = time.perf_counter() - t0
    print(f"    Total Time (Python): {python_total:.2f}s")

    # --- FINAL REPORT ---
    speedup = python_total / numba_total
    print("\n" + "="*60)
    print("STRESS BENCHMARK SUMMARY")
    print("="*60)
    print(f"Real-World Workflow Speedup: {speedup:.2f}x")
    print(f"Estimated time for 1000 sims in Python: {(python_total/12)*1000/3600:.1f} hours")
    print(f"Estimated time for 1000 sims in Numba:  {(numba_total/12)*1000/3600:.1f} hours")
    print("="*60)

if __name__ == "__main__":
    run_stress_benchmark()