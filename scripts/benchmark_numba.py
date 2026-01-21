import time
import statistics
import numpy as np


from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.CA import PP

def python_async_logic(pp):
    """The original Pure Python asynchronous logic for benchmarking."""
    grid = pp.grid
    params = pp.params
    rows, cols = grid.shape
    grid_ref = grid.copy()

    # 1. Neighbor shifts
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dr_arr = [s[0] for s in shifts]
    dc_arr = [s[1] for s in shifts]

    # 2. Get occupied cells and shuffle (The slow part)
    occupied = np.argwhere(grid_ref != 0)
    if occupied.size > 0:
        order = np.random.permutation(len(occupied))
        for idx in order:
            r, c = occupied[idx]
            state = grid_ref[r, c]
            
            # Pick random neighbor
            nbi = np.random.randint(0, 8)
            nr, nc = (r + dr_arr[nbi]) % rows, (c + dc_arr[nbi]) % cols
            
            if state == 1: # Prey
                if np.random.random() < params["prey_death"]:
                    grid[r, c] = 0
                elif grid_ref[nr, nc] == 0 and np.random.random() < params["prey_birth"]:
                    grid[nr, nc] = 1
            elif state == 2: # Predator
                if np.random.random() < params["predator_death"]:
                    grid[r, c] = 0
                elif grid_ref[nr, nc] == 1 and np.random.random() < params["predator_birth"]:
                    grid[nr, nc] = 2

def benchmark_numba_impact(rows=150, cols=150, repeats=50):
    pp = PP(rows=rows, cols=cols, densities=(0.2, 0.1), seed=42, synchronous=False)
    initial_grid = pp.grid.copy()
    
    # --- PURE PYTHON ---
    print(f"[*] Benchmarking Pure Python Async ({rows}x{cols})...")
    python_times = []
    for _ in range(repeats):
        pp.grid[:] = initial_grid
        t0 = time.perf_counter()
        python_async_logic(pp)
        python_times.append(time.perf_counter() - t0)
        
    # --- NUMBA ---
    print(f"[*] Benchmarking Numba-Accelerated Async...")
    # Warm up compilation
    pp.update_async() 
    
    numba_times = []
    for _ in range(repeats):
        pp.grid[:] = initial_grid
        t0 = time.perf_counter()
        pp.update_async() 
        numba_times.append(time.perf_counter() - t0)

    # --- SUMMARY ---
    py_mean = statistics.mean(python_times) * 1000
    nb_mean = statistics.mean(numba_times) * 1000
    speedup = py_mean / nb_mean

    print("\n" + "="*50)
    print(f"ASYNC PERFORMANCE RESULTS ({rows}x{cols})")
    print("="*50)
    print(f"Pure Python: {py_mean:.2f} ms / step")
    print(f"Numba JIT:   {nb_mean:.2f} ms / step")
    print(f"Speedup:     {speedup:.1f}x")
    print("="*50)

if __name__ == "__main__":
    benchmark_numba_impact()