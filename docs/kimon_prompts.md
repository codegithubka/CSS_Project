### Mean Field class

1. Create a baseline mean-field class based on the attached research paper on predator-prey dynamics. The class should adhere to the papers specifications. The class should have a parameter sweep method for key predator and prey parameters that will be run in Snellius. Also include a method for equilibrium analysis. Make sure to justify the logic for this method. Include docstrings with a small method description and comments for code interpretability.

2. Justify initialization parameter values for a small test expiriment. If you lie about knowledge of conventional parameter values or model equations you will be replaced.

3. Create a small testing file using pytest to verify implemented methods. Make sure to cover edge cases and list them after the .py file output for me please. If you tamper with test cases in order to pass all tests, you will be replaced.

4. We are now ready to plot some of the results of the mean fielf baseline. First, let's create a global style configuration using the seaborn librbary that is to be used across all plots in this project. Make sure the legend is at the bottom of each plot.

5. Plot the phase portait to confirm the system spiral into a stable point. Show the nullclines as well. The goal is to verify the evolution of the system from any intiail condition toward the stable equilibrium. 

6. Create a time series analysis plot of the evolution of prey and predator density vs. time. Make sure enough time steps all visible to see how the system eventually stabilizes.

7. Create a bifuracation diagram to confirm the monotonic relationship for a varying prey death rate vs. equilibrium density. 

---

### Testing CA class

1. Create a comprehensive testing suite for the CA and PP classes. Test initialization, async update changes, synchronous update changes, prey growth in isolation behavior, predator starvation, parameter evolution and long run dynamics. Also make sur ethe test_viz mehtod works as desired

---

### Parameter Sweep and PP Class Analysis

2. Create a skeletal version of a .py script that will be subimtted into Snellius for parameter analysis. The purpose of this script should be to identify power law distribution with the consideration of finite size scaling, the hydra effect, and suitable parameter configurtaions for time series analysis for model evolution. Compare the non-evo to the evo model.

3. Create a config class adjustable depending on the CPU budget. We want to run a prey_birth vs. predator_death parameter sweep (2D), quantify the hydra effect using the derivative, search for the critical point (power law relartionship paramameter), quantify evolution sensitivity and analyze finite grid size scaling. Include script options for cost optimal runs as well. Make sure to have a summary of collected data stored for reference and future usage.


4. Add configuration option to run the asynchronous version of the CA class. The synchronous functionality should also be preserved. Provide me with a small smoke test to see if the updated file runs as expected.

5. Create a minimal bash script for Snellius. Use the rome configiuration.

6. Fix predator-prey analysis script so that the hydra effect focuses on the prey hydra effect as a result of the increasing prey death rate.


7. Add PCF analysis functonality for prey auto, predator auto and cross correlation. Also, integrate the snapshot method from the CA clas as an optional functionality of the analysis module. Add the folowing plots: 1. phase diagrams showing segregation, prey-clusterin, and predator clustering. Scatter plots tetsing if Hydra effect correlates with spatial segregation, and CA-style snapshots, neighbor histogram, and evolution trajectory.


8. Help me create a testing module for the analysis file. Use unittest.mock to create a mock model for testing. If you lie or falsify tests so that they pass my script, you will be replaced.


9. Add a larger scale simulation in the testing file to verify plots are as desired.

---

### Script Optimization

1. I am considering using numba for optimization and faster runs in the HPC. Outline an implementation plan, practical considerations, and feasibility within a logical timeframe.

2. Walk me through modifying the update_async method for simulation speedup

3. Provide me with a script using cProfile to profile the analysis script and target optimization goals specifically.

4. Write a small benchmark script for a large simulation to test how much the performance using numba has imrpoved (or no)

5. Identify the major bottlenecks in the script and suggest possible (realisic) fixes. if you lie about performance improvement expecatations, you will be replaced.

6. I am currently storing anslysis data in a .json file. Help me refactor the 2D sweep function to store binary data instead to minimize communication overhead.

7. The PCF is O(N^2). Show me how to use cell lists instead and spatial hashing. This is the biggest bottlneck so we must prioritize this. Walk me through reftorings in numba_optimized.py

8. We should also pre-allocate the buffers outside the kernel. Right now we are allocating memoery on every call.

9. Help me write a benchmark script to test the updated version against a non-numba optimized version. Create a summary txt with benchmark results.

10. Create unit tests for pp_analysis and numba_optimized files. Test all methods and make sure the scripts are HPC ready. Provide me with an outline of tests.


11. Explain to me how seeding works with numba and multiple processes. One of the tests fails which may lead to unreproducable results. There is a potential bug in the way runs are seeded.



```
OUTPUT:

(This was really interesting and might be useful in the future of our project as well)

================================================================================
NUMBA RNG SEEDING FIX - Implementation Guide
================================================================================

PROBLEM:
--------
Numba's JIT-compiled functions use their own internal RNG (Xoshiro128++), 
which is completely separate from NumPy's RNG. Calling np.random.seed() 
from Python only seeds NumPy's RNG, not Numba's.

SOLUTION:
---------
Create a @njit function that seeds the RNG from within the JIT context.

================================================================================
CHANGE 1: Add to numba_optimized.py (after line ~35, after the dummy decorators)
================================================================================

# --- ADD THIS BLOCK AFTER THE IMPORTS AND DUMMY DECORATORS ---

@njit(cache=True)
def set_numba_seed(seed: int) -> None:
    """
    Seed Numba's internal RNG from within a JIT context.
    
    IMPORTANT: This must be called to get reproducible results from 
    Numba-accelerated functions. Calling np.random.seed() from Python
    only affects NumPy's RNG, not Numba's internal Xoshiro128++ RNG.
    
    Args:
        seed: Integer seed value
        
    Usage:
        from numba_optimized import set_numba_seed
        set_numba_seed(42)  # Now Numba functions will be deterministic
    """
    np.random.seed(seed)

# --- END OF ADDITION ---


================================================================================
CHANGE 2: Update the exports/imports
================================================================================

In numba_optimized.py, update the module docstring to include set_numba_seed:

"""
...
Usage:
    from scripts.numba_optimized import (
        PPKernel,
        compute_all_pcfs_fast,
        measure_cluster_sizes_fast,
        set_numba_seed,  # <-- ADD THIS
        NUMBA_AVAILABLE
    )
    
    # Seed Numba's RNG for reproducibility
    set_numba_seed(42)
    
    # Create kernel once, reuse for all updates
    kernel = PPKernel(rows, cols)
    ...
"""


================================================================================
CHANGE 3: Update pp_analysis.py - Import set_numba_seed
================================================================================

Find the import block (around line 20-30) and add set_numba_seed:

# BEFORE:
from scripts.numba_optimized import (
    PPKernel,
    compute_all_pcfs_fast,
    measure_cluster_sizes_fast,
    warmup_numba_kernels,
    NUMBA_AVAILABLE,
)

# AFTER:
from scripts.numba_optimized import (
    PPKernel,
    compute_all_pcfs_fast,
    measure_cluster_sizes_fast,
    warmup_numba_kernels,
    set_numba_seed,  # <-- ADD THIS
    NUMBA_AVAILABLE,
)


================================================================================
CHANGE 4: Update run_single_simulation() in pp_analysis.py
================================================================================

Find the run_single_simulation function and add set_numba_seed call at the start:

def run_single_simulation(
    prey_birth: float,
    prey_death: float,
    grid_size: int,
    seed: int,
    with_evolution: bool,
    cfg: Config,
) -> Dict[str, Any]:
    """Run a single simulation and return results."""
    
    # --- ADD THESE LINES AT THE VERY START OF THE FUNCTION ---
    # Seed both NumPy and Numba RNGs for full reproducibility
    np.random.seed(seed)
    if NUMBA_AVAILABLE:
        set_numba_seed(seed)
    # --- END OF ADDITION ---
    
    # ... rest of the function remains unchanged ...


================================================================================
CHANGE 5: Update run_single_simulation_fss() in pp_analysis.py (if it exists)
================================================================================

Same pattern - add seeding at the start:

def run_single_simulation_fss(...):
    """Run FSS simulation."""
    
    # Seed both RNGs
    np.random.seed(seed)
    if NUMBA_AVAILABLE:
        set_numba_seed(seed)
    
    # ... rest unchanged ...


================================================================================
CHANGE 6: Update warmup_numba_kernels() in numba_optimized.py
================================================================================

Add a deterministic seed during warmup to avoid variability:

def warmup_numba_kernels(grid_size: int = 100):
    """
    Pre-compile all Numba kernels.
    """
    if not NUMBA_AVAILABLE:
        return
    
    # --- ADD THIS LINE ---
    set_numba_seed(0)  # Deterministic warmup
    # --- END OF ADDITION ---
    
    # Dummy data
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    # ... rest unchanged ...


================================================================================
COMPLETE UPDATED numba_optimized.py (key sections only)
================================================================================

Here's how the top of your file should look after changes:

```python
#!/usr/bin/env python3
"""
Numba-optimized kernels for predator-prey cellular automaton.

...

Usage:
    from scripts.numba_optimized import (
        PPKernel,
        compute_all_pcfs_fast,
        measure_cluster_sizes_fast,
        set_numba_seed,
        NUMBA_AVAILABLE
    )
    
    # Seed Numba's RNG for reproducibility
    set_numba_seed(42)
    
    # Create kernel once, reuse for all updates
    kernel = PPKernel(rows, cols)
    for step in range(n_steps):
        kernel.update(grid, prey_death_arr, params...)
"""

import numpy as np
from typing import Tuple, Dict, Optional

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args):
        return range(*args)


# ============================================================================
# RNG SEEDING
# ============================================================================

@njit(cache=True)
def set_numba_seed(seed: int) -> None:
    """
    Seed Numba's internal RNG from within a JIT context.
    
    IMPORTANT: This must be called to get reproducible results from 
    Numba-accelerated functions. Calling np.random.seed() from Python
    only affects NumPy's RNG, not Numba's internal Xoshiro128++ RNG.
    
    Args:
        seed: Integer seed value
    """
    np.random.seed(seed)


# ============================================================================
# ASYNC KERNEL
# ============================================================================

@njit(cache=True)
def _pp_async_kernel(...):
    # ... unchanged ...

================================================================================
TESTING THE FIX
================================================================================

After making these changes, this test should pass:

```python
def test_kernel_deterministic_with_seed():
    """Same seed should produce same results."""
    from numba_optimized import PPKernel, set_numba_seed
    
    results = []
    
    for _ in range(2):
        # Seed BOTH RNGs
        np.random.seed(12345)
        set_numba_seed(12345)  # <-- This is the key!
        
        grid = np.random.choice([0, 1, 2], (30, 30), p=[0.5, 0.3, 0.2]).astype(np.int32)
        prey_death = np.full((30, 30), 0.05, dtype=np.float64)
        prey_death[grid != 1] = np.nan
        
        kernel = PPKernel(30, 30, "moore")
        for _ in range(10):
            kernel.update(grid, prey_death, 0.2, 0.05, 0.2, 0.1)
        
        results.append(grid.copy())
    
    assert np.array_equal(results[0], results[1]), "Results should be deterministic"

================================================================================
WHY THIS WORKS
================================================================================

1. When you call `set_numba_seed(seed)` from Python, it invokes the @njit function
2. Inside the JIT context, `np.random.seed(seed)` seeds Numba's internal RNG
3. All subsequent calls to `np.random.random()`, `np.random.randint()`, etc. 
   inside @njit functions will use this seeded state
4. The RNG state persists across JIT function calls until re-seeded

Note: Each worker process in parallel execution needs its own seed call.
For parallel simulations, use different seeds per worker (e.g., base_seed + worker_id).

```

5. Help me write additional tests for the hunting feature logic using the numba kernels.  The additional tests will be added to the test_numba and test_pp_analysis test files and should adhere to their exisiting implementation logic. If you falsify tests, you will be replaced.

6. Write a final smoke test for the HPC simulation. Tests module imports. numba kernel, a full mock simulation, the pcf computation, cluster measurement, seeding and the binary roundtrip for saving output.

7. Use the attached legacy simulation function to compute benchmarking resukts for our optimization. Include functionality to save in a csv and plots showing the most significant results. Include flags to run with or without plots and csv output.


8. Write a few run mock tests for the analysis file to see that the plots render properly.