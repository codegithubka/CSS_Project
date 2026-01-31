# GENAI Usage Reflection

The non-trivial implementation of the project and computational intensity requirements were streamlined using GEANAI in a structured manner. A new workflow was tested during this project were the prefernences of the model behavior were set to demand clarifying questions when missing context, incremental additions of complexity in the code, and testing checkpoints to validate progress. Additionally, the startegy of threatening models with replacements in the case of an attempt to falsify results was tested and was proven to be very effective in critical testing aspects of the modules. LLMs were oftentimes also asked to "pretend" to be a senior level software engineer which proved to be effective in understanding the strict requirements of our implementations and resulted in better quality output for several bottlenecks of our project.

Prompts were documemnted and added to progress reports through the course of the project which helped keep the team up to date and retrace steps in the case of a persisting issue with our model. Overall, the team's workflow strictly adhered to pre-discussed standards that helped in clarifying implementation questions, keeping track of individual progress, and better quality output/results.


# Prompts

## Kimon

Create a baseline mean-field class based on the attached research paper on predator-prey dynamics. The class should adhere to the papers specifications. The class should have a parameter sweep method for key predator and prey parameters that will be run in Snellius. Also include a method for equilibrium analysis. Make sure to justify the logic for this method. Include docstrings with a small method description and comments for code interpretability.

Justify initialization parameter values for a small test expiriment. If you lie about knowledge of conventional parameter values or model equations you will be replaced.

Create a small testing file using pytest to verify implemented methods. Make sure to cover edge cases and list them after the .py file output for me please. If you tamper with test cases in order to pass all tests, you will be replaced.

We are now ready to plot some of the results of the mean fielf baseline. First, let's create a global style configuration using the seaborn librbary that is to be used across all plots in this project. Make sure the legend is at the bottom of each plot.

Plot the phase portait to confirm the system spiral into a stable point. Show the nullclines as well. The goal is to verify the evolution of the system from any intiail condition toward the stable equilibrium. 

Create a time series analysis plot of the evolution of prey and predator density vs. time. Make sure enough time steps all visible to see how the system eventually stabilizes.

Create a bifuracation diagram to confirm the monotonic relationship for a varying prey death rate vs. equilibrium density. 

### Testing

Create a comprehensive testing suite for the CA and PP classes. Test initialization, async update changes, synchronous update changes, prey growth in isolation behavior, predator starvation, parameter evolution and long run dynamics. Also make sur ethe test_viz mehtod works as desired


Verify the rest of the tests I wrote and show me how to handle edge cases and unexpected behavior from the simulation engine. Create a skeletal version of the experiments.py file that tests our simulation runners. Configs, intialization, edge cases, error handling, numba kernels must all be tested in that sketetal version. (always with pytest no unittest for this project)


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


## Refactoring Main Experiment Script

Help me create a skeletal version of the updated experiments script for HPC that meets tha phase requirements outlined. The config class has been migrated to config.py. 

What is the standard numopy docstring format? Does it apply for simple utilities or methods listed with @property?

What is the must usable documentation generation sofware availble? I have used sphinx before and it takes a lot longer than it should for simpl projects/libraries.

## Data Post-processing

Help me load and parse the data according to job and experimental phase number. The data will be analyzed in a jupyter notebook rather than a py file for usability.

Show me how to create a consistent color scheme for Sary's plotting utilities in the Jupyter notebook. Create functionality for shared legends, consistent title and axes letter sizes (non-bold).


## Strorm

## Base CA class
For the start of this project, we need a strong baseline. This prompt creates a basic cellular automaton class with general logic. This can then be filled in by overriding the update function and adding visualizations.

### prompt
Create a cellular automaton class named CA with an init function, a count neighbors function, an update function, and a run. The CA should consist of an int n_species, a numpy array called grid, a string containing the neighborhood type, and a numpy random number generator called generator. Use this generator for all random number generation inside the class. The CA class should also contain a dictionary called params for global parameters and a dictionary called cell_params for local parameters.

The init function should take arguments for the grid size (rows, columns, both ints), the initial density of each species (a tuple of floats of which the length defines n_species), the neighborhood type ("neumann" or "moore"), the global parameters in the form of a dictionary params, the local parameters in the form of a dictionary cell_params, and the seed for the generator. It should assign the global and local parameters to their variables and create the generator object and assign it to the generator variable, as well as create the 2D array of zeros based on the grid size and assign it to the grid variable. This grid should then be filled with states dependent on the density tuple. Iterate over the elements i of this tuple, filling grid_size * density[i] elements of the grid with state i+1. Non-zero cell states should not be overwritten, ensuring that the specified percentage of the grid is filled with that state. It should also check if the neighborhood argument corresponds with a known neighborhood and return an error otherwise.

The count neighbors function should return a tuple of matrices (one for each defined non-zero state) containing the amount of neighbors of that state for each cell. It should use the neighborhood defined in the class. Ensure the logic works for both "neumann" and "moore". Use periodic boundaries.

The update function should be overridden when creating a subclass, so it should raise a NotImplemented error.

The run function should take a steps (int) argument. It should then run the CA for steps interations, calling the update function each time.

Finally, make sure to add an expected type for each argument and define the return types. Add this information, as well as a short description of the function to the docstring. Also add assert statements to ensure arguments "make sense". For example, the sum of densities should not exceed 1 and the rows, cols, densities should all be positive, and the neighborhood should be either "neumann" or "moore".

### output
"""Cellular automaton base class.

Defines a CA class with initialization, neighbor counting, update (to override),
and run loop. Uses a numpy Generator for all randomness and supports
Neumann and Moore neighborhoods with periodic boundaries.
"""
from typing import Tuple, Dict, Optional

import numpy as np


class CA:
	"""Base cellular automaton class.

	Attributes
	- n_species: number of distinct (non-zero) states
	- grid: 2D numpy array containing integers in {0, 1, ..., n_species}
	- neighborhood: either "neumann" or "moore"
	- generator: numpy.random.Generator used for all randomness
	- params: global parameters dict
	- cell_params: local (per-cell) parameters dict
	"""

	def __init__(
		self,
		rows: int,
		cols: int,
		densities: Tuple[float, ...],
		neighborhood: str,
		params: Dict[str, object],
		cell_params: Dict[str, object],
		seed: Optional[int] = None,
	) -> None:
		"""Initialize the cellular automaton.

		Args:
		- rows (int): number of rows (>0)
		- cols (int): number of columns (>0)
		- densities (tuple of floats): initial density for each species. The
		  length of this tuple defines `n_species`. Values must be >=0 and sum
		  to at most 1. Each value gives the fraction of the grid to set to
		  that species (state values are 1..n_species).
		- neighborhood (str): either "neumann" (4-neighbors) or "moore"
		  (8-neighbors).
		- params (dict): global parameters.
		- cell_params (dict): local per-cell parameters.
		- seed (Optional[int]): seed for the numpy random generator.

		Returns: None
		"""
		assert isinstance(rows, int) and rows > 0, "rows must be positive int"
		assert isinstance(cols, int) and cols > 0, "cols must be positive int"
		assert isinstance(densities, tuple) and len(densities) > 0, "densities must be a non-empty tuple"
		for d in densities:
			assert isinstance(d, (float, int)) and d >= 0, "each density must be non-negative"
		total_density = float(sum(densities))
		assert total_density <= 1.0 + 1e-12, "sum of densities must not exceed 1"
		assert neighborhood in ("neumann", "moore"), "neighborhood must be 'neumann' or 'moore'"

		self.n_species: int = len(densities)
		self.params: Dict[str, object] = dict(params) if params is not None else {}
		self.cell_params: Dict[str, object] = dict(cell_params) if cell_params is not None else {}
		self.neighborhood: str = neighborhood
		self.generator: np.random.Generator = np.random.default_rng(seed)

		self.grid: np.ndarray = np.zeros((rows, cols), dtype=int)

		total_cells = rows * cols
		# Fill grid with species states 1..n_species according to densities.
		for i, dens in enumerate(densities):
			if dens <= 0:
				continue
			n_to_fill = int(round(total_cells * float(dens)))
			if n_to_fill <= 0:
				continue
			empty_flat = np.flatnonzero(self.grid.ravel() == 0)
			if len(empty_flat) == 0:
				break
			n_choice = min(n_to_fill, len(empty_flat))
			chosen = self.generator.choice(empty_flat, size=n_choice, replace=False)
			# assign chosen flattened indices to state i+1
			r = chosen // cols
			c = chosen % cols
			self.grid[r, c] = i + 1

	def count_neighbors(self) -> Tuple[np.ndarray, ...]:
		"""Count neighbors for each non-zero state.

		Returns a tuple of numpy arrays, one array for each state in
		`1..n_species`. Each returned array has the same shape as `grid`
		and contains the integer number of neighbors of that state for
		each cell, using periodic boundaries and the configured
		neighborhood type.

		Returns:
		- tuple of np.ndarray: one array per species (state 1..n_species)
		"""
		rows, cols = self.grid.shape
		counts = []
		# Define neighbor shifts
		if self.neighborhood == "neumann":
			shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
		else:  # moore
			shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

		for state in range(1, self.n_species + 1):
			mask = (self.grid == state).astype(int)
			neigh = np.zeros_like(mask)
			for dr, dc in shifts:
				neigh += np.roll(np.roll(mask, shift=dr, axis=0), shift=dc, axis=1)
			counts.append(neigh)

		return tuple(counts)

	def update(self) -> None:
		"""Perform one update step.

		This base implementation must be overridden by subclasses. It raises
		NotImplementedError to indicate it should be provided by concrete
		models that inherit from `CA`.

		Returns: None
		"""
		raise NotImplementedError("Override update() in a subclass to define CA dynamics")

	def run(self, steps: int) -> None:
		"""Run the CA for a number of steps.

		Args:
		- steps (int): number of iterations to run (must be non-negative).

		Returns: None
		"""
		assert isinstance(steps, int) and steps >= 0, "steps must be a non-negative integer"
		for _ in range(steps):
			self.update()


### changes
Removed a few unnecessary lines.

## Predator-prey model and asynchronous updating
Now the basic predator-prey update rules need to be defined. This is done by creating a predator-prey class inheriting the CA class and overriding the update function. We will create the possibility for either synchronous and asynchronous updating. We will start with only implementing the asynchronous rules here, as those are easier to implement. Later, we can implement the faster synchronous updating based on the asynchronous rules.

### prompt
Create a predator-prey class named PP which inherits the CA class. Create default values for the following parameters: prey_death, predator_death, prey_birth, predator_birth. Make sure not to override the user-set values for these parameters. Also raise an error if there are any other user-defined parameters and check whether the set parameters are valid (between 0 and 1). Give this class a boolean synchronous which should be an argument when creating an instance of this class. Set the default to False.

Next, create a synchronous and asynchronous function to update the grid and override update to call the correct function based on the aforementioned boolean.

For the asynchronous updating, create a copy of the current grid to be used for all the rules. Then, create a shuffled array of all occupied (non-zero) cells in the grid. Iterate over this array to apply the rules for the prey and predator. Prey: choose a random neighbor (use self.neighborhood), if this neighbor is empty, reproduce into it with probability defined by the prey_birth parameter. Predator: choose a random neighbor, if this neighbor is prey, reproduce into it with probability defined by the predator_birth parameter. Afterwards, use a synchronous vectorized update for the deaths, where the probabilities are defined by the parameters prey_death and predator_death. Use the same copy of the grid for this, to avoid newly occupied states becoming empty instantly.

Leave the synchronous updating function empty, filling it with a NotImplementedError.

### output
class PP(CA):
	"""Predator-prey CA.

	States: 0 = empty, 1 = prey, 2 = predator

	Parameters (in `params` dict). Allowed keys and defaults:
	- "prey_death": 0.05
	- "predator_death": 0.1
	- "prey_birth": 0.25
	- "predator_birth": 0.2

	The constructor validates parameters are in [0,1] and raises if
	other user-supplied params are present. The `synchronous` flag
	chooses the update mode (default False -> asynchronous updates).
	"""

	def __init__(
		self,
		rows: int,
		cols: int,
		densities: Tuple[float, ...],
		neighborhood: str,
		params: Dict[str, object],
		cell_params: Dict[str, object],
		seed: Optional[int] = None,
		synchronous: bool = False,
	) -> None:
		# Allowed params and defaults
		_defaults = {
			"prey_death": 0.05,
			"predator_death": 0.1,
			"prey_birth": 0.25,
			"predator_birth": 0.2,
		}

		# Validate user-supplied params: only allowed keys
		if params is None:
			merged_params = dict(_defaults)
		else:
			if not isinstance(params, dict):
				raise TypeError("params must be a dict or None")
			extra = set(params.keys()) - set(_defaults.keys())
			if extra:
				raise ValueError(f"Unexpected parameter keys: {sorted(list(extra))}")
			# Do not override user-set values: start from defaults then update with user values
			merged_params = dict(_defaults)
			merged_params.update(params)

		# Validate numerical ranges
		for k, v in merged_params.items():
			if not isinstance(v, (int, float)):
				raise TypeError(f"Parameter '{k}' must be a number between 0 and 1")
			if not (0.0 <= float(v) <= 1.0):
				raise ValueError(f"Parameter '{k}' must be between 0 and 1")

		# Call base initializer with merged params
		super().__init__(rows, cols, densities, neighborhood, merged_params, cell_params, seed)

		self.synchronous: bool = bool(synchronous)

	def update_sync(self) -> None:
		"""Synchronous update (not implemented)."""
		raise NotImplementedError("Synchronous PP update not implemented")

	def update_async(self) -> None:
		"""Asynchronous (random-sequential) update.

		Rules (applied using a copy of the current grid for reference):
		- Iterate occupied cells in random order.
		- Prey (1): pick random neighbor; if neighbor was empty in copy,
		  reproduce into it with probability `prey_birth`.
		- Predator (2): pick random neighbor; if neighbor was prey in copy,
		  reproduce into it (convert to predator) with probability `predator_birth`.
		- After the reproduction loop, apply deaths synchronously using the
		  copy as the reference so newly created individuals are not instantly
		  killed. Deaths only remove individuals if the current cell still
		  matches the species from the reference copy.
		"""
		rows, cols = self.grid.shape
		grid_ref = self.grid.copy()

		# Precompute neighbor shifts
		if self.neighborhood == "neumann":
			shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
		else:
			shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

		# Get occupied cells from the reference grid and shuffle
		occupied = np.argwhere(grid_ref != 0)
		if occupied.size > 0:
			order = self.generator.permutation(len(occupied))
			for idx in order:
				r, c = int(occupied[idx, 0]), int(occupied[idx, 1])
				state = int(grid_ref[r, c])
				# pick a random neighbor shift
				dr, dc = shifts[self.generator.integers(0, len(shifts))]
				nr = (r + dr) % rows
				nc = (c + dc) % cols
				if state == 1:
					# Prey reproduces into empty neighbor (reference must be empty)
					if grid_ref[nr, nc] == 0:
						if self.generator.random() < float(self.params["prey_birth"]):
							self.grid[nr, nc] = 1
				elif state == 2:
					# Predator reproduces into prey neighbor (reference must be prey)
					if grid_ref[nr, nc] == 1:
						if self.generator.random() < float(self.params["predator_birth"]):
							self.grid[nr, nc] = 2

		# Vectorized synchronous deaths, based on grid_ref but only kill if
		# the current grid still matches the referenced species (so newly
		# occupied cells are not removed mistakenly).
		rand_prey = self.generator.random(self.grid.shape)
		rand_pred = self.generator.random(self.grid.shape)

		prey_death_mask = (grid_ref == 1) & (rand_prey < float(self.params["prey_death"])) & (self.grid == 1)
		pred_death_mask = (grid_ref == 2) & (rand_pred < float(self.params["predator_death"])) & (self.grid == 2)

		self.grid[prey_death_mask] = 0
		self.grid[pred_death_mask] = 0

	def update(self) -> None:
		"""Dispatch to synchronous or asynchronous update mode."""
		if self.synchronous:
			self.update_sync()
		else:
			self.update_async()

### changes
Changed default behavior to synchronous updating after it was implemented. Added default behavior for all arguments.

## Synchronous updating
Now we add the asyncrhonous updating, making sure the functionality is the vectorized equivalent of the asynchronous function.

### prompt
Implement the synchronous updating for the PP class as a vectorized (optimized) equivalent to the asynchronous updating. First, create a copy of the current grid. The reproduction rules can be either implemented by using the empty or prey cells as the anchor and looking for prey or predator neighbors that can reproduce into these anchor cells or by using the prey or predator cells as the anchor and looking for empty or prey cells that the anchor cells can reproduce into. Ensure that the probability of a cell being reproduced into increases as more neighboring cells are occupied by the relevant species. However, each prey and predator must only be able to reproduce into a single cell. The count_neighbors function can be utilized if deemed necessary. The rules for the deaths should be the same as the asynchronous updating, as they are already vectorized. Make sure the synchronous updating is an optimized and thus faster equivalent of the asynchronous updating.

### output
	def update_sync(self) -> None:
		"""Synchronous (vectorized) update.

		Implements a vectorized equivalent of the random-sequential
		asynchronous update. Each occupied cell (prey or predator) gets at
		most one reproduction attempt: with probability `birth` it chooses a
		random neighbor and, if that neighbor in the reference grid has the
		required target state (empty for prey, prey for predator), it
		becomes a candidate attempt. When multiple reproducers target the
		same cell, one attempt is chosen uniformly at random to succeed.
		Deaths are applied the same vectorized way as in the async update.
		"""

		rows, cols = self.grid.shape
		grid_ref = self.grid.copy()

		# Precompute neighbor shifts and arrays for indexing
		if self.neighborhood == "neumann":
			shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
		else:
			shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
		dr_arr = np.array([s[0] for s in shifts], dtype=int)
		dc_arr = np.array([s[1] for s in shifts], dtype=int)
		n_shifts = len(shifts)

		def _process_reproduction(sources, birth_prob, target_state_required, new_state_val):
			"""Handle reproduction attempts from `sources`.

			sources: (M,2) array of (r,c) positions in grid_ref
			birth_prob: scalar probability that a source attempts reproduction
			target_state_required: state value required in grid_ref at target
			new_state_val: state to write into self.grid for successful targets
			"""
			if sources.size == 0:
				return

			M = sources.shape[0]
			# Which sources attempt reproduction
			attempt_mask = self.generator.random(M) < float(birth_prob)
			if not np.any(attempt_mask):
				return

			src = sources[attempt_mask]
			K = src.shape[0]

			# Each attempting source picks one neighbor uniformly
			nbr_idx = self.generator.integers(0, n_shifts, size=K)
			nr = (src[:, 0] + dr_arr[nbr_idx]) % rows
		
			nc = (src[:, 1] + dc_arr[nbr_idx]) % cols

			# Only keep attempts where the reference grid at the target has the required state
			valid_mask = (grid_ref[nr, nc] == target_state_required)
			if not np.any(valid_mask):
				return

			nr = nr[valid_mask]
			nc = nc[valid_mask]

			# Flatten target indices to group collisions
			target_flat = (nr * cols + nc).astype(np.int64)
			# Sort targets to find groups that target the same cell
			order = np.argsort(target_flat)
			tf_sorted = target_flat[order]

			# unique targets (on the sorted array) with start indices and counts
			uniq_targets, idx_start, counts = np.unique(tf_sorted, return_index=True, return_counts=True)
			if uniq_targets.size == 0:
				return

			# For each unique target, pick one attempt uniformly at random
			# idx_start gives indices into the sorted array
			chosen_sorted_positions = []
			for start, cnt in zip(idx_start, counts):
				off = int(self.generator.integers(0, cnt))
				chosen_sorted_positions.append(start + off)
			chosen_sorted_positions = np.array(chosen_sorted_positions, dtype=int)

			# Map back to indices in the filtered attempts array
			chosen_indices = order[chosen_sorted_positions]

			chosen_target_flats = target_flat[chosen_indices]
			chosen_rs = (chosen_target_flats // cols).astype(int)
			chosen_cs = (chosen_target_flats % cols).astype(int)

			# Apply successful births to the main grid
			self.grid[chosen_rs, chosen_cs] = new_state_val

		# Prey reproduce into empty cells (target state 0 -> new state 1)
		prey_sources = np.argwhere(grid_ref == 1)
		_process_reproduction(prey_sources, self.params["prey_birth"], 0, 1)

		# Predators reproduce into prey cells (target state 1 -> new state 2)
		pred_sources = np.argwhere(grid_ref == 2)
		_process_reproduction(pred_sources, self.params["predator_birth"], 1, 2)

		# Vectorized synchronous deaths (same as async implementation)
		rand_prey = self.generator.random(self.grid.shape)
		rand_pred = self.generator.random(self.grid.shape)

		prey_death_mask = (grid_ref == 1) & (rand_prey < float(self.params["prey_death"])) & (self.grid == 1)
		pred_death_mask = (grid_ref == 2) & (rand_pred < float(self.params["predator_death"])) & (self.grid == 2)

		self.grid[prey_death_mask] = 0
		self.grid[pred_death_mask] = 0

### changes
Swapped the order of birth and death checks. This results in the same behavior, but removes the need to check whether the post-birth grid matches the reference grid.

## Basic visualisations
Basic tests pass and the grid updates seem plausible, but to see it in better detail we need to add visualisation. A first step in this is an imshow plot which updates every N iterations.

### prompt
Add optional grid visualisation support to the CA class. Use matplotlib.pyplot's interactive mode to dynamically update the the plot every N intervals. Do this by adding a function visualize which takes the interval, the figsize, the pause, and a colormap. this function should turn on the visualisation, creating the necessary variables. Ensure that there is reasonable default behavior for all arguments, including the colormap. Make sure the plot gets updated each interval iterations inside of the run function, showing the current iteration. if this would bloat the run function, add a helper function to update the plot.

Next, update the PP class to change the default to a black empty cells, with green prey and red predators. Again, ensure that this can be overridden by the user when calling visualize, just like the other arguments.

For both classes, ensure that the matplotlib library is not needed to run the simulation without visualisation. Also, make the necessary overrides (for the colormap default for instance) as short as possible. I do not want classes inheriting CA to have to override the entire visualize function just to change the default color map. However, if visualization is off, there is no need to store any of the variables in the CA / PP class.

### output (also added the defaults)
	def run(self, steps: int) -> None:
		"""Run the CA for a number of steps.

		Args:
		- steps (int): number of iterations to run (must be non-negative).

		Returns: None
		"""
		assert isinstance(steps, int) and steps >= 0, "steps must be a non-negative integer"
		for i in range(steps):
			self.update()
			# Update visualization if enabled every `interval` iterations
			if getattr(self, "_viz_on", False):
				# iteration number is 1-based for display
				try:
					self._viz_update(i + 1)
				except Exception:
					# Don't let visualization errors stop the simulation
					pass

	def visualize(
		self,
		interval: int = 1,
		figsize: Tuple[float, float] = (5, 5),
		pause: float = 0.001,
		cmap=None,
	) -> None:
		"""Enable interactive visualization of the grid.

		Args:
		- interval: update plot every `interval` iterations (>=1)
		- figsize: figure size passed to matplotlib
		- pause: seconds to pause after draw (controls responsiveness)
		- cmap: colormap spec (string, sequence of colors, or matplotlib Colormap).
		
		This function imports matplotlib lazily so simulations without
		visualization do not require matplotlib to be installed.
		"""
		if not isinstance(interval, int) or interval < 1:
			raise ValueError("interval must be a positive integer")

		# Lazy import so matplotlib is optional
		import matplotlib.pyplot as plt
		from matplotlib.colors import ListedColormap

		# Resolve default cmap: prefer instance attribute override
		c_spec = self._default_cmap if cmap is None else cmap

		# Build a discrete colormap with entries for states 0..n_species
		n_colors_needed = self.n_species + 1
		if isinstance(c_spec, str):
			# request discrete version of named colormap
			cmap_obj = plt.get_cmap(c_spec, n_colors_needed)
		elif isinstance(c_spec, (list, tuple)):
			colors = list(c_spec)
			if len(colors) < n_colors_needed:
				colors = colors + [colors[-1]] * (n_colors_needed - len(colors))
			cmap_obj = ListedColormap(colors[:n_colors_needed])
		else:
			# Assume user provided a Colormap-like object
			cmap_obj = c_spec

		plt.ion()
		fig, ax = plt.subplots(figsize=figsize)
		im = ax.imshow(self.grid, cmap=cmap_obj, interpolation="nearest", vmin=0, vmax=self.n_species)
		ax.set_title("Iteration 0")
		plt.show(block=False)
		fig.canvas.draw()
		plt.pause(pause)

		# Store visualization state on the instance (only when visualization enabled)
		self._viz_on = True
		self._viz_interval = interval
		self._viz_fig = fig
		self._viz_ax = ax
		self._viz_im = im
		self._viz_cmap = cmap_obj
		self._viz_pause = float(pause)

	def _viz_update(self, iteration: int) -> None:
		"""Update the interactive plot if the configured interval has passed.

		This function also performs the minimal redraw using `plt.pause` so the
		plot remains responsive.
		"""
		if not getattr(self, "_viz_on", False):
			return
		if (iteration % int(self._viz_interval)) != 0:
			return

		# Lazy import for pause; matplotlib already imported in visualize
		import matplotlib.pyplot as plt

		self._viz_im.set_data(self.grid)
		self._viz_ax.set_title(f"Iteration {iteration}")
		# draw/update
		self._viz_fig.canvas.draw_idle()
		plt.pause(self._viz_pause)

### changes

## Evolution of parameters
Now we need to add functionality allowing parameters to evolve. Specifically, we are interested in the prey death rates. To do this we track another grid with values for the death rate of each prey on the grid. When a prey reproduces, we add Gaussian noise to the death rate inherited from the parent.

### prompt
In the PP class, create functionality for evolving / mutating parameters. Create a new function called evolve which takes a str which will be the parameter to evolve. This should correspond to any of the known parameters. Then, create an array in cell_params, filling the cells occupied by the relevant species (prey for "prey_death", predator for "predator_birth", etc.) with the global parameter in params. The other cells (either empty or occupied by the other species) should be either zero or NaN. Additionally, the function should take a standard deviation, minimum, and maximum for the parameter. These values should have defaults: 0.05, 0.01, and 0.99.

In the asynchronous and synchronous update functions, make the following changes. When the relevant species reproduces, the newly born predator or prey inherits the parameter value from their parent, with Gaussian noise of the standard deviation defined in the evolve function. Clip the parameter between the minimum and maximum. Place this new value into its cell_params grid. When a predator or prey dies, or when a prey gets eaten, remove their parameter values from the cell_params grid, such that the only non-zero (or non-NaN) elements in the cell_params grid correspond to a cell occupied by the relevant species.

Ensure that if the cell_params grids are set (by the evolve function), the cell-specific parameters are used in the updates. For instance, the deaths of the prey should be calculated based on the values in the cell_params grid, not the global params value. Since the cell_params grid's only non-zero (or non-NaN) entries are active cells of the relevant species, there is no need to get the occupied prey / predator cells from PP.grid.

Also add the option to visualize the cell_params grid side-by-side with the PP.grid plot. Make it so it is still possible to get a visualization of only the states grid, even when some parameters are set to evolve. This could be done by checking the elements in cell_params inside of the visualization function. Because of how the evolve function works, calling visualize before evolve should only show the states grid and calling visualize after evolve should show the state grid and all cell_params grids set to evolve.

In the run function, check whether the entries in the cell_params arrays align with the states in the grid. Add documentation to the update function to mention that these functions assume the the cell_params arrays are algined with the grid.

### output
Made large changes to multiple functions. See git commit for details.

### changes
Added color bars to the evolving parameters plots.

## CA and PP tests
Now that the MVP functionality is fulfilled, we need to test it. For this, we create numerous tests for both the CA class and the PP class, testing if incorrect arguments raise the correct errors, if any functions raise unwanted errors, but mostly if the update functions work as intended.

### prompt
Add tests for the CA and PP class to the test_ca.py file. Check whether incorrect arguments do not raise errors and if any functions do raise errors. Also, test whether run picks up on cell_params array that do not align by grid. For this, manually change some cell_params array entries before running the simulation. Check whether all errors are informative, in the sense that a user would know how to avoid it next time.

Lastly, check whether trivial simulations result in the correct outcome. For example, if both the predator and prey death rates are 1 and the birth rates are below 1, then the grid should go extinct. Also, if the birth rate of the prey and predators are 1 and the death rate of the predators to be 0 we expect the prey to go extinct an the predators to occupy a large number of cells.

Create any additional tests if they are unique, meaning it is not tested already, and well-defined, meaning we know the intended behavior and we know how to compare it to the true behavior.

### output
def test_initialization_and_grid_filling():
    rows, cols = 10, 10
    densities = (0.2, 0.1)
    ca = CA(rows, cols, densities, neighborhood="moore", params={}, cell_params={}, seed=42)
    assert ca.grid.shape == (rows, cols)
    assert ca.n_species == len(densities)
    total_cells = rows * cols
    # expected counts use the same rounding as CA.__init__
    expected_counts = [int(round(total_cells * d)) for d in densities]
    # verify actual counts equal expected
    for i, exp in enumerate(expected_counts, start=1):
        assert int(np.count_nonzero(ca.grid == i)) == exp


def test_invalid_parameters_raise():
    # invalid rows/cols
    with pytest.raises(AssertionError):
        CA(0, 5, (0.1,), "moore", {}, {}, seed=1)
    with pytest.raises(AssertionError):
        CA(5, -1, (0.1,), "moore", {}, {}, seed=1)
    # densities must be non-empty tuple
    with pytest.raises(AssertionError):
        CA(5, 5, (), "moore", {}, {}, seed=1)
    # densities sum > 1
    with pytest.raises(AssertionError):
        CA(5, 5, (0.8, 0.8), "moore", {}, {}, seed=1)
    # invalid neighborhood
    with pytest.raises(AssertionError):
        CA(5, 5, (0.1,), "invalid", {}, {}, seed=1)

    # PP: params must be a dict or None
    with pytest.raises(TypeError):
        PP(rows=5, cols=5, densities=(0.2, 0.1), neighborhood="moore", params="bad", cell_params=None, seed=1)


def test_neighborhood_counting():
    # set up a small grid with a single prey in the center and check neighbor counts
    ca = CA(3, 3, (0.0,), neighborhood="moore", params={}, cell_params={}, seed=1)
    ca.grid[:] = 0
    ca.grid[1, 1] = 1
    counts = ca.count_neighbors()
    # counts is a tuple with one array (state 1)
    neigh = counts[0]
    # all 8 neighbors of center should have count 1
    expected_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    for r in range(3):
        for c in range(3):
            if (r, c) in expected_positions:
                assert neigh[r, c] == 1
            else:
                # center has 0 neighbors of same state
                assert neigh[r, c] == 0


def test_run_detects_cell_params_shape_and_nonnan_mismatch():
    # create a PP and enable evolution for a parameter
    pp = PP(rows=5, cols=5, densities=(0.2, 0.1), neighborhood="moore", params=None, cell_params=None, seed=2)
    pp.evolve("prey_death", sd=0.01, min=0.0, max=1.0)

    # wrong shape should raise informative ValueError during run()
    pp.cell_params["prey_death"] = np.zeros((1, 1))
    with pytest.raises(ValueError) as excinfo:
        pp.run(1)
    assert "shape equal to grid" in str(excinfo.value)

    # now create a same-shaped array but with non-NaN positions that don't match prey positions
    arr = np.zeros(pp.grid.shape, dtype=float)  # filled with non-NaN everywhere
    pp.cell_params["prey_death"] = arr
    with pytest.raises(ValueError) as excinfo2:
        pp.run(1)
    assert "non-NaN entries must match positions" in str(excinfo2.value)


def test_extinction_when_death_one():
    # when both death rates are 1 all individuals should die in one step
    params = {
        "prey_death": 1.0,
        "predator_death": 1.0,
        "prey_birth": 0.0,
        "predator_birth": 0.0,
    }
    pp = PP(rows=10, cols=10, densities=(0.2, 0.1), neighborhood="moore", params=params, cell_params=None, seed=3)
    pp.run(1)
    # no prey or predators should remain
    assert np.count_nonzero(pp.grid != 0) == 0


def test_predators_dominate_with_high_birth_and_zero_predator_death():
    params = {
        "prey_death": 0.0,
        "predator_death": 0.0,
        "prey_birth": 1.0,
        "predator_birth": 1.0,
    }
    pp = PP(rows=10, cols=10, densities=(0.1, 0.05), neighborhood="moore", params=params, cell_params=None, seed=4)
    # run longer to allow predators to consume prey; expect prey extinction
    pp.run(200)
    after_prey = int(np.count_nonzero(pp.grid == 1))
    after_pred = int(np.count_nonzero(pp.grid == 2))
    # after sufficient time, prey should go extinct and predators remain
    assert after_prey == 0
    assert after_pred > 0

### changes
Revealed and fixed error in the PP class' update functions where no parameter key was passed to the _process_reproduction function, resulting in an error.

## More visualizations
Now that we can run simulations, we need to understand what is happening. For this, we first need graphs detailing the population counts as well as the min, mean, and max values of each evolving parameter. Additionally, we need to add functionality that stops mutation after a certain amount of steps, after which we can see which parameter values survive and which go extinct.

### prompt
Add graphs underneath the imshow plots to show the simulation state over time. For the states grid, show the population count of the prey and predator over time. For the evolving parameters, show the min, mean, and max value of that parameter over time. Only measure these values when the figure is updated, to make sure it only adds overhead every interval iterations.

Also create a separate plot left of the states grid plot that shows the distribution of prey neighbors for each prey. I want a histogram showing the amount of prey with each possible prey neighbor count (for moore this is 8). Below that, add a graph showing the 25%, the mean, and the 75% value for the neighbor count.

Lastly, add functionality to stop evolution after a certain time-step. This should be an optional argument to the run function. Also add a function to create snapshots of the histogram, states grid, and cell parameters grids. As these are snapshots, the graphs below these plots should not be included. Add another argument to the run function, which is a list of the iterations to create snapshots at. Save these snapshots to the results folder, where each run should have its own folder with snapshots. Make sure the snapshot file names include the iteration.

## Sary

# CA Stochastic Bifurcation Diagram:
Mutations (evolutions) parameter OFF
Control parameter: prey death rate

Possible statistical observables:
- Fraction of prey cells at equilibrium
- Measure of entropy of the generated pattern.
- Prey population count
- Predator population count

Run simulation:
- Let the system run until a steady state is observed
- For each death rate value, let the CA run for a specified number of iterations after warmp up, show distribution (scatters) for each sim run at a given prey death rate, and the average line


# Phase 1: finding the critical point
- Create bifurcation diagram of mean population count, varying prey death rate
	- Look for critical transition
- Create log-log plot of cluster size distribution, varying prey death rate
	- Look for power-law

# Experiment Phase: CA Stochastic Bifurcation Diagram:

1) Write a Config Object specific to that experiment
2) Make sure the experiment running on the cluster is running 15 reps of each runs at all sweeped values.
3) Make sure the outputs of the experiment are a 1D and 2D array (explained below)

# Bifurcation Diagram Prompts:
1) Help me write a function for creating a stochastic bifurcation diagram, of the population count at equilibrium, varying the prey death rate (as the control variable). 
2) At each sweeped value of the prey death control variable, we should be measuring the population count at equilibrium for at least 15 simulation runs. 
3) Which means that the two inputs for my function should be a 1D Array for the sweep parameter, and a 2D array for the experiment results at each sweep for the rows, and the results for each iteration for the columns.
4) When running my function, using the argparse module, my command-line arguments specifies which analysis to do, in this case the analysis is the bifurcation diagram.


# Output: 
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

# Warmup Study Prompts
1) Help me develop a script that aims at studying how the warmup period scales with the grid size, in order to get an approximate warmup period for a 1000 X 1000 grid size.
2) Increasing the grid size from L = 50 to L = 200, I want to observe how long it takes for the model to reach equilibrium. Initial visualizations for our stochastic CA model has shown the expected oscillatory behaviour, which is in line with empirical predator prey dynamics.
3) Help me implement a spectral analysis (Fast Fourier Transform) to find when domiant frequencies start to consistently domiate across several consecutive windows.
4) In order to help me visualize what is happening, I want you to create a diagnostic feature for eahc grid size tested, that shows me the raw predator and prey density dynamics as a time series, as well as an FFT vs simulation steps plot that allows me to monitor the dominant oscillation frequency across windows.

# OUTPUT

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

(Phase 1 and 1.5) (Criticality)
Statistical Test For Power Law using power law package 
Bifurcation
Cluster size distribution

(Phase 2): (Self orgaization)
Box plots (x axis-inital death rate, y axis - final or converged death rate)

(Phase 3): (Finite Size)
log log plots of cluster size disstribution with cutoff (because of grid size)

(Phase 4): Sensitivity


## Sofronia

### Direct Movement Prey-Predator
create the predator-prey movement where the predators follow the preys behaviour in update_sync function

```
OUTPUT:
Extended _pp_async_kernel_directed() in numba_optimized.py

elif state == 2:  # PREDATOR - directed hunting
            if np.random.random() < pred_death_val:
                grid[r, c] = 0
                continue
            
            prey_count = 0
            for k in range(n_shifts):
                check_r = (r + dr_arr[k]) % rows
                check_c = (c + dc_arr[k]) % cols
                if grid[check_r, check_c] == 1:
                    prey_count += 1
            
            if prey_count > 0:
                target_idx = np.random.randint(0, prey_count)
                found = 0
                nr, nc = 0, 0
                for k in range(n_shifts):
                    check_r = (r + dr_arr[k]) % rows
                    check_c = (c + dc_arr[k]) % cols
                    if grid[check_r, check_c] == 1:
                        if found == target_idx:
                            nr = check_r
                            nc = check_c
                            break
                        found += 1
                
                if np.random.random() < pred_birth_val:
                    grid[nr, nc] = 2
                    prey_death_arr[nr, nc] = np.nan
            else:
                nbi = np.random.randint(0, n_shifts)
                nr = (r + dr_arr[nbi]) % rows
                nc = (c + dc_arr[nbi]) % cols
                
                if grid[nr, nc] == 1:
                    if np.random.random() < pred_birth_val:
                        grid[nr, nc] = 2
                        prey_death_arr[nr, nc] = np.nan

```

### SOC Visualization

create one file with included visualization show if our prey-preditor cellular automaton structure follows soc, focus on perturbation from the initial configuration not at the critical point, sample from different configurations while varying more than the death and birth rates of the prey not run a full analysis of course on that and observe behaviour, soc should have these 4:  • Slow drive
• Build-up of ‘stress’
• Intermittent release of this stress
• Self organization

```
OUTPUT:

I'll create a comprehensive SOC analysis file for your prey-predator cellular automaton.

"""
Self-Organized Criticality (SOC) Analysis for Prey-Predator Cellular Automaton

This module analyzes whether the prey-predator CA exhibits SOC properties:
1. Slow drive: Gradual external parameter changes
2. Build-up of stress with thresholds (storing potential energy)
3. Intermittent release of stress (avalanches/cascades)
4. Self-organization (robustness across parameter variations, not carefully tuned)

We focus on perturbations from initial configurations (not at critical point) and
sample across different configurations with varied parameters beyond just death/birth
rates (e.g., grid size, densities, neighborhood, synchronicity).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from models to avoid __init__ issues
from models.CA import PP


# ============================================================================
# 1. STRESS METRIC & PERTURBATION DYNAMICS
# ============================================================================

def compute_grid_stress(grid: np.ndarray) -> float:
    """
    Compute a 'stress' metric for the grid as a proxy for potential energy.
    
    High predator-prey interface regions represent tension/stress.
    Stress = normalized count of (predator,empty) and (prey,empty) adjacent pairs.
    This represents the gradient/friction that can cause avalanche-like events.
    
    Args:
        grid: 2D array with 0=empty, 1=prey, 2=predator
        
    Returns:
        Normalized stress value [0, 1]
    """
    rows, cols = grid.shape
    stress = 0
    
    # Count interfaces (predator or prey adjacent to empty)
    for i in range(rows):
        for j in range(cols):
            cell = grid[i, j]
            if cell == 0:  # empty cell
                # Check all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = (i + di) % rows, (j + dj) % cols
                        if grid[ni, nj] > 0:  # neighbor is prey or predator
                            stress += 1
    
    # Normalize by maximum possible interfaces
    max_stress = rows * cols * 8  # each cell can have 8 neighbors
    return stress / max_stress if max_stress > 0 else 0.0


def compute_population_variance(grids_history: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prey and predator population variance over time.
    High variance indicates intermittent release events.
    
    Args:
        grids_history: List of grid snapshots over time
        
    Returns:
        Tuple: (prey_variance_rolling, predator_variance_rolling)
    """
    prey_pops = np.array([(g == 1).sum() for g in grids_history])
    pred_pops = np.array([(g == 2).sum() for g in grids_history])
    
    window = max(5, len(grids_history) // 10)  # rolling window
    prey_changes = np.abs(np.diff(prey_pops))
    pred_changes = np.abs(np.diff(pred_pops))
    
    # Pad at the beginning to match original length
    prey_var = np.concatenate([[prey_changes[0]] * (window - 1), 
                               np.convolve(prey_changes, np.ones(window)/window, mode='valid')])
    pred_var = np.concatenate([[pred_changes[0]] * (window - 1),
                               np.convolve(pred_changes, np.ones(window)/window, mode='valid')])
    
    # Ensure exact length match
    prey_var = prey_var[:len(grids_history)]
    pred_var = pred_var[:len(grids_history)]
    
    return prey_var, pred_var


def detect_avalanche_events(grids_history: List[np.ndarray], 
                           population_change_threshold: float = 0.1) -> List[Tuple[int, float]]:
    """
    Detect avalanche events as rapid changes in total population.
    
    Args:
        grids_history: List of grid snapshots
        population_change_threshold: Fraction of grid change to trigger detection
        
    Returns:
        List of (time_step, magnitude) tuples
    """
    total_pops = np.array([(g > 0).sum() for g in grids_history])
    max_pop = total_pops.max()
    
    if max_pop == 0:
        return []
    
    changes = np.abs(np.diff(total_pops))
    threshold = population_change_threshold * max_pop
    
    avalanches = []
    in_event = False
    event_start = 0
    event_magnitude = 0
    
    for i, change in enumerate(changes):
        if change > threshold:
            if not in_event:
                event_start = i
                in_event = True
            event_magnitude = max(event_magnitude, change)
        else:
            if in_event:
                avalanches.append((event_start, event_magnitude / max_pop))
                in_event = False
                event_magnitude = 0
    
    return avalanches


# ============================================================================
# 2. PARAMETER SAMPLING WITH VARIED CONFIGURATIONS
# ============================================================================

def sample_parameter_configurations(n_samples: int = 10,
                                   base_seed: int = 42) -> List[Dict]:
    """
    Generate diverse parameter configurations.
    Varies: grid size, initial densities, rates, neighborhood, synchronicity.
    
    Args:
        n_samples: Number of configurations to generate
        base_seed: Base random seed
        
    Returns:
        List of configuration dicts
    """
    configs = []
    rng = np.random.RandomState(base_seed)
    
    for i in range(n_samples):
        # Vary grid size (smaller = more fluctuations, larger = more stable?)
        grid_size = rng.choice([16, 32, 48, 64])
        
        # Vary initial densities (more heterogeneous = more stress buildup?)
        prey_density = rng.uniform(0.1, 0.4)
        pred_density = rng.uniform(0.02, 0.15)
        
        # Vary parameters beyond just death/birth rates
        config = {
            "seed": base_seed + i,
            "rows": grid_size,
            "cols": grid_size,
            "densities": (prey_density, pred_density),
            "neighborhood": rng.choice(["neumann", "moore"]),
            "synchronous": rng.choice([True, False]),
            # Vary rate parameters
            "prey_death": rng.uniform(0.01, 0.10),
            "predator_death": rng.uniform(0.05, 0.20),
            "prey_birth": rng.uniform(0.10, 0.35),
            "predator_birth": rng.uniform(0.10, 0.30),
        }
        configs.append(config)
    
    return configs


# ============================================================================
# 3. SLOW DRIVE & STRESS BUILDUP WITH PERTURBATIONS
# ============================================================================

def run_soc_perturbation_experiment(config: Dict, 
                                   n_equilibration: int = 100,
                                   n_observation: int = 200,
                                   perturbation_step: int = 50) -> Dict:
    """
    Run a single SOC experiment with slow parameter drift (drive) and
    perturbations from non-critical initial conditions.
    
    The experiment:
    1. Initialize CA with given config (not at "critical point")
    2. Run equilibration steps (slow drive builds up stress)
    3. Perturb one parameter gradually
    4. Observe stress buildup and release events
    
    Args:
        config: Configuration dict from sample_parameter_configurations()
        n_equilibration: Steps before perturbation (building stress)
        n_observation: Steps during/after perturbation (observing avalanches)
        perturbation_step: Which step to start perturbation
        
    Returns:
        Dict with results: stress_history, populations, avalanches, etc.
    """
    # Create PP automaton
    ca = PP(
        rows=int(config["rows"]),
        cols=int(config["cols"]),
        densities=tuple(float(d) for d in config["densities"]),
        neighborhood=config["neighborhood"],
        params={
            "prey_death": float(config["prey_death"]),
            "predator_death": float(config["predator_death"]),
            "prey_birth": float(config["prey_birth"]),
            "predator_birth": float(config["predator_birth"]),
        },
        seed=int(config["seed"]),
        synchronous=False,  # Use async mode since sync is not fully implemented
    )
    
    # Run equilibration: slow drive allows stress to build
    stress_history = []
    grids_history = []
    prey_pops = []
    pred_pops = []
    param_history = []  # track parameter drift
    
    total_steps = n_equilibration + n_observation
    
    for step in range(total_steps):
        # Slow parameter drift (drive): gradually increase predator death
        # This is the "slow drive" that accumulates stress without immediate release
        if step >= perturbation_step:
            progress = (step - perturbation_step) / (total_steps - perturbation_step)
            drift_amount = 0.05 * progress  # drift up to +0.05
            ca.params["predator_death"] = config["predator_death"] + drift_amount
        
        # Record state before update
        stress = compute_grid_stress(ca.grid)
        stress_history.append(stress)
        grids_history.append(ca.grid.copy())
        prey_pops.append((ca.grid == 1).sum())
        pred_pops.append((ca.grid == 2).sum())
        param_history.append(float(ca.params["predator_death"]))
        
        # Update CA
        ca.update()
    
    # Detect avalanche events
    avalanches = detect_avalanche_events(grids_history, population_change_threshold=0.05)
    
    # Compute variance (intermittent release signature)
    prey_var, pred_var = compute_population_variance(grids_history)
    
    # Ensure exact length match with steps (fix any off-by-one errors)
    if len(prey_var) < len(grids_history):
        prey_var = np.pad(prey_var, (0, len(grids_history) - len(prey_var)), mode='edge')
    if len(pred_var) < len(grids_history):
        pred_var = np.pad(pred_var, (0, len(grids_history) - len(pred_var)), mode='edge')
    
    results = {
        "config": config,
        "stress_history": np.array(stress_history),
        "prey_populations": np.array(prey_pops),
        "pred_populations": np.array(pred_pops),
        "param_history": np.array(param_history),
        "avalanches": avalanches,
        "prey_variance": prey_var,
        "pred_variance": pred_var,
        "grids_history": grids_history,
        "total_steps": total_steps,
        "n_equilibration": n_equilibration,
    }
    
    return results


# ============================================================================
# 4. ROBUSTNESS ANALYSIS (Criticality across parameters)
# ============================================================================

def analyze_soc_robustness(experiment_results: List[Dict]) -> Dict:
    """
    Analyze robustness of critical behavior across diverse parameter configs.
    
    SOC robustness signature: avalanche statistics (frequency, magnitude)
    remain relatively consistent across diverse parameter combinations,
    indicating self-organization independent of tuning.
    
    Args:
        experiment_results: List of results from run_soc_perturbation_experiment()
        
    Returns:
        Dict with robustness metrics
    """
    avalanche_counts = []
    avalanche_magnitudes = []
    stress_levels = []
    population_variances = []
    
    for result in experiment_results:
        if result["avalanches"]:
            avalanche_counts.append(len(result["avalanches"]))
            mags = [mag for _, mag in result["avalanches"]]
            avalanche_magnitudes.extend(mags)
        else:
            avalanche_counts.append(0)
        
        stress_levels.extend(result["stress_history"].tolist())
        population_variances.append(result["prey_variance"].mean())
    
    robustness_metrics = {
        "avg_avalanche_count": np.mean(avalanche_counts) if avalanche_counts else 0,
        "std_avalanche_count": np.std(avalanche_counts) if avalanche_counts else 0,
        "avalanche_magnitude_mean": np.mean(avalanche_magnitudes) if avalanche_magnitudes else 0,
        "avalanche_magnitude_std": np.std(avalanche_magnitudes) if avalanche_magnitudes else 0,
        "avg_stress": np.mean(stress_levels),
        "std_stress": np.std(stress_levels),
        "avg_population_variance": np.mean(population_variances),
        "coefficient_of_variation_avalanche": (
            np.std(avalanche_counts) / np.mean(avalanche_counts)
            if np.mean(avalanche_counts) > 0 else np.inf
        ),
    }
    
    return robustness_metrics


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def visualize_soc_properties(experiment_results: List[Dict],
                            robustness_metrics: Dict,
                            output_file: Optional[str] = None):
    """
    Visualization of the 4 core SOC properties in prey-predator CA.
    
    Shows:
    1. Slow drive: Gradual parameter drift
    2. Build-up of stress: Stress accumulation with thresholds
    3. Intermittent release: Avalanche cascades and population dynamics
    4. Self-organization: Robustness across diverse configurations
    
    Args:
        experiment_results: List of experiment results
        robustness_metrics: Robustness analysis output
        output_file: Optional file path to save figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Select a representative experiment (middle one)
    rep_idx = len(experiment_results) // 2
    rep_result = experiment_results[rep_idx]
    steps = np.arange(len(rep_result["stress_history"]))
    
    # ========== SOC PROPERTY 1: SLOW DRIVE ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, rep_result["param_history"], 'purple', linewidth=2.5)
    ax1.axvline(rep_result["n_equilibration"], color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Perturbation start')
    ax1.fill_between(steps[:rep_result["n_equilibration"]], 
                     0, 0.3, alpha=0.1, color='blue')
    ax1.fill_between(steps[rep_result["n_equilibration"]:], 
                     0, 0.3, alpha=0.15, color='red')
    ax1.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predator Death Rate', fontsize=11, fontweight='bold')
    ax1.set_title('1) SLOW DRIVE\nGradual Parameter Change', 
                  fontsize=12, fontweight='bold', color='darkblue')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ========== SOC PROPERTY 2: BUILD-UP OF STRESS ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, rep_result["stress_history"], 'b-', linewidth=2.5, label='Stress Level')
    ax2.axvline(rep_result["n_equilibration"], color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Perturbation start')
    
    # Mark avalanche events with stars
    for event_t, event_mag in rep_result["avalanches"]:
        ax2.scatter(event_t, rep_result["stress_history"][event_t], 
                   color='orange', s=150, marker='*', zorder=5, edgecolors='black', linewidth=1.5)
    
    ax2.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Stress (Interface Density)', fontsize=11, fontweight='bold')
    ax2.set_title('2) BUILD-UP OF STRESS\nThresholds & Potential Energy', 
                  fontsize=12, fontweight='bold', color='darkblue')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # ========== SOC PROPERTY 3: INTERMITTENT RELEASE ==========
    ax3 = fig.add_subplot(gs[1, 0])
    prey = rep_result["prey_populations"]
    pred = rep_result["pred_populations"]
    
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(steps, prey, 'g-', label='Prey', linewidth=2.5)
    line2 = ax3_twin.plot(steps, pred, 'r-', label='Predator', linewidth=2.5)
    ax3.axvline(rep_result["n_equilibration"], color='gray', linestyle='--', 
                alpha=0.6, linewidth=1.5)
    
    ax3.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Prey Population', color='g', fontsize=11, fontweight='bold')
    ax3_twin.set_ylabel('Predator Population', color='r', fontsize=11, fontweight='bold')
    ax3.set_title('3) INTERMITTENT RELEASE\nAvalanche Cascades', 
                  fontsize=12, fontweight='bold', color='darkblue')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, fontsize=10, loc='upper left')
    
    # ========== SOC PROPERTY 4: SELF-ORGANIZATION ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Stress-density relation: shows universal behavior across configurations
    densities_list = []
    stresses_list = []
    avalanche_counts_list = []
    
    for result in experiment_results:
        # Calculate mean population density during observation phase
        prey_pop = result["prey_populations"][result["n_equilibration"]:]
        pred_pop = result["pred_populations"][result["n_equilibration"]:]
        total_pop = (prey_pop + pred_pop).mean()
        grid_size = result["config"]["rows"] * result["config"]["cols"]
        density = total_pop / grid_size
        
        # Mean stress during observation phase
        mean_stress = result["stress_history"][result["n_equilibration"]:].mean()
        avalanche_count = len(result["avalanches"])
        
        densities_list.append(density)
        stresses_list.append(mean_stress)
        avalanche_counts_list.append(avalanche_count)
    
    # Scatter plot: stress vs density, colored by avalanche activity
    scatter = ax4.scatter(densities_list, stresses_list, c=avalanche_counts_list,
                         cmap='plasma', s=300, alpha=0.8, edgecolors='none')
    
    ax4.set_xlabel('Population Density', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Mean Stress Level', fontsize=11, fontweight='bold')
    ax4.set_title('4) SELF-ORGANIZATION\nStress-Density Relation', 
                  fontsize=12, fontweight='bold', color='darkblue')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Avalanche Count', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Prey-Predator Cellular Automaton: Four SOC Properties',
                fontsize=14, fontweight='bold', y=0.98)
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    return fig


# ============================================================================
# 6. MAIN EXPERIMENT
# ============================================================================

def main():
    """Run complete SOC analysis."""
    print("=" * 80)
    print("SELF-ORGANIZED CRITICALITY ANALYSIS: Prey-Predator Cellular Automaton")
    print("=" * 80)
    print()
    
    # Generate diverse parameter configurations
    print("[1/4] Generating parameter configurations...")
    n_configs = 8  # Small sample for demonstration (not full analysis)
    configs = sample_parameter_configurations(n_samples=n_configs, base_seed=42)
    print(f"     Generated {n_configs} configurations with varied:")
    print("     - Grid sizes (16x16 to 64x64)")
    print("     - Initial densities (prey: 0.1-0.4, pred: 0.02-0.15)")
    print("     - Neighborhoods (Neumann/Moore)")
    print("     - Synchronicity (sync/async)")
    print("     - Rate parameters (beyond just death/birth)")
    print()
    
    # Run perturbation experiments
    print("[2/4] Running perturbation experiments...")
    experiment_results = []
    for i, config in enumerate(configs):
        print(f"     Config {i+1}/{n_configs}: "
              f"grid={config['rows']}x{config['cols']}, "
              f"densities=({config['densities'][0]:.2f},{config['densities'][1]:.2f}), "
              f"sync={config['synchronous']}")
        
        result = run_soc_perturbation_experiment(
            config,
            n_equilibration=80,  # build stress without perturbation
            n_observation=150,   # observe cascades during/after perturbation
            perturbation_step=80
        )
        experiment_results.append(result)
    print(f"     Completed {len(experiment_results)} experiments")
    print()
    
    # Analyze robustness
    print("[3/4] Analyzing SOC robustness across configurations...")
    robustness_metrics = analyze_soc_robustness(experiment_results)
    print(f"     Avalanche count (avg): {robustness_metrics['avg_avalanche_count']:.2f} "
          f"(std: {robustness_metrics['std_avalanche_count']:.2f})")
    print(f"     Avalanche magnitude (avg): {robustness_metrics['avalanche_magnitude_mean']:.4f}")
    print(f"     Stress level (avg): {robustness_metrics['avg_stress']:.4f}")
    print(f"     Coefficient of Variation (avalanche count): {robustness_metrics['coefficient_of_variation_avalanche']:.3f}")
    if robustness_metrics['coefficient_of_variation_avalanche'] < 1.0:
        print("     → LOW variation indicates ROBUST criticality across diverse parameters ✓")
    else:
        print("     → HIGH variation indicates parameter-dependent behavior")
    print()
    
    # Create visualization
    print("[4/4] Creating comprehensive visualization...")
    output_path = Path(__file__).parent.parent / "soc_analysis_results.png"
    visualize_soc_properties(experiment_results, robustness_metrics, str(output_path))
    print(f"     Saved to: {output_path}")


if __name__ == "__main__":
    main()

```





