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

---


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

### Parameter Sweep and PP Class Analysis

2. Create a skeletal version of a .py script that will be subimtted into Snellius for parameter analysis. The purpose of this script should be to identify power law distribution with the consideration of finite size scaling, the hydra effect, and suitable parameter configurtaions for time series analysis for model evolution. Compare the non-evo to the evo model.

3. Create a config class adjustable depending on the CPU budget. We want to run a prey_birth vs. predator_death parameter sweep (2D), quantify the hydra effect using the derivative, search for the critical point (power law relartionship paramameter), quantify evolution sensitivity and analyze finite grid size scaling. Include script options for cost optimal runs as well. Make sure to have a summary of collected data stored for reference and future usage.

