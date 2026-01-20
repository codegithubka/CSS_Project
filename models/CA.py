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

	# Default colormap spec (string or sequence); resolved in `visualize` at runtime
	_default_cmap = "viridis"

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

		# One-time validation for per-cell evolved parameters (if present).
		# For performance, PP.update_* assumes that any per-cell parameter
		# arrays in `self.cell_params` have already been validated to:
		# - be numpy arrays of the same shape as `self.grid`, and
		# - have non-NaN entries exactly at the positions occupied by the
		#   corresponding species (e.g. `prey_death` non-NaNs at prey cells).
		# If this object defines `_evolve_info`, perform those checks once
		# before the run loop so updates can be fast.
		if hasattr(self, "_evolve_info") and isinstance(self._evolve_info, dict):
			for pname in self._evolve_info.keys():
				cp_arr = self.cell_params.get(pname)
				if isinstance(cp_arr, np.ndarray):
					if cp_arr.shape != self.grid.shape:
						raise ValueError(f"cell_params['{pname}'] must have shape equal to grid")
					# expected non-NaN positions correspond to species in grid
					species = 1 if pname.startswith("prey_") else 2
					nonnan = ~np.isnan(cp_arr)
					expected = (self.grid == species)
					if not np.array_equal(nonnan, expected):
						raise ValueError(f"cell_params['{pname}'] non-NaN entries must match positions of species {species} in the grid")

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
		show_cell_params: bool = False,
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
		# Determine whether to show any per-cell parameter arrays
		param_arrays = []
		param_keys = []
		if show_cell_params and isinstance(self.cell_params, dict):
			for k, v in self.cell_params.items():
				if isinstance(v, np.ndarray) and v.shape == self.grid.shape:
					param_arrays.append(v)
					param_keys.append(k)

		n_extra = len(param_arrays)
		if n_extra == 0:
			fig, ax = plt.subplots(figsize=figsize)
			im = ax.imshow(self.grid, cmap=cmap_obj, interpolation="nearest", vmin=0, vmax=self.n_species)
			ax.set_title("States: Iteration 0")
			axes = [ax]
			ims = [im]
		else:
			# create side-by-side plots: states + one per-param
			fig, axes = plt.subplots(1, 1 + n_extra, figsize=(figsize[0] * (1 + n_extra), figsize[1]))
			# normalize axes to a list for consistent indexing
			if not isinstance(axes, (list, tuple, np.ndarray)):
				axes = [axes]
			else:
				try:
					axes = list(np.array(axes).flatten())
				except Exception:
					axes = list(axes)
			# first axis: states
			im0 = axes[0].imshow(self.grid, cmap=cmap_obj, interpolation="nearest", vmin=0, vmax=self.n_species)
			axes[0].set_title("States: Iteration 0")
			ims = [im0]
			# other axes: parameter arrays
			for i, arr in enumerate(param_arrays, start=1):
				axp = axes[i]
				# use a continuous cmap for parameter values
				vmin = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else 0.0
				vmax = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
				imp = axp.imshow(arr, cmap=plt.get_cmap("viridis"), interpolation="nearest", vmin=vmin, vmax=vmax)
				axp.set_title(f"{param_keys[i-1]}: Iteration 0")
				ims.append(imp)

		plt.show(block=False)
		fig.canvas.draw()
		plt.pause(pause)

		# Store visualization state on the instance (only when visualization enabled)
		self._viz_on = True
		self._viz_interval = interval
		self._viz_fig = fig
		self._viz_axes = axes
		self._viz_ims = ims
		self._viz_param_keys = param_keys
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

		# update states grid
		if hasattr(self, "_viz_ims") and len(self._viz_ims) > 0:
			# first image is states
			self._viz_ims[0].set_data(self.grid)
			self._viz_axes[0].set_title(f"States: Iteration {iteration}")
			# update param images if any
			for i in range(1, len(self._viz_ims)):
				key = self._viz_param_keys[i-1]
				arr = self.cell_params.get(key)
				if arr is None:
					# blank out with NaNs
					arr = np.full(self.grid.shape, np.nan)
				self._viz_ims[i].set_data(arr)
				# adjust color scaling if possible
				try:
					vmin = float(np.nanmin(arr))
					vmax = float(np.nanmax(arr))
					if vmin < vmax:
						self._viz_ims[i].set_clim(vmin=vmin, vmax=vmax)
				except Exception:
					pass

		# draw/update
		self._viz_fig.canvas.draw_idle()
		plt.pause(self._viz_pause)


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
	chooses the update mode (default True -> synchronous updates).
	"""

	# Default colors: 0=empty black, 1=prey green, 2=predator red
	_default_cmap = ("black", "green", "red")

	def __init__(
		self,
		rows: int = 10,
		cols: int = 10,
		densities: Tuple[float, ...] = (0.2, 0.1),
		neighborhood: str = "moore",
		params: Dict[str, object] = None,
		cell_params: Dict[str, object] = None,
		seed: Optional[int] = None,
		synchronous: bool = True,
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

		# Information about which parameters are being evolved and their mutation specs
		# Maps parameter name -> dict with keys: 'sd', 'min', 'max'
		self._evolve_info: Dict[str, Dict[str, float]] = {}


	def evolve(self, param: str, sd: float = 0.05, min: float = 0.01, max: float = 0.99) -> None:
		"""Enable per-cell evolution for a given parameter.

		Creates a per-cell array in `self.cell_params[param]` with the same
		shape as the grid. Cells currently occupied by the relevant species are
		initialized to the global value in `self.params[param]`; other cells are
		set to NaN. Mutation metadata (sd, min, max) are stored in
		`self._evolve_info[param]`.

		Args:
		- param: one of the keys in `self.params` (e.g. 'prey_death')
		- sd: standard deviation for Gaussian mutations
		- min: minimum clipped value after mutation
		- max: maximum clipped value after mutation
		"""
		if param not in self.params:
			raise ValueError(f"Unknown parameter '{param}'")

		# determine target species for this parameter
		if param.startswith("prey_"):
			species = 1
		elif param.startswith("predator_"):
			species = 2
		else:
			raise ValueError("Parameter must start with 'prey_' or 'predator_' to evolve")

		# create per-cell float array with NaNs for non-relevant cells
		arr = np.full(self.grid.shape, np.nan, dtype=float)
		mask = (self.grid == species)
		arr[mask] = float(self.params[param])
		self.cell_params[param] = arr
		self._evolve_info[param] = {"sd": float(sd), "min": float(min), "max": float(max)}


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

		# Assumes any per-cell parameter arrays (in `self.cell_params`) are
		# well-formed and correspond to the current grid. `run()` performs a
		# one-time validation before the update loop; update methods do not
		# re-check for performance reasons.
		rows, cols = self.grid.shape
		grid_ref = self.grid.copy()

		# Sample deaths based on the reference grid and apply them first.
		# Death probabilities are sampled from the state at the start of
		# the iteration (grid_ref) so sampling remains consistent.
		rand_prey = self.generator.random(self.grid.shape)
		rand_pred = self.generator.random(self.grid.shape)

		# Use per-cell arrays directly if present, otherwise fall back to global scalar
		cp_prey = self.cell_params.get("prey_death")
		if isinstance(cp_prey, np.ndarray):
			prey_death_mask = (rand_prey < cp_prey)
		else:
			prey_death_mask = (grid_ref == 1) & (rand_prey < float(self.params["prey_death"]))

		cp_pred = self.cell_params.get("predator_death")
		if isinstance(cp_pred, np.ndarray):
			pred_death_mask = (rand_pred < cp_pred)
		else:
			pred_death_mask = (grid_ref == 2) & (rand_pred < float(self.params["predator_death"]))

		# Apply deaths to the current grid. We deliberately do this before
		# reproduction but keep using `grid_ref` for all reproduction checks
		# below to preserve the original (birth-before-death) semantics.
		self.grid[prey_death_mask] = 0
		self.grid[pred_death_mask] = 0

		# Clear per-cell parameters for individuals that died
		for pname in list(self._evolve_info.keys()):
			if pname.startswith("prey_"):
				arr = self.cell_params.get(pname)
				if isinstance(arr, np.ndarray) and arr.shape == self.grid.shape:
					arr[prey_death_mask] = np.nan
			elif pname.startswith("predator_"):
				arr = self.cell_params.get(pname)
				if isinstance(arr, np.ndarray) and arr.shape == self.grid.shape:
					arr[pred_death_mask] = np.nan

		# Precompute neighbor shifts and arrays for indexing (used by reproduction)
		if self.neighborhood == "neumann":
			shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
		else:
			shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
		dr_arr = np.array([s[0] for s in shifts], dtype=int)
		dc_arr = np.array([s[1] for s in shifts], dtype=int)
		n_shifts = len(shifts)

		def _process_reproduction(sources, birth_param_key, birth_prob, target_state_required, new_state_val):
			"""Handle reproduction attempts from `sources`.

			sources: (M,2) array of (r,c) positions in grid_ref
			birth_prob: scalar probability that a source attempts reproduction
			target_state_required: state value required in grid_ref at target
			new_state_val: state to write into self.grid for successful targets
			"""
			if sources.size == 0:
				return

			M = sources.shape[0]
			# Determine per-source birth probabilities (from cell_params if present)
			parent_grid = None
			if isinstance(birth_param_key, str) and birth_param_key in self.cell_params:
				parent_grid = self.cell_params.get(birth_param_key)
			if isinstance(parent_grid, np.ndarray):
				# parent_grid was already validated above to match species positions
				parent_probs = parent_grid[sources[:, 0], sources[:, 1]]
			else:
				parent_probs = np.full(M, float(birth_prob))

			# Which sources attempt reproduction
			attempt_mask = self.generator.random(M) < parent_probs
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

			src_valid = src[valid_mask]
		
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

			# Determine parent positions corresponding to chosen attempts
			parents = src_valid[chosen_indices]

			# Apply successful births to the main grid
			self.grid[chosen_rs, chosen_cs] = new_state_val

			# For any evolved parameters, inherit from parent with Gaussian noise
			for pname, meta in self._evolve_info.items():
				# check which species this parameter belongs to
				if pname.startswith("prey_"):
					target_species = 1
				elif pname.startswith("predator_"):
					target_species = 2
				else:
					continue
				# only assign if the new_state matches target_species
				if new_state_val != target_species:
					# ensure that parameters for other species are cleared at these cells
					arr = self.cell_params.get(pname)
					if isinstance(arr, np.ndarray) and arr.shape == self.grid.shape:
						arr[chosen_rs, chosen_cs] = np.nan
					continue

				# get parent values (fallback to global param if parent value missing)
				arr = self.cell_params.get(pname)
				if isinstance(arr, np.ndarray) and arr.shape == self.grid.shape:
					parent_vals = arr[parents[:, 0], parents[:, 1]]
					# where parent_vals is NaN, fall back to global
					mask_nan = np.isnan(parent_vals)
					if np.any(mask_nan):
						parent_vals[mask_nan] = float(self.params.get(pname, 0.0))
				else:
					parent_vals = np.full(parents.shape[0], float(self.params.get(pname, 0.0)))

				# mutate and clip
				sd = float(meta["sd"])
				mn = float(meta["min"])
				mx = float(meta["max"])
				mut = parent_vals + self.generator.normal(0.0, sd, size=parent_vals.shape)
				mut = np.clip(mut, mn, mx)
				# ensure array exists in cell_params
				if pname not in self.cell_params or not isinstance(self.cell_params[pname], np.ndarray):
					self.cell_params[pname] = np.full(self.grid.shape, np.nan)
				self.cell_params[pname][chosen_rs, chosen_cs] = mut

		# Prey reproduce into empty cells (target state 0 -> new state 1)
		prey_sources = np.argwhere(grid_ref == 1)
		_process_reproduction(prey_sources, self.params["prey_birth"], 0, 1)

		# Predators reproduce into prey cells (target state 1 -> new state 2)
		pred_sources = np.argwhere(grid_ref == 2)
		_process_reproduction(pred_sources, self.params["predator_birth"], 1, 2)

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

		# Assumes any per-cell parameter arrays (in `self.cell_params`) are
		# well-formed and correspond to the current grid. `run()` performs a
		# one-time validation before the update loop; update methods do not
		# re-check for performance reasons.

		# Sample and apply deaths first (based on the reference grid). Deaths
		# are sampled from `grid_ref` so statistics remain identical.
		rand_prey = self.generator.random(self.grid.shape)
		rand_pred = self.generator.random(self.grid.shape)

		# Determine death masks: use per-cell arrays when present, otherwise global scalars
		cp_pre = self.cell_params.get("prey_death")
		if isinstance(cp_pre, np.ndarray):
			prey_death_mask = (rand_prey < cp_pre)
		else:
			prey_death_mask = (grid_ref == 1) & (rand_prey < float(self.params["prey_death"]))

		cp_pred = self.cell_params.get("predator_death")
		if isinstance(cp_pred, np.ndarray):
			pred_death_mask = (rand_pred < cp_pred)
		else:
			pred_death_mask = (grid_ref == 2) & (rand_pred < float(self.params["predator_death"]))

		self.grid[prey_death_mask] = 0
		self.grid[pred_death_mask] = 0

		# clear per-cell params for dead individuals
		for pname in list(self._evolve_info.keys()):
			if pname.startswith("prey_"):
				arr = self.cell_params.get(pname)
				if isinstance(arr, np.ndarray) and arr.shape == self.grid.shape:
					arr[prey_death_mask] = np.nan
			elif pname.startswith("predator_"):
				arr = self.cell_params.get(pname)
				if isinstance(arr, np.ndarray) and arr.shape == self.grid.shape:
					arr[pred_death_mask] = np.nan

		# Precompute neighbor shifts
		if self.neighborhood == "neumann":
			shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
		else:
			shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

		# Get occupied cells from the original reference grid and shuffle.
		# We iterate over `grid_ref` so that sources can die and reproduce
		# in the same iteration, meaning we are order-agnostic.
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
						# per-parent birth prob
						bp = float(self.params["prey_birth"])
						cpb = self.cell_params.get("prey_birth")
						if isinstance(cpb, np.ndarray) and cpb.shape == self.grid.shape:
							pval = cpb[r, c]
							if np.isnan(pval):
								pval = bp
						else:
							pval = bp
						if self.generator.random() < float(pval):
							# birth: set new prey and inherit per-cell params (if any)
							self.grid[nr, nc] = 1
							# clear other-species params at target
							for pname, meta in self._evolve_info.items():
								if pname.startswith("predator_"):
									arr = self.cell_params.get(pname)
									if isinstance(arr, np.ndarray) and arr.shape == self.grid.shape:
										arr[nr, nc] = np.nan
							# assign evolved params for prey from parent
							for pname, meta in self._evolve_info.items():
								if not pname.startswith("prey_"):
									continue
								arr = self.cell_params.get(pname)
								if isinstance(arr, np.ndarray) and arr.shape == self.grid.shape:
									parent_val = arr[r, c]
									if np.isnan(parent_val):
										parent_val = float(self.params.get(pname, 0.0))
									sd = float(meta["sd"])
									mn = float(meta["min"])
									mx = float(meta["max"])
									child_val = parent_val + self.generator.normal(0.0, sd)
									child_val = float(np.clip(child_val, mn, mx))
									arr[nr, nc] = child_val
				elif state == 2:
					# Predator reproduces into prey neighbor (reference must be prey)
					if grid_ref[nr, nc] == 1:
						bp = float(self.params["predator_birth"])
						cpb = self.cell_params.get("predator_birth")
						if isinstance(cpb, np.ndarray) and cpb.shape == self.grid.shape:
							pval = cpb[r, c]
							if np.isnan(pval):
								pval = bp
						else:
							pval = bp
						if self.generator.random() < float(pval):
							# predator converts prey -> predator: assign and handle params
							self.grid[nr, nc] = 2
							# clear prey-specific params at the eaten cell
							for pname in list(self._evolve_info.keys()):
								if pname.startswith("prey_"):
									arr = self.cell_params.get(pname)
									if isinstance(arr, np.ndarray) and arr.shape == self.grid.shape:
										arr[nr, nc] = np.nan
							# assign predator params inherited from parent
							for pname, meta in self._evolve_info.items():
								if not pname.startswith("predator_"):
									continue
								arr = self.cell_params.get(pname)
								if isinstance(arr, np.ndarray) and arr.shape == self.grid.shape:
									parent_val = arr[r, c]
									if np.isnan(parent_val):
										parent_val = float(self.params.get(pname, 0.0))
									sd = float(meta["sd"])
									mn = float(meta["min"])
									mx = float(meta["max"])
									child_val = parent_val + self.generator.normal(0.0, sd)
									child_val = float(np.clip(child_val, mn, mx))
									arr[nr, nc] = child_val

	def update(self) -> None:
		"""Dispatch to synchronous or asynchronous update mode."""
		if self.synchronous:
			self.update_sync()
		else:
			self.update_async()