"""Cellular automaton base class.

Defines a CA class with initialization, neighbor counting, update (to override),
and run loop. Uses a numpy Generator for all randomness and supports
Neumann and Moore neighborhoods with periodic boundaries.
"""
from typing import Tuple, Dict, Optional

import numpy as np
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.numba_optimized import PPKernel, set_numba_seed
from models.cluster_analysis import ClusterAnalyzer

# Module logger
logger = logging.getLogger(__name__)

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

	# Read-only accessors for size/densities (protected attributes set in __init__)
	@property
	def rows(self) -> int:
		return getattr(self, "_rows")

	@property
	def cols(self) -> int:
		return getattr(self, "_cols")

	@property
	def densities(self) -> Tuple[float, ...]:
		return tuple(getattr(self, "_densities"))

	# make n_species protected with read-only property
	@property
	def n_species(self) -> int:
		return int(getattr(self, "_n_species"))

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

		self._n_species: int = len(densities)
		# store protected size/density attributes (read-only properties exposed)
		self._rows: int = rows
		self._cols: int = cols
		self._densities: Tuple[float, ...] = tuple(densities)
		self.params: Dict[str, object] = dict(params) if params is not None else {}
		self.cell_params: Dict[str, object] = dict(cell_params) if cell_params is not None else {}

		# per-parameter evolve metadata and evolution state
		# maps parameter name -> dict with keys 'sd','min','max','species'
		self._evolve_info: Dict[str, Dict[str, float]] = {}
		# when True, inheritance uses deterministic copy from parent (no mutation)
		self._evolution_stopped: bool = False

		# human-readable species names (useful for visualization). Default
		# generates generic names based on n_species; subclasses may override.
		self.species_names: Tuple[str, ...] = tuple(f"species{i+1}" for i in range(self._n_species))
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

	def validate(self) -> None:
		"""Validate core CA invariants.

		Checks that `neighborhood` is valid, that `self.grid` has the
	texpected shape `(rows, cols)`, and that any numpy arrays in
	`self.cell_params` have matching shapes. Raises `ValueError` on
	validation failure.
		"""
		if self.neighborhood not in ("neumann", "moore"):
			raise ValueError("neighborhood must be 'neumann' or 'moore'")

		expected_shape = (int(getattr(self, "_rows")), int(getattr(self, "_cols")))
		if self.grid.shape != expected_shape:
			raise ValueError(f"grid shape {self.grid.shape} does not match expected {expected_shape}")

		# Ensure any array in cell_params matches grid shape
		for k, v in (self.cell_params or {}).items():
			if isinstance(v, np.ndarray) and v.shape != expected_shape:
				raise ValueError(f"cell_params['{k}'] must have shape equal to grid")

	def _infer_species_from_param_name(self, param_name: str) -> Optional[int]:
		"""Infer species index (1-based) from a parameter name using `species_names`.

		Returns the 1-based species index if a matching prefix is found,
		otherwise `None`.
		"""
		if not isinstance(param_name, str):
			return None
		for idx, name in enumerate(self.species_names or ()):  # type: ignore
			if isinstance(name, str) and param_name.startswith(f"{name}_"):
				return idx + 1
		return None

	def evolve(self, param: str, species: Optional[int] = None, sd: float = 0.05, min_val: Optional[float] = None, max_val: Optional[float] = None) -> None:
		"""Enable per-cell evolution for `param` on `species`.

		If `species` is None, attempt to infer the species using
		`_infer_species_from_param_name(param)` which matches against
		`self.species_names`. This keeps `CA` free of domain-specific
		(predator/prey) logic while preserving backward compatibility when
		subclasses set `species_names` (e.g. `('prey','predator')`).
		"""
		if min_val is None:
			min_val = 0.01
		if max_val is None:
			max_val = 0.99
		if param not in self.params:
			raise ValueError(f"Unknown parameter '{param}'")
		if species is None:
			species = self._infer_species_from_param_name(param)
			if species is None:
				raise ValueError("species must be provided or inferable from param name and species_names")
		if not isinstance(species, int) or species <= 0 or species > self._n_species:
			raise ValueError("species must be an integer between 1 and n_species")

		arr = np.full(self.grid.shape, np.nan, dtype=float)
		mask = (self.grid == int(species))
		arr[mask] = float(self.params[param])
		self.cell_params[param] = arr
		self._evolve_info[param] = {"sd": float(sd), "min": float(min_val), "max": float(max_val), "species": int(species)}

	def update(self) -> None:
		"""Perform one update step.

		This base implementation must be overridden by subclasses. It raises
		NotImplementedError to indicate it should be provided by concrete
		models that inherit from `CA`.

		Returns: None
		"""
		raise NotImplementedError("Override update() in a subclass to define CA dynamics")

	def run(self, steps: int, stop_evolution_at: Optional[int] = None, snapshot_iters: Optional[list] = None) -> None:
		"""Run the CA for a number of steps.

		Args:
		- steps (int): number of iterations to run (must be non-negative).

		Returns: None
		"""
		assert isinstance(steps, int) and steps >= 0, "steps must be a non-negative integer"

		# normalize snapshot iteration list
		snapshot_set = set(snapshot_iters) if snapshot_iters is not None else set()

		for i in range(steps):
			self.update()
			# Update visualization if enabled every `interval` iterations
			if getattr(self, "_viz_on", False):
				# iteration number is 1-based for display
				try:
					self._viz_update(i + 1)
				except Exception:
					# Log visualization errors but don't stop the simulation
					logger.exception("Visualization update failed at iteration %d", i + 1)

			# create snapshots if requested at this iteration
			if (i + 1) in snapshot_set:
				try:
					# create snapshot folder if not present
					if not hasattr(self, "_viz_snapshot_dir") or self._viz_snapshot_dir is None:
						import os, time

						base = "results"
						ts = int(time.time())
						run_folder = f"run-{ts}"
						full = os.path.join(base, run_folder)
						os.makedirs(full, exist_ok=True)
						self._viz_snapshot_dir = full
					self._viz_save_snapshot(i + 1)
				except (OSError, PermissionError):
					logger.exception("Failed to create or write snapshot at iteration %d", i + 1)

			# stop evolution at specified time-step (disable further evolution)
			if stop_evolution_at is not None and (i + 1) == int(stop_evolution_at):
				# mark evolution as stopped; do not erase evolve metadata so
				# deterministic inheritance can still use parent values
				self._evolution_stopped = True

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
		directed_hunting: bool = False, # New directed hunting option
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
		self.directed_hunting: bool = bool(directed_hunting)
    
		# set human-friendly species names for PP
		self.species_names = ("prey", "predator")
  
		if seed is not None:
			# This sets the seed for all @njit functions globally
			set_numba_seed(seed)
   
		self._kernel = PPKernel(rows, cols, neighborhood, directed_hunting=directed_hunting)


	# Remove PP-specific evolve wrapper; use CA.evolve with optional species

	def validate(self) -> None:
		"""Validate PP-specific invariants in addition to base CA checks.

		Checks:
		- each global parameter is numeric and in [0,1]
		- per-cell evolved parameter arrays (in `_evolve_info`) have non-NaN
		  positions matching the species grid and contain values within the
		  configured min/max range (or are NaN).
		"""
		super().validate()

		# Validate global params
		for k, v in (self.params or {}).items():
			if not isinstance(v, (int, float)):
				raise TypeError(f"Parameter '{k}' must be numeric")
			if not (0.0 <= float(v) <= 1.0):
				raise ValueError(f"Parameter '{k}' must be between 0 and 1")

		# Validate per-cell evolve arrays
		for pname, meta in (self._evolve_info or {}).items():
			arr = self.cell_params.get(pname)
			if not isinstance(arr, np.ndarray):
				# absent or non-array per-cell params are allowed; skip
				continue
			# shape already checked in super().validate(), but be explicit
			if arr.shape != self.grid.shape:
				raise ValueError(f"cell_params['{pname}'] must match grid shape")
			# expected non-NaN positions correspond to species stored in metadata
			species = None
			if isinstance(meta, dict) and "species" in meta:
				species = int(meta.get("species"))
			else:
				# try to infer species from parameter name using species_names
				species = self._infer_species_from_param_name(pname)
				if species is None:
					raise ValueError(f"cell_params['{pname}'] missing species metadata and could not infer from name")
			nonnan = ~np.isnan(arr)
			expected = (self.grid == species)
			if not np.array_equal(nonnan, expected):
				raise ValueError(f"cell_params['{pname}'] non-NaN entries must match positions of species {species}")
			# values must be within configured range where not NaN
			mn = float(meta.get("min", 0.0))
			mx = float(meta.get("max", 1.0))
			vals = arr[~np.isnan(arr)]
			if vals.size > 0:
				if np.any(vals < mn) or np.any(vals > mx):
					raise ValueError(f"cell_params['{pname}'] contains values outside [{mn}, {mx}]")

	def update_async(self) -> None:
		# Get the evolved prey death map
		# Fallback to a full array of the global param if it doesn't exist yet
		p_death_arr = self.cell_params.get("prey_death")
		if p_death_arr is None:
			p_death_arr = np.full(self.grid.shape, self.params["prey_death"], dtype=np.float64)

		meta = self._evolve_info.get("prey_death", {"sd": 0.05, "min": 0.001, "max": 0.1})

		# Call the optimized kernel (uses pre-allocated buffers)
		self._kernel.update(
			self.grid,
			p_death_arr,
			float(self.params["prey_birth"]),
			float(self.params["prey_death"]),
			float(self.params["predator_birth"]),
			float(self.params["predator_death"]),
			float(meta["sd"]),
			float(meta["min"]),
			float(meta["max"]),
			self._evolution_stopped,
		)
    
	def update(self) -> None:
		"""Dispatch to synchronous or asynchronous update mode."""
		if self.synchronous:
			self.update_sync()
		else:
			self.update_async()