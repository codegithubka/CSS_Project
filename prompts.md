## Create base CA class
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

## Add update logic
Now the basic predator-prey update rules need to be defined. This is done by creating a predator-prey class inheriting the CA class and overriding the update function.

### prompt
Create a predator-prey class named PP which inherits the CA class. Create default values for the following parameters: prey_death, predator_death, reproduction, consumption. Make sure not to override the user-set values for these parameters. Also raise an error if there are any other user-defined parameters and check whether the set parameters are valid (between 0 and 1).

Next, override the update function with the following functionality. First, create a copy of the current grid to be used for all the rules. Then, use the inherited count_neighbors function. For each empty cell with at least one neighboring prey, populate it with a prey (state 1) with a random chance. Use generator.random() < 1 - (1 - birth)**prey_neighbors to simulate the possibility for each neighboring prey to reproduce into the cell. For each prey cell in the copied grid with at least one neighboring predator, replace it with a predator with a random chance. use the same logic for this. Afterwards, handle the prey and predator deaths stochastically based on the relevant parameter. Use the copied grid for this to ensure newly occupied cells dont instantly become empty.

### output
class PP(CA):
	"""Predator-Prey cellular automaton.

	States: 0 = empty, 1 = prey, 2 = predator

	Expected params keys (all values in [0,1]):
	- "prey_death": probability a prey dies each step
	- "predator_death": probability a predator dies each step
	- "reproduction": per-neighbor prey reproduction probability
	- "consumption": per-neighbor predation probability

	Defaults are provided for any missing keys and user-provided values
	are preserved. Any unknown keys in `params` will raise a ValueError.
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
		# initialize base CA
		super().__init__(rows, cols, densities, neighborhood, params, cell_params, seed)

		# Enforce predator-prey has exactly two species
		assert self.n_species == 2, "PP model requires exactly two species (prey=1, predator=2)"

		# Allowed parameter keys and defaults
		_allowed = {
			"prey_death": 0.02,
			"predator_death": 0.05,
			"reproduction": 0.2,
			"consumption": 0.5,
		}

		# Check for unknown user-specified keys (in the params dict provided by user)
		user_keys = set(self.params.keys())
		unknown = user_keys.difference(_allowed.keys())
		if len(unknown) > 0:
			raise ValueError(f"Unknown parameter keys for PP: {sorted(list(unknown))}")

		# Fill defaults for missing keys without overriding user-specified values
		for k, v in _allowed.items():
			if k not in self.params:
				self.params[k] = v

		# Validate parameter ranges
		for k in _allowed.keys():
			val = self.params[k]
			if not isinstance(val, (int, float)):
				raise TypeError(f"Parameter '{k}' must be numeric")
			if not (0.0 <= float(val) <= 1.0):
				raise ValueError(f"Parameter '{k}' must be between 0 and 1 (got {val})")

	def update(self) -> None:
		"""One update step for predator-prey dynamics.

		Uses a copy of the current grid to evaluate rules so newly changed
		cells do not immediately influence other rules in the same step.
		"""
		# copy of the grid to base all decisions on
		old = self.grid.copy()

		# neighbor counts for each species (index 0 -> prey, 1 -> predator)
		counts = self.count_neighbors()
		prey_neighbors = counts[0]
		pred_neighbors = counts[1] if self.n_species >= 2 else np.zeros_like(self.grid)

		rows, cols = self.grid.shape
		# Reproduction into empty cells from neighboring prey
		empty_mask = old == 0
		# probability that at least one neighboring prey reproduces into the cell
		birth_param = float(self.params["reproduction"])
		birth_prob = 1.0 - np.power(1.0 - birth_param, prey_neighbors)
		rand = self.generator.random(size=(rows, cols))
		birth_cells = empty_mask & (prey_neighbors > 0) & (rand < birth_prob)
		self.grid[birth_cells] = 1

		# Predation: prey replaced by predator due to neighboring predators
		prey_mask = old == 1
		cons_param = float(self.params["consumption"])
		cons_prob = 1.0 - np.power(1.0 - cons_param, pred_neighbors)
		rand = self.generator.random(size=(rows, cols))
		predation_cells = prey_mask & (pred_neighbors > 0) & (rand < cons_prob)
		self.grid[predation_cells] = 2

		# Deaths: use the copied `old` grid so newly-occupied cells are not killed immediately
		# Prey death
		prey_death_p = float(self.params["prey_death"])
		rand = self.generator.random(size=(rows, cols))
		prey_death_cells = (old == 1) & (rand < prey_death_p)
		self.grid[prey_death_cells] = 0

		# Predator death
		pred_death_p = float(self.params["predator_death"])
		rand = self.generator.random(size=(rows, cols))
		pred_death_cells = (old == 2) & (rand < pred_death_p)
		self.grid[pred_death_cells] = 0

### changes