from typing import Tuple, Dict, Any, Optional
import numpy as np

class CA:
	"""Cellular Automaton container.

	Attributes
	- `grid` (np.ndarray): 2D integer array with cell states (0 = empty).
	- `neighborhood` (str): either "neumann" or "moore".
	- `generator` (np.random.Generator): RNG used for all random operations.
	- `params` (dict): user-provided parameters.
	"""

	def __init__(
		self,
		rows: int,
		cols: int,
		densities: Tuple[float, ...],
		neighborhood: str = "neumann",
		params: Optional[Dict[str, Any]] = None,
		seed: Optional[int] = None,
	) -> None:
		"""Initialize the CA.

		Args:
		- `rows` (int): number of rows (> 0).
		- `cols` (int): number of columns (> 0).
		- `densities` (tuple of float): density fraction for each species (each >= 0).
		- `neighborhood` (str): either "neumann" or "moore".
		- `params` (dict, optional): additional parameters to store.
		- `seed` (int or None): seed for the RNG.

		Returns:
		- None

		Raises:
		- AssertionError if arguments are invalid.
		"""
		assert isinstance(rows, int) and rows > 0, "`rows` must be a positive int"
		assert isinstance(cols, int) and cols > 0, "`cols` must be a positive int"
		assert isinstance(densities, tuple) or isinstance(densities, list), "`densities` must be a tuple or list"
		densities = tuple(float(d) for d in densities)
		assert all(d >= 0 for d in densities), "all densities must be non-negative"
		assert sum(densities) <= 1.0 + 1e-12, "sum of densities must not exceed 1"
		assert neighborhood in ("neumann", "moore"), "neighborhood must be 'neumann' or 'moore'"

		self.params: Dict[str, Any] = dict(params) if params is not None else {}
		self.generator: np.random.Generator = np.random.default_rng(seed)
		self.neighborhood: str = neighborhood

		self.rows = rows
		self.cols = cols
		self.grid: np.ndarray = np.zeros((rows, cols), dtype=int)

		total_cells = rows * cols

		# Fill grid for each species (states 1..N) according to densities.
		# Use floor to avoid over-allocation; do not overwrite non-zero cells.
		for i, d in enumerate(densities):
			if d <= 0:
				continue
			desired = int(np.floor(total_cells * d))
			if desired <= 0:
				continue

			# available positions (flat indices)
			available = np.flatnonzero(self.grid.ravel() == 0)
			if len(available) == 0:
				break

			take = min(desired, len(available))
			chosen = self.generator.choice(available, size=take, replace=False)
			self.grid.ravel()[chosen] = i + 1

		# store number of species expected based on densities length
		self._n_species = len(densities)

	def count_neighbors(self) -> Tuple[np.ndarray, ...]:
		"""Count neighbors for each non-zero state.

		Uses periodic boundary conditions and the neighborhood specified in the instance.

		Returns:
		- Tuple of np.ndarray: each array has shape `(rows, cols)` and contains
		  the count of neighbors of state k (for k = 1..N) at each cell.
		"""
		neighbors = []

		if self.neighborhood == "moore":
			offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
		else:  # neumann
			offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

		for state in range(1, self._n_species + 1):
			mask = (self.grid == state).astype(int)
			count = np.zeros_like(self.grid, dtype=int)
			for dr, dc in offsets:
				shifted = np.roll(np.roll(mask, dr, axis=0), dc, axis=1)
				count += shifted
			neighbors.append(count)

		return tuple(neighbors)

	def update(self) -> None:
		"""Perform a single update of the CA.

		This method is intentionally left as a stub; implement specific rule
		logic by overriding or editing this method.

		Returns:
		- None
		"""
		pass

	def run(self, steps: int) -> None:
		"""Run the CA for a number of iterations.

		Args:
		- `steps` (int): number of iterations to execute (>= 0).

		Returns:
		- None
		"""
		assert isinstance(steps, int) and steps >= 0, "`steps` must be a non-negative int"
		for _ in range(steps):
			self.update()