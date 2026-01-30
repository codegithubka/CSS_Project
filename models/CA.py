"""
Cellular automaton base class.
"""

from typing import Tuple, Dict, Optional

import numpy as np
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.numba_optimized import PPKernel, set_numba_seed

# Module logger
logger = logging.getLogger(__name__)


class CA:
    """
    Base cellular automaton class for spatial simulations.

    This class provides a framework for multi-species cellular automata with 
    support for global parameters, per-cell evolving parameters, and 
    grid initialization based on density.

    Attributes
    ----------
    grid : np.ndarray
        2D numpy array containing integers in range [0, n_species].
    params : Dict[str, Any]
        Global parameters shared by all cells.
    cell_params : Dict[str, Any]
        Local per-cell parameters, typically stored as numpy arrays matching the grid shape.
    neighborhood : str
        The adjacency rule used ('neumann' or 'moore').
    generator : np.random.Generator
        The random number generator instance for reproducibility.
    species_names : Tuple[str, ...]
        Human-readable names for each species state.
    """

    # Default colormap spec (string or sequence); resolved in `visualize` at runtime
    _default_cmap = "viridis"

    # Read-only accessors for size/densities (protected attributes set in __init__)
    @property
    def rows(self) -> int:
        """int: Number of rows in the grid."""
        return getattr(self, "_rows")

    @property
    def cols(self) -> int:
        """int: Number of columns in the grid."""
        return getattr(self, "_cols")

    @property
    def densities(self) -> Tuple[float, ...]:
        """Tuple[float, ...]: Initial density fraction for each species."""
        return tuple(getattr(self, "_densities"))

    # make n_species protected with read-only property
    @property
    def n_species(self) -> int:
        """int: Number of distinct species states (excluding empty state 0)."""
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
        """
        Initialize the cellular automaton grid and configurations.

        Parameters
        ----------
        rows : int
            Number of rows in the grid (must be > 0).
        cols : int
            Number of columns in the grid (must be > 0).
        densities : Tuple[float, ...]
            Initial density for each species. Length defines `n_species`.
            Values must sum to <= 1.0.
        neighborhood : {'neumann', 'moore'}
            Type of neighborhood connectivity.
        params : Dict[str, Any]
            Initial global parameter values.
        cell_params : Dict[str, Any]
            Initial local per-cell parameters.
        seed : int, optional
            Seed for the random number generator.
        """
        assert isinstance(rows, int) and rows > 0, "rows must be positive int"
        assert isinstance(cols, int) and cols > 0, "cols must be positive int"
        assert (
            isinstance(densities, tuple) and len(densities) > 0
        ), "densities must be a non-empty tuple"
        for d in densities:
            assert (
                isinstance(d, (float, int)) and d >= 0
            ), "each density must be non-negative"
        total_density = float(sum(densities))
        assert total_density <= 1.0 + 1e-12, "sum of densities must not exceed 1"
        assert neighborhood in (
            "neumann",
            "moore",
        ), "neighborhood must be 'neumann' or 'moore'"

        self._n_species: int = len(densities)
        # store protected size/density attributes (read-only properties exposed)
        self._rows: int = rows
        self._cols: int = cols
        self._densities: Tuple[float, ...] = tuple(densities)
        self.params: Dict[str, object] = dict(params) if params is not None else {}
        self.cell_params: Dict[str, object] = (
            dict(cell_params) if cell_params is not None else {}
        )

        # per-parameter evolve metadata and evolution state
        # maps parameter name -> dict with keys 'sd','min','max','species'
        self._evolve_info: Dict[str, Dict[str, float]] = {}
        # when True, inheritance uses deterministic copy from parent (no mutation)
        self._evolution_stopped: bool = False

        # human-readable species names (useful for visualization). Default
        # generates generic names based on n_species; subclasses may override.
        self.species_names: Tuple[str, ...] = tuple(
            f"species{i+1}" for i in range(self._n_species)
        )
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
        """
        Validate core CA invariants and grid dimensions.

        Checks that the neighborhood is valid, the grid matches initialized dimensions,
        and that local parameter arrays match the grid shape.

        Raises
        ------
        ValueError
            If any structural invariant is violated.
        """
        if self.neighborhood not in ("neumann", "moore"):
            raise ValueError("neighborhood must be 'neumann' or 'moore'")

        expected_shape = (int(getattr(self, "_rows")), int(getattr(self, "_cols")))
        if self.grid.shape != expected_shape:
            raise ValueError(
                f"grid shape {self.grid.shape} does not match expected {expected_shape}"
            )

        # Ensure any array in cell_params matches grid shape
        for k, v in (self.cell_params or {}).items():
            if isinstance(v, np.ndarray) and v.shape != expected_shape:
                raise ValueError(f"cell_params['{k}'] must have shape equal to grid")

    def _infer_species_from_param_name(self, param_name: str) -> Optional[int]:
        """
        Infer the 1-based species index from a parameter name using `species_names`.

        This method checks if the given parameter name starts with any of the 
        defined species names followed by an underscore (e.g., 'prey_birth'). 
        It is used to automatically route global parameters to the correct 
        species' local parameter arrays.

        Parameters
        ----------
        param_name : str
            The name of the parameter to check.

        Returns
        -------
        Optional[int]
            The 1-based index of the species if a matching prefix is found; 
            otherwise, None.

        Notes
        -----
        The method expects `self.species_names` to be a collection of strings. 
        If `param_name` is not a string or no match is found, it returns None.
        """
        if not isinstance(param_name, str):
            return None
        for idx, name in enumerate(self.species_names or ()):  # type: ignore
            if isinstance(name, str) and param_name.startswith(f"{name}_"):
                return idx + 1
        return None

    def evolve(
        self,
        param: str,
        species: Optional[int] = None,
        sd: float = 0.05,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> None:
        """
        Enable per-cell evolution for a specific parameter on a given species.

        This method initializes a spatial parameter array (local parameter map) 
        for a global parameter. It allows individual cells to carry their own 
        values for that parameter, which can then mutate and evolve during 
        the simulation.

        Parameters
        ----------
        param : str
            The name of the global parameter to enable for evolution. 
            Must exist in `self.params`.
        species : int, optional
            The 1-based index of the species to which this parameter applies. 
            If None, the method attempts to infer the species from the 
            parameter name prefix.
        sd : float, default 0.05
            The standard deviation of the Gaussian mutation applied during 
            inheritance/reproduction.
        min_val : float, optional
            The minimum allowable value for the parameter (clamping). 
            Defaults to 0.01 if not provided.
        max_val : float, optional
            The maximum allowable value for the parameter (clamping). 
            Defaults to 0.99 if not provided.

        Raises
        ------
        ValueError
            If the parameter is not in `self.params`, the species cannot be 
            inferred, or the species index is out of bounds.

        Notes
        -----
        The local parameter is stored in `self.cell_params` as a 2D numpy 
        array initialized with the current global value for all cells of 
        the target species, and `NaN` elsewhere.
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
                raise ValueError(
                    "species must be provided or inferable from param name and species_names"
                )
        if not isinstance(species, int) or species <= 0 or species > self._n_species:
            raise ValueError("species must be an integer between 1 and n_species")

        arr = np.full(self.grid.shape, np.nan, dtype=float)
        mask = self.grid == int(species)
        arr[mask] = float(self.params[param])
        self.cell_params[param] = arr
        self._evolve_info[param] = {
            "sd": float(sd),
            "min": float(min_val),
            "max": float(max_val),
            "species": int(species),
        }

    def update(self) -> None:
        """
        Perform one update step of the cellular automaton.

        This is an abstract method that defines the transition rules of the 
        system. It must be implemented by concrete subclasses to specify 
        how cell states and parameters change over time based on their 
        current state and neighborhood.

        Raises
        ------
        NotImplementedError
            If called directly on the base class instead of an implementation.

        Returns
        -------
        None

        Notes
        -----
        In a typical implementation, this method handles the logic for 
        stochastic transitions, movement, or predator-prey interactions.
        """
        raise NotImplementedError(
            "Override update() in a subclass to define CA dynamics"
        )

    def run(
        self,
        steps: int,
        stop_evolution_at: Optional[int] = None,
        snapshot_iters: Optional[list] = None,
    ) -> None:
        """
        Execute the cellular automaton simulation for a specified number of steps.

        This method drives the simulation loop, calling `update()` at each 
        iteration. It manages visualization updates, directory creation for 
        data persistence, and handles the freezing of evolving parameters 
        at a specific time step.

        Parameters
        ----------
        steps : int
            The total number of iterations to run (must be non-negative).
        stop_evolution_at : int, optional
            The 1-based iteration index after which parameter mutation is 
            disabled. Useful for observing system stability after a period 
            of adaptation.
        snapshot_iters : List[int], optional
            A list of specific 1-based iteration indices at which to save 
            the current grid state to the results directory.

        Returns
        -------
        None

        Notes
        -----
        If snapshots are requested, a results directory is automatically created 
        using a timestamped subfolder (e.g., 'results/run-1700000000/'). 
        Visualization errors are logged but do not terminate the simulation.
        """
        assert (
            isinstance(steps, int) and steps >= 0
        ), "steps must be a non-negative integer"

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
                    logger.exception(
                        "Visualization update failed at iteration %d", i + 1
                    )

            # create snapshots if requested at this iteration
            if (i + 1) in snapshot_set:
                try:
                    # create snapshot folder if not present
                    if (
                        not hasattr(self, "_viz_snapshot_dir")
                        or self._viz_snapshot_dir is None
                    ):
                        import os, time

                        base = "results"
                        ts = int(time.time())
                        run_folder = f"run-{ts}"
                        full = os.path.join(base, run_folder)
                        os.makedirs(full, exist_ok=True)
                        self._viz_snapshot_dir = full
                    self._viz_save_snapshot(i + 1)
                except (OSError, PermissionError):
                    logger.exception(
                        "Failed to create or write snapshot at iteration %d", i + 1
                    )

            # stop evolution at specified time-step (disable further evolution)
            if stop_evolution_at is not None and (i + 1) == int(stop_evolution_at):
                # mark evolution as stopped; do not erase evolve metadata so
                # deterministic inheritance can still use parent values
                self._evolution_stopped = True


class PP(CA):
    """
    Predator-Prey Cellular Automaton model with Numba-accelerated kernels.

    This model simulates a stochastic predator-prey system where species 
    interact on a 2D grid. It supports evolving per-cell death rates, 
    periodic boundary conditions, and both random and directed hunting 
    behaviors.

    Parameters
    ----------
    rows : int, default 10
        Number of rows in the simulation grid.
    cols : int, default 10
        Number of columns in the simulation grid.
    densities : Tuple[float, ...], default (0.2, 0.1)
        Initial population densities for (prey, predator).
    neighborhood : {'moore', 'neumann'}, default 'moore'
        The neighborhood type for cell interactions.
    params : Dict[str, object], optional
        Global parameters: "prey_death", "predator_death", "prey_birth", 
        "predator_birth".
    cell_params : Dict[str, object], optional
        Initial local parameter maps (2D arrays).
    seed : int, optional
        Random seed for reproducibility.
    synchronous : bool, default True
        If True, updates the entire grid at once. If False, updates 
        cells asynchronously.
    directed_hunting : bool, default False
        If True, predators selectively hunt prey rather than choosing 
        neighbors at random.

    Attributes
    ----------
    species_names : Tuple[str, ...]
        Labels for the species ('prey', 'predator').
    synchronous : bool
        Current update mode.
    directed_hunting : bool
        Current hunting strategy logic.
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
        directed_hunting: bool = False,  # New directed hunting option
    ) -> None:
        """
        Initialize the Predator-Prey CA with validated parameters and kernels.
        """
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
        super().__init__(
            rows, cols, densities, neighborhood, merged_params, cell_params, seed
        )

        self.synchronous: bool = bool(synchronous)
        self.directed_hunting: bool = bool(directed_hunting)

        # set human-friendly species names for PP
        self.species_names = ("prey", "predator")

        if seed is not None:
            # This sets the seed for all @njit functions globally
            set_numba_seed(seed)

        self._kernel = PPKernel(
            rows, cols, neighborhood, directed_hunting=directed_hunting
        )

    # Remove PP-specific evolve wrapper; use CA.evolve with optional species

    def validate(self) -> None:
        """
        Validate Predator-Prey specific invariants and spatial parameter arrays.

        Extends the base CA validation to ensure that numerical parameters are 
        within the [0, 1] probability range and that evolved parameter maps 
        (e.g., prey_death) correctly align with the species locations.

        Raises
        ------
        ValueError
            If grid shapes, parameter ranges, or species masks are inconsistent.
        TypeError
            If parameters are non-numeric.
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
                    raise ValueError(
                        f"cell_params['{pname}'] missing species metadata and could not infer from name"
                    )
            nonnan = ~np.isnan(arr)
            expected = self.grid == species
            if not np.array_equal(nonnan, expected):
                raise ValueError(
                    f"cell_params['{pname}'] non-NaN entries must match positions of species {species}"
                )
            # values must be within configured range where not NaN
            mn = float(meta.get("min", 0.0))
            mx = float(meta.get("max", 1.0))
            vals = arr[~np.isnan(arr)]
            if vals.size > 0:
                if np.any(vals < mn) or np.any(vals > mx):
                    raise ValueError(
                        f"cell_params['{pname}'] contains values outside [{mn}, {mx}]"
                    )

    def update_async(self) -> None:
        """
        Execute an asynchronous update using the optimized Numba kernel.

        This method retrieves the evolved parameter maps and delegates the 
        stochastic transitions to the `PPKernel`. Asynchronous updates 
        typically handle cell-by-cell logic where changes can be 
        immediately visible to neighbors.
        """
        # Get the evolved prey death map
        # Fallback to a full array of the global param if it doesn't exist yet
        p_death_arr = self.cell_params.get("prey_death")
        if p_death_arr is None:
            p_death_arr = np.full(
                self.grid.shape, self.params["prey_death"], dtype=np.float64
            )

        meta = self._evolve_info.get(
            "prey_death", {"sd": 0.05, "min": 0.001, "max": 0.1}
        )

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
        """
        Dispatch the simulation step based on the configured update mode.
        """
        if self.synchronous:
            self.update_sync()
        else:
            self.update_async()
