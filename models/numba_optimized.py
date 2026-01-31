#!/usr/bin/env python3
"""
Numba-Optimized Kernels
=======================

This module provides Numba-accelerated kernels for the predator-prey
cellular automaton, including update kernels and spatial analysis functions.

Classes
-------
PPKernel
    Wrapper for predator-prey update kernels with pre-allocated buffers.

Cluster Analysis
----------------
```python
measure_cluster_sizes_fast # Fast cluster size measurement (sizes only).
detect_clusters_fast # Full cluster detection with labels.
get_cluster_stats_fast # Comprehensive cluster statistics.
```

Pair Correlation Functions
--------------------------
```python
compute_pcf_periodic_fast # PCF for two position sets with periodic boundaries.
compute_all_pcfs_fast #Compute prey-prey, pred-pred, and prey-pred PCFs.
```

Utilities
---------
```python
set_numba_seed # Seed Numba's internal RNG.
warmup_numba_kernels # Pre-compile kernels to avoid first-run latency.
```

Example
-------
```python
from models.numba_optimized import (
    PPKernel,
    get_cluster_stats_fast,
    compute_all_pcfs_fast,
)

# Cluster analysis
stats = get_cluster_stats_fast(grid, species=1)
print(f"Largest cluster: {stats['largest']}")

# PCF computation
pcfs = compute_all_pcfs_fast(grid, max_distance=20.0)
prey_prey_dist, prey_prey_gr, _ = pcfs['prey_prey']
```
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
    Seed Numba's internal random number generator from within a JIT context.

    This function ensures that Numba's independent random number generator
    is synchronized with the provided seed, enabling reproducibility for
    jit-compiled functions that use NumPy's random operations.

    Parameters
    ----------
    seed : int
        The integer value used to initialize the random number generator.

    Returns
    -------
    None

    Notes
    -----
    Because Numba maintains its own internal state for random number
    generation, calling `np.random.seed()` in standard Python code will not
    affect jit-compiled functions. This helper must be called to bridge
    that gap.
    """
    np.random.seed(seed)


# ============================================================================
# PREDATOR-PREY KERNELS
# ============================================================================


@njit(cache=True)
def _pp_async_kernel_random(
    grid: np.ndarray,
    prey_death_arr: np.ndarray,
    p_birth_val: float,
    p_death_val: float,
    pred_birth_val: float,
    pred_death_val: float,
    dr_arr: np.ndarray,
    dc_arr: np.ndarray,
    evolve_sd: float,
    evolve_min: float,
    evolve_max: float,
    evolution_stopped: bool,
    occupied_buffer: np.ndarray,
) -> np.ndarray:
    """
    Asynchronous predator-prey update kernel with random neighbor selection.

    This Numba-accelerated kernel performs an asynchronous update of the
    simulation grid. It identifies all occupied cells, shuffles them to
    ensure unbiased processing, and applies stochastic rules for prey
    mortality, prey reproduction (with optional parameter evolution),
    predator mortality, and predation.

    Parameters
    ----------
    grid : np.ndarray
        2D integer array representing the simulation grid (0: Empty, 1: Prey, 2: Predator).
    prey_death_arr : np.ndarray
        2D float array storing the individual prey death rates for evolution tracking.
    p_birth_val : float
        Base probability of prey reproduction into an adjacent empty cell.
    p_death_val : float
        Base probability of prey death (though individual rates in `prey_death_arr` are used).
    pred_birth_val : float
        Probability of a predator reproducing after consuming prey.
    pred_death_val : float
        Probability of a predator dying.
    dr_arr : np.ndarray
        Array of row offsets defining the neighborhood.
    dc_arr : np.ndarray
        Array of column offsets defining the neighborhood.
    evolve_sd : float
        Standard deviation of the mutation applied to the prey death rate during reproduction.
    evolve_min : float
        Lower bound for the evolved prey death rate.
    evolve_max : float
        Upper bound for the evolved prey death rate.
    evolution_stopped : bool
        If True, offspring inherit the parent's death rate without mutation.
    occupied_buffer : np.ndarray
        Pre-allocated 2D array used to store and shuffle coordinates of occupied cells.

    Returns
    -------
    grid : np.ndarray
        The updated simulation grid.

    Notes
    -----
    The kernel uses periodic boundary conditions. The Fisher-Yates shuffle on
    `occupied_buffer` ensures that the asynchronous updates do not introduce
    directional bias.
    """
    rows, cols = grid.shape
    n_shifts = len(dr_arr)

    # Collect occupied cells
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                occupied_buffer[count, 0] = r
                occupied_buffer[count, 1] = c
                count += 1

    # Fisher-Yates shuffle
    for i in range(count - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        occupied_buffer[i, 0], occupied_buffer[j, 0] = (
            occupied_buffer[j, 0],
            occupied_buffer[i, 0],
        )
        occupied_buffer[i, 1], occupied_buffer[j, 1] = (
            occupied_buffer[j, 1],
            occupied_buffer[i, 1],
        )

    # Process each occupied cell
    for i in range(count):
        r = occupied_buffer[i, 0]
        c = occupied_buffer[i, 1]

        state = grid[r, c]
        if state == 0:
            continue

        # Random neighbor selection
        nbi = np.random.randint(0, n_shifts)
        nr = (r + dr_arr[nbi]) % rows
        nc = (c + dc_arr[nbi]) % cols

        if state == 1:  # PREY
            if np.random.random() < prey_death_arr[r, c]:
                grid[r, c] = 0
                prey_death_arr[r, c] = np.nan
            elif grid[nr, nc] == 0:
                if np.random.random() < p_birth_val:
                    grid[nr, nc] = 1
                    parent_val = prey_death_arr[r, c]
                    if not evolution_stopped:
                        child_val = parent_val + np.random.normal(0, evolve_sd)
                        if child_val < evolve_min:
                            child_val = evolve_min
                        if child_val > evolve_max:
                            child_val = evolve_max
                        prey_death_arr[nr, nc] = child_val
                    else:
                        prey_death_arr[nr, nc] = parent_val

        elif state == 2:  # PREDATOR
            if np.random.random() < pred_death_val:
                grid[r, c] = 0
            elif grid[nr, nc] == 1:
                if np.random.random() < pred_birth_val:
                    grid[nr, nc] = 2
                    prey_death_arr[nr, nc] = np.nan

    return grid


@njit(cache=True)
def _pp_async_kernel_directed(
    grid: np.ndarray,
    prey_death_arr: np.ndarray,
    p_birth_val: float,
    p_death_val: float,
    pred_birth_val: float,
    pred_death_val: float,
    dr_arr: np.ndarray,
    dc_arr: np.ndarray,
    evolve_sd: float,
    evolve_min: float,
    evolve_max: float,
    evolution_stopped: bool,
    occupied_buffer: np.ndarray,
) -> np.ndarray:
    """
    Asynchronous predator-prey update kernel with directed behavior.

    This kernel implements "intelligent" species behavior: prey actively search
    for empty spaces to reproduce, and predators actively search for nearby
    prey to hunt. A two-pass approach is used to stochastically select a
    valid target from the neighborhood without heap allocation.

    Parameters
    ----------
    grid : np.ndarray
        2D integer array representing the simulation grid (0: Empty, 1: Prey, 2: Predator).
    prey_death_arr : np.ndarray
        2D float array storing individual prey mortality rates for evolution.
    p_birth_val : float
        Probability of prey reproduction attempt.
    p_death_val : float
        Base probability of prey mortality.
    pred_birth_val : float
        Probability of a predator reproduction attempt (hunting success).
    pred_death_val : float
        Probability of predator mortality.
    dr_arr : np.ndarray
        Row offsets defining the spatial neighborhood (e.g., Moore or von Neumann).
    dc_arr : np.ndarray
        Column offsets defining the spatial neighborhood.
    evolve_sd : float
        Standard deviation for mutations in prey death rates.
    evolve_min : float
        Minimum allowable value for evolved prey death rates.
    evolve_max : float
        Maximum allowable value for evolved prey death rates.
    evolution_stopped : bool
        If True, prevents mutation during prey reproduction.
    occupied_buffer : np.ndarray
        Pre-allocated array for storing and shuffling active cell coordinates.

    Returns
    -------
    grid : np.ndarray
        The updated simulation grid.

    Notes
    -----
    The directed behavior significantly changes the system dynamics compared to
    random neighbor selection, often leading to different critical thresholds
    and spatial patterning. Periodic boundary conditions are applied.
    """
    rows, cols = grid.shape
    n_shifts = len(dr_arr)

    # Collect occupied cells
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                occupied_buffer[count, 0] = r
                occupied_buffer[count, 1] = c
                count += 1

    # Fisher-Yates shuffle
    for i in range(count - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        occupied_buffer[i, 0], occupied_buffer[j, 0] = (
            occupied_buffer[j, 0],
            occupied_buffer[i, 0],
        )
        occupied_buffer[i, 1], occupied_buffer[j, 1] = (
            occupied_buffer[j, 1],
            occupied_buffer[i, 1],
        )

    # Process each occupied cell
    for i in range(count):
        r = occupied_buffer[i, 0]
        c = occupied_buffer[i, 1]

        state = grid[r, c]
        if state == 0:
            continue

        if state == 1:  # PREY - directed reproduction into empty cells
            # Check for death first
            if np.random.random() < prey_death_arr[r, c]:
                grid[r, c] = 0
                prey_death_arr[r, c] = np.nan
                continue

            # Attempt reproduction with directed selection
            if np.random.random() < p_birth_val:
                # Pass 1: Count empty neighbors
                empty_count = 0
                for k in range(n_shifts):
                    check_r = (r + dr_arr[k]) % rows
                    check_c = (c + dc_arr[k]) % cols
                    if grid[check_r, check_c] == 0:
                        empty_count += 1

                # Pass 2: Select random empty neighbor
                if empty_count > 0:
                    target_idx = np.random.randint(0, empty_count)
                    found = 0
                    nr, nc = r, c  # Initialize (will be overwritten)
                    for k in range(n_shifts):
                        check_r = (r + dr_arr[k]) % rows
                        check_c = (c + dc_arr[k]) % cols
                        if grid[check_r, check_c] == 0:
                            if found == target_idx:
                                nr, nc = check_r, check_c
                                break
                            found += 1

                    # Reproduce into selected empty cell
                    grid[nr, nc] = 1
                    parent_val = prey_death_arr[r, c]
                    if not evolution_stopped:
                        child_val = parent_val + np.random.normal(0, evolve_sd)
                        if child_val < evolve_min:
                            child_val = evolve_min
                        if child_val > evolve_max:
                            child_val = evolve_max
                        prey_death_arr[nr, nc] = child_val
                    else:
                        prey_death_arr[nr, nc] = parent_val

        elif state == 2:  # PREDATOR - directed hunting
            # Check for death first
            if np.random.random() < pred_death_val:
                grid[r, c] = 0
                continue

            # Attempt hunting with directed selection
            if np.random.random() < pred_birth_val:
                # Pass 1: Count prey neighbors
                prey_count = 0
                for k in range(n_shifts):
                    check_r = (r + dr_arr[k]) % rows
                    check_c = (c + dc_arr[k]) % cols
                    if grid[check_r, check_c] == 1:
                        prey_count += 1

                # Pass 2: Select random prey neighbor
                if prey_count > 0:
                    target_idx = np.random.randint(0, prey_count)
                    found = 0
                    nr, nc = r, c  # Initialize (will be overwritten)
                    for k in range(n_shifts):
                        check_r = (r + dr_arr[k]) % rows
                        check_c = (c + dc_arr[k]) % cols
                        if grid[check_r, check_c] == 1:
                            if found == target_idx:
                                nr, nc = check_r, check_c
                                break
                            found += 1

                    # Hunt: prey cell becomes predator
                    grid[nr, nc] = 2
                    prey_death_arr[nr, nc] = np.nan

    return grid


class PPKernel:
    """
    Wrapper for predator-prey kernel with pre-allocated buffers.

    This class manages the spatial configuration and memory buffers required
    for the Numba-accelerated update kernels. By pre-allocating the
    `occupied_buffer`, it avoids expensive memory allocations during the
    simulation loop.

    Parameters
    ----------
    rows : int
        Number of rows in the simulation grid.
    cols : int
        Number of columns in the simulation grid.
    neighborhood : {'moore', 'von_neumann'}, optional
        The neighborhood type determining adjacent cells. 'moore' includes
        diagonals (8 neighbors), 'von_neumann' does not (4 neighbors).
        Default is 'moore'.
    directed_hunting : bool, optional
        If True, uses the directed behavior kernel where species search for
        targets. If False, uses random neighbor selection. Default is False.

    Attributes
    ----------
    rows : int
        Grid row count.
    cols : int
        Grid column count.
    directed_hunting : bool
        Toggle for intelligent behavior logic.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        neighborhood: str = "moore",
        directed_hunting: bool = False,
    ):
        self.rows = rows
        self.cols = cols
        self.directed_hunting = directed_hunting
        self._occupied_buffer = np.empty((rows * cols, 2), dtype=np.int32)

        if neighborhood == "moore":
            self._dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
            self._dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        else:  # von Neumann
            self._dr = np.array([-1, 1, 0, 0], dtype=np.int32)
            self._dc = np.array([0, 0, -1, 1], dtype=np.int32)

    def update(
        self,
        grid: np.ndarray,
        prey_death_arr: np.ndarray,
        prey_birth: float,
        prey_death: float,
        pred_birth: float,
        pred_death: float,
        evolve_sd: float = 0.1,
        evolve_min: float = 0.001,
        evolve_max: float = 0.1,
        evolution_stopped: bool = True,
    ) -> np.ndarray:
        """
        Execute a single asynchronous update step using the configured kernel.

        Parameters
        ----------
        grid : np.ndarray
            The current 2D simulation grid.
        prey_death_arr : np.ndarray
            2D array of individual prey mortality rates.
        prey_birth : float
            Prey reproduction probability.
        prey_death : float
            Base prey mortality probability.
        pred_birth : float
            Predator reproduction (hunting success) probability.
        pred_death : float
            Predator mortality probability.
        evolve_sd : float, optional
            Mutation standard deviation (default 0.1).
        evolve_min : float, optional
            Minimum evolved death rate (default 0.001).
        evolve_max : float, optional
            Maximum evolved death rate (default 0.1).
        evolution_stopped : bool, optional
            Whether to disable mutation during this step (default True).

        Returns
        -------
        np.ndarray
            The updated grid after one full asynchronous pass.
        """
        if self.directed_hunting:
            return _pp_async_kernel_directed(
                grid,
                prey_death_arr,
                prey_birth,
                prey_death,
                pred_birth,
                pred_death,
                self._dr,
                self._dc,
                evolve_sd,
                evolve_min,
                evolve_max,
                evolution_stopped,
                self._occupied_buffer,
            )
        else:
            return _pp_async_kernel_random(
                grid,
                prey_death_arr,
                prey_birth,
                prey_death,
                pred_birth,
                pred_death,
                self._dr,
                self._dc,
                evolve_sd,
                evolve_min,
                evolve_max,
                evolution_stopped,
                self._occupied_buffer,
            )


# ============================================================================
# CLUSTER DETECTION (ENHANCED)
# ============================================================================


@njit(cache=True)
def _flood_fill(
    grid: np.ndarray,
    visited: np.ndarray,
    start_r: int,
    start_c: int,
    target: int,
    rows: int,
    cols: int,
    moore: bool,
) -> int:
    """
    Perform a stack-based flood fill to measure the size of a connected cluster.

    This Numba-accelerated function identifies all contiguous cells of a
    specific target value starting from a given coordinate. It supports
    both Moore and von Neumann neighborhoods and implements periodic
    boundary conditions (toroidal topology).

    Parameters
    ----------
    grid : np.ndarray
        2D integer array representing the simulation environment.
    visited : np.ndarray
        2D boolean array tracked across calls to avoid re-processing cells.
    start_r : int
        Starting row index for the flood fill.
    start_c : int
        Starting column index for the flood fill.
    target : int
        The cell value (e.g., 1 for Prey, 2 for Predator) to include in the cluster.
    rows : int
        Total number of rows in the grid.
    cols : int
        Total number of columns in the grid.
    moore : bool
        If True, use a Moore neighborhood (8 neighbors). If False, use a
        von Neumann neighborhood (4 neighbors).

    Returns
    -------
    size : int
        The total number of connected cells belonging to the cluster.

    Notes
    -----
    The function uses a manual stack implementation to avoid recursion limit
    issues and is optimized for use within JIT-compiled loops.
    """
    max_stack = rows * cols
    stack_r = np.empty(max_stack, dtype=np.int32)
    stack_c = np.empty(max_stack, dtype=np.int32)
    stack_ptr = 0

    stack_r[stack_ptr] = start_r
    stack_c[stack_ptr] = start_c
    stack_ptr += 1
    visited[start_r, start_c] = True

    size = 0

    if moore:
        dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
        dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        n_neighbors = 8
    else:
        dr = np.array([-1, 1, 0, 0], dtype=np.int32)
        dc = np.array([0, 0, -1, 1], dtype=np.int32)
        n_neighbors = 4

    while stack_ptr > 0:
        stack_ptr -= 1
        r = stack_r[stack_ptr]
        c = stack_c[stack_ptr]
        size += 1

        for k in range(n_neighbors):
            nr = (r + dr[k]) % rows
            nc = (c + dc[k]) % cols

            if not visited[nr, nc] and grid[nr, nc] == target:
                visited[nr, nc] = True
                stack_r[stack_ptr] = nr
                stack_c[stack_ptr] = nc
                stack_ptr += 1

    return size


@njit(cache=True)
def _measure_clusters(grid: np.ndarray, species: int, moore: bool = True) -> np.ndarray:
    """
    Identify and measure the sizes of all connected clusters for a specific species.

    This function scans the entire grid and initiates a flood-fill algorithm
    whenever an unvisited cell of the target species is encountered. It
    returns an array containing the size (cell count) of each identified cluster.

    Parameters
    ----------
    grid : np.ndarray
        2D integer array representing the simulation environment.
    species : int
        The target species identifier (e.g., 1 for Prey, 2 for Predator).
    moore : bool, optional
        Determines the connectivity logic. If True, uses the Moore neighborhood
        (8 neighbors); if False, uses the von Neumann neighborhood (4 neighbors).
        Default is True.

    Returns
    -------
    cluster_sizes : np.ndarray
        A 1D array of integers where each element represents the size of
        one connected cluster.

    Notes
    -----
    This function is Numba-optimized and utilizes an internal `visited` mask
    to ensure each cell is processed only once, maintaining $O(N)$
    complexity relative to the number of cells.
    """
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=np.bool_)

    max_clusters = rows * cols
    sizes = np.empty(max_clusters, dtype=np.int32)
    n_clusters = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == species and not visited[r, c]:
                size = _flood_fill(grid, visited, r, c, species, rows, cols, moore)
                sizes[n_clusters] = size
                n_clusters += 1

    return sizes[:n_clusters]


@njit(cache=True)
def _detect_clusters_numba(
    grid: np.ndarray,
    species: int,
    moore: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full cluster detection returning labels and sizes (Numba-accelerated).

    Returns:
        labels: 2D int32 array where each cell contains its cluster ID (0 = non-target)
        sizes: 1D int32 array of cluster sizes (index i = size of cluster i+1)
    """
    rows, cols = grid.shape
    labels = np.zeros((rows, cols), dtype=np.int32)

    if moore:
        dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
        dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        n_neighbors = 8
    else:
        dr = np.array([-1, 1, 0, 0], dtype=np.int32)
        dc = np.array([0, 0, -1, 1], dtype=np.int32)
        n_neighbors = 4

    max_clusters = rows * cols
    sizes = np.empty(max_clusters, dtype=np.int32)
    n_clusters = 0
    current_label = 1

    max_stack = rows * cols
    stack_r = np.empty(max_stack, dtype=np.int32)
    stack_c = np.empty(max_stack, dtype=np.int32)

    for start_r in range(rows):
        for start_c in range(cols):
            if grid[start_r, start_c] != species or labels[start_r, start_c] != 0:
                continue

            stack_ptr = 0
            stack_r[stack_ptr] = start_r
            stack_c[stack_ptr] = start_c
            stack_ptr += 1
            labels[start_r, start_c] = current_label
            size = 0

            while stack_ptr > 0:
                stack_ptr -= 1
                r = stack_r[stack_ptr]
                c = stack_c[stack_ptr]
                size += 1

                for k in range(n_neighbors):
                    nr = (r + dr[k]) % rows
                    nc = (c + dc[k]) % cols

                    if grid[nr, nc] == species and labels[nr, nc] == 0:
                        labels[nr, nc] = current_label
                        stack_r[stack_ptr] = nr
                        stack_c[stack_ptr] = nc
                        stack_ptr += 1

            sizes[n_clusters] = size
            n_clusters += 1
            current_label += 1

    return labels, sizes[:n_clusters]


# ============================================================================
# PUBLIC API - CLUSTER DETECTION
# ============================================================================


def measure_cluster_sizes_fast(
    grid: np.ndarray,
    species: int,
    neighborhood: str = "moore",
) -> np.ndarray:
    """
    Measure cluster sizes for a specific species using Numba-accelerated flood fill.

    This function provides a high-performance interface for calculating cluster
    size statistics without the overhead of generating a full label map. It is
    optimized for large-scale simulation analysis where only distribution
    metrics (e.g., mean size, max size) are required.

    Parameters
    ----------
    grid : np.ndarray
        A 2D array representing the simulation environment.
    species : int
        The target species identifier (e.g., 1 for Prey, 2 for Predator).
    neighborhood : {'moore', 'neumann'}, optional
        The connectivity rule. 'moore' uses 8-way connectivity (including diagonals);
        'neumann' uses 4-way connectivity. Default is 'moore'.

    Returns
    -------
    cluster_sizes : np.ndarray
        A 1D array of integers, where each element is the cell count of an
        identified cluster.

    Notes
    -----
    The input grid is cast to `int32` to ensure compatibility with the
    underlying JIT-compiled `_measure_clusters` kernel.

    Examples
    --------
    >>> sizes = measure_cluster_sizes_fast(grid, species=1, neighborhood='moore')
    >>> if sizes.size > 0:
    ...     print(f"Largest cluster: {sizes.max()}")
    """
    grid_int = np.asarray(grid, dtype=np.int32)
    moore = neighborhood == "moore"
    return _measure_clusters(grid_int, np.int32(species), moore)


def detect_clusters_fast(
    grid: np.ndarray,
    species: int,
    neighborhood: str = "moore",
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Perform full cluster detection with labels using Numba acceleration.

    This function returns a label array for spatial analysis and a dictionary
    of cluster sizes. It is significantly faster than standard Python or
    SciPy equivalents for large simulation grids.

    Parameters
    ----------
    grid : np.ndarray
        A 2D array representing the simulation environment.
    species : int
        The target species identifier (e.g., 1 for Prey, 2 for Predator).
    neighborhood : {'moore', 'neumann'}, optional
        The connectivity rule. 'moore' uses 8-way connectivity; 'neumann'
        uses 4-way connectivity. Default is 'moore'.

    Returns
    -------
    labels : np.ndarray
        A 2D int32 array where each cell contains its unique cluster ID.
        Cells not belonging to the target species are 0.
    sizes : dict
        A dictionary mapping cluster IDs to their respective cell counts.

    Notes
    -----
    The underlying Numba kernel uses a stack-based flood fill to avoid
    recursion limits and handles periodic boundary conditions.

    Examples
    --------
    >>> labels, sizes = detect_clusters_fast(grid, species=1)
    >>> if sizes:
    ...     largest_id = max(sizes, key=sizes.get)
    ...     print(f"Cluster {largest_id} size: {sizes[largest_id]}")
    """
    grid_int = np.asarray(grid, dtype=np.int32)
    moore = neighborhood == "moore"
    labels, sizes_arr = _detect_clusters_numba(grid_int, np.int32(species), moore)
    sizes_dict = {i + 1: int(sizes_arr[i]) for i in range(len(sizes_arr))}
    return labels, sizes_dict


def get_cluster_stats_fast(
    grid: np.ndarray,
    species: int,
    neighborhood: str = "moore",
) -> Dict:
    """
    Compute comprehensive cluster statistics for a species using Numba acceleration.

    This function integrates cluster detection and labeling to provide a
    full suite of spatial metrics. It calculates the cluster size distribution
    and the largest cluster fraction, which often serves as an order
    parameter in percolation theory and Phase 1-3 analyses.

    Parameters
    ----------
    grid : np.ndarray
        A 2D array representing the simulation environment.
    species : int
        The target species identifier (e.g., 1 for Prey, 2 for Predator).
    neighborhood : {'moore', 'neumann'}, optional
        The connectivity rule. 'moore' uses 8-way connectivity; 'neumann'
        uses 4-way connectivity. Default is 'moore'.

    Returns
    -------
    stats : dict
        A dictionary containing:
        - 'n_clusters': Total count of isolated clusters.
        - 'sizes': Sorted array (descending) of all cluster sizes.
        - 'largest': Size of the single largest cluster.
        - 'largest_fraction': Size of the largest cluster divided by
          the total population of the species.
        - 'mean_size': Average size of all clusters.
        - 'size_distribution': Frequency mapping of {size: count}.
        - 'labels': 2D array of unique cluster IDs.
        - 'size_dict': Mapping of {label_id: size}.

    Examples
    --------
    >>> stats = get_cluster_stats_fast(grid, species=1)
    >>> print(f"Found {stats['n_clusters']} prey clusters.")
    >>> print(f"Order parameter: {stats['largest_fraction']:.3f}")
    """
    labels, size_dict = detect_clusters_fast(grid, species, neighborhood)

    if len(size_dict) == 0:
        return {
            "n_clusters": 0,
            "sizes": np.array([], dtype=np.int32),
            "largest": 0,
            "largest_fraction": 0.0,
            "mean_size": 0.0,
            "size_distribution": {},
            "labels": labels,
            "size_dict": size_dict,
        }

    sizes = np.array(list(size_dict.values()), dtype=np.int32)
    sizes_sorted = np.sort(sizes)[::-1]
    total_pop = int(np.sum(sizes))
    largest = int(sizes_sorted[0])

    size_dist = {}
    for s in sizes:
        s_int = int(s)
        size_dist[s_int] = size_dist.get(s_int, 0) + 1

    return {
        "n_clusters": len(size_dict),
        "sizes": sizes_sorted,
        "largest": largest,
        "largest_fraction": float(largest) / total_pop if total_pop > 0 else 0.0,
        "mean_size": float(np.mean(sizes)),
        "size_distribution": size_dist,
        "labels": labels,
        "size_dict": size_dict,
    }


# ============================================================================
# PCF COMPUTATION (Cell-list accelerated)
# ============================================================================


@njit(cache=True)
def _build_cell_list(
    positions: np.ndarray,
    n_cells: int,
    L_row: float,
    L_col: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Build a cell list for spatial hashing to accelerate neighbor lookups.

    This Numba-optimized function partitions a set of coordinates into a
    grid of cells. It uses a three-pass approach to calculate cell occupancy,
    compute starting offsets for each cell in a flat index array, and finally
    populate that array with position indices.

    Parameters
    ----------
    positions : np.ndarray
        An (N, 2) float array of coordinates within the simulation domain.
    n_cells : int
        The number of cells along one dimension of the square grid.
    L_row : float
        The total height (row extent) of the simulation domain.
    L_col : float
        The total width (column extent) of the simulation domain.

    Returns
    -------
    indices : np.ndarray
        A 1D array of original position indices, reordered so that indices
        belonging to the same cell are contiguous.
    offsets : np.ndarray
        A 2D array where `offsets[r, c]` is the starting index in the
        `indices` array for cell (r, c).
    cell_counts : np.ndarray
        A 2D array where `cell_counts[r, c]` is the number of points
        contained in cell (r, c).
    cell_size_r : float
        The calculated height of an individual cell.
    cell_size_c : float
        The calculated width of an individual cell.

    Notes
    -----
    This implementation assumes periodic boundary conditions via the
    modulo operator during coordinate-to-cell mapping. It is designed to
    eliminate heap allocations within the main simulation loop by using
    Numba's efficient array handling.
    """
    n_pos = len(positions)
    cell_size_r = L_row / n_cells
    cell_size_c = L_col / n_cells

    cell_counts = np.zeros((n_cells, n_cells), dtype=np.int32)
    for i in range(n_pos):
        cr = int(positions[i, 0] / cell_size_r) % n_cells
        cc = int(positions[i, 1] / cell_size_c) % n_cells
        cell_counts[cr, cc] += 1

    offsets = np.zeros((n_cells, n_cells), dtype=np.int32)
    running = 0
    for cr in range(n_cells):
        for cc in range(n_cells):
            offsets[cr, cc] = running
            running += cell_counts[cr, cc]

    indices = np.empty(n_pos, dtype=np.int32)
    fill_counts = np.zeros((n_cells, n_cells), dtype=np.int32)
    for i in range(n_pos):
        cr = int(positions[i, 0] / cell_size_r) % n_cells
        cc = int(positions[i, 1] / cell_size_c) % n_cells
        idx = offsets[cr, cc] + fill_counts[cr, cc]
        indices[idx] = i
        fill_counts[cr, cc] += 1

    return indices, offsets, cell_counts, cell_size_r, cell_size_c


@njit(cache=True)
def _periodic_dist_sq(
    r1: float,
    c1: float,
    r2: float,
    c2: float,
    L_row: float,
    L_col: float,
) -> float:
    """
    Calculate the squared Euclidean distance between two points with periodic boundary conditions.

    This Numba-optimized function accounts for toroidal topology by finding the
    shortest path between coordinates across the grid edges. Using the squared
    distance avoids the computational expense of a square root operation,
    making it ideal for high-frequency spatial queries.

    Parameters
    ----------
    r1 : float
        Row coordinate of the first point.
    c1 : float
        Column coordinate of the first point.
    r2 : float
        Row coordinate of the second point.
    c2 : float
        Column coordinate of the second point.
    L_row : float
        Total height (row extent) of the periodic domain.
    L_col : float
        Total width (column extent) of the periodic domain.

    Returns
    -------
    dist_sq : float
        The squared shortest distance between the two points.

    Notes
    -----
    The function applies the minimum image convention, ensuring that the
    distance never exceeds half the domain length in any dimension.
    """
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    if dr > L_row * 0.5:
        dr = L_row - dr
    if dc > L_col * 0.5:
        dc = L_col - dc
    return dr * dr + dc * dc


@njit(parallel=True, cache=True)
def _pcf_cell_list(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    indices_j: np.ndarray,
    offsets_j: np.ndarray,
    counts_j: np.ndarray,
    cell_size_r: float,
    cell_size_c: float,
    L_row: float,
    L_col: float,
    max_distance: float,
    n_bins: int,
    self_correlation: bool,
    n_cells: int,
) -> np.ndarray:
    """
    Compute a Pair Correlation Function (PCF) histogram using spatial cell lists.

    This Numba-accelerated parallel kernel calculates distances between two sets
    of points (pos_i and pos_j). It uses a cell list (spatial hashing) to
    restrict distance calculations to neighboring cells within the maximum
    specified distance, significantly improving performance from $O(N^2)$
    to $O(N)$.

    Parameters
    ----------
    pos_i : np.ndarray
        (N, 2) float array of coordinates for the primary species.
    pos_j : np.ndarray
        (M, 2) float array of coordinates for the secondary species.
    indices_j : np.ndarray
        Flattened indices of pos_j sorted by cell, produced by `_build_cell_list`.
    offsets_j : np.ndarray
        2D array of starting offsets for each cell in `indices_j`.
    counts_j : np.ndarray
        2D array of particle counts within each cell for species J.
    cell_size_r : float
        Height of a single spatial cell.
    cell_size_c : float
        Width of a single spatial cell.
    L_row : float
        Total height of the periodic domain.
    L_col : float
        Total width of the periodic domain.
    max_distance : float
        Maximum radial distance (r) to consider for the correlation.
    n_bins : int
        Number of bins in the distance histogram.
    self_correlation : bool
        If True, assumes species I and J are the same and avoids double-counting
        or self-interaction.
    n_cells : int
        Number of cells per dimension in the spatial hash grid.

    Returns
    -------
    hist : np.ndarray
        A 1D array of length `n_bins` containing the counts of pairs found
        at each radial distance.

    Notes
    -----
    The kernel uses `prange` for parallel execution across points in `pos_i`.
    Local histograms are used per thread to prevent race conditions during
    reduction. Periodic boundary conditions are handled via `_periodic_dist_sq`.
    """
    n_i = len(pos_i)
    bin_width = max_distance / n_bins
    max_dist_sq = max_distance * max_distance
    cells_to_check = int(np.ceil(max_distance / min(cell_size_r, cell_size_c))) + 1

    hist = np.zeros(n_bins, dtype=np.int64)

    for i in prange(n_i):
        local_hist = np.zeros(n_bins, dtype=np.int64)
        r1, c1 = pos_i[i, 0], pos_i[i, 1]

        cell_r = int(r1 / cell_size_r) % n_cells
        cell_c = int(c1 / cell_size_c) % n_cells

        for dcr in range(-cells_to_check, cells_to_check + 1):
            for dcc in range(-cells_to_check, cells_to_check + 1):
                ncr = (cell_r + dcr) % n_cells
                ncc = (cell_c + dcc) % n_cells

                start = offsets_j[ncr, ncc]
                end = start + counts_j[ncr, ncc]

                for idx in range(start, end):
                    j = indices_j[idx]

                    if self_correlation and j <= i:
                        continue

                    r2, c2 = pos_j[j, 0], pos_j[j, 1]
                    d_sq = _periodic_dist_sq(r1, c1, r2, c2, L_row, L_col)

                    if 0 < d_sq < max_dist_sq:
                        d = np.sqrt(d_sq)
                        bin_idx = int(d / bin_width)
                        if bin_idx >= n_bins:
                            bin_idx = n_bins - 1
                        local_hist[bin_idx] += 1

        for b in range(n_bins):
            hist[b] += local_hist[b]

    if self_correlation:
        for b in range(n_bins):
            hist[b] *= 2

    return hist


def compute_pcf_periodic_fast(
    positions_i: np.ndarray,
    positions_j: np.ndarray,
    grid_shape: Tuple[int, int],
    max_distance: float,
    n_bins: int = 50,
    self_correlation: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute the Pair Correlation Function (PCF) using cell-list acceleration.

    This high-level function coordinates the spatial hashing and histogram
    calculation to determine the $g(r)$ function. It normalizes the resulting
    histogram by the expected number of pairs in an ideal gas of the same
    density, accounting for the toroidal area of each radial bin.

    Parameters
    ----------
    positions_i : np.ndarray
        (N, 2) array of coordinates for species I.
    positions_j : np.ndarray
        (M, 2) array of coordinates for species J.
    grid_shape : tuple of int
        The (rows, cols) dimensions of the simulation grid.
    max_distance : float
        The maximum radius to calculate correlations for.
    n_bins : int, optional
        Number of bins for the radial distribution (default 50).
    self_correlation : bool, optional
        Set to True if computing the correlation of a species with itself
        to avoid self-counting (default False).

    Returns
    -------
    bin_centers : np.ndarray
        The central radial distance for each histogram bin.
    pcf : np.ndarray
        The normalized $g(r)$ values. A value of 1.0 indicates no spatial
        correlation; > 1.0 indicates clustering; < 1.0 indicates repulsion.
    total_pairs : int
        The total count of pairs found within the `max_distance`.

    Notes
    -----
    The function dynamically determines the optimal number of cells for the
    spatial hash based on the `max_distance` and grid dimensions to maintain
    linear time complexity.
    """
    rows, cols = grid_shape
    L_row, L_col = float(rows), float(cols)
    area = L_row * L_col

    bin_width = max_distance / n_bins
    bin_centers = np.linspace(bin_width / 2, max_distance - bin_width / 2, n_bins)

    if len(positions_i) == 0 or len(positions_j) == 0:
        return bin_centers, np.ones(n_bins), 0

    n_cells = max(4, int(min(rows, cols) / max_distance))

    pos_i = np.ascontiguousarray(positions_i, dtype=np.float64)
    pos_j = np.ascontiguousarray(positions_j, dtype=np.float64)

    indices_j, offsets_j, counts_j, cell_size_r, cell_size_c = _build_cell_list(
        pos_j, n_cells, L_row, L_col
    )

    hist = _pcf_cell_list(
        pos_i,
        pos_j,
        indices_j,
        offsets_j,
        counts_j,
        cell_size_r,
        cell_size_c,
        L_row,
        L_col,
        max_distance,
        n_bins,
        self_correlation,
        n_cells,
    )

    n_i, n_j = len(positions_i), len(positions_j)
    if self_correlation:
        density_product = n_i * (n_i - 1) / (area * area)
    else:
        density_product = n_i * n_j / (area * area)

    expected = np.zeros(n_bins)
    for i in range(n_bins):
        r = bin_centers[i]
        annulus_area = 2 * np.pi * r * bin_width
        expected[i] = density_product * annulus_area * area

    pcf = np.ones(n_bins)
    mask = expected > 1.0
    pcf[mask] = hist[mask] / expected[mask]

    return bin_centers, pcf, int(np.sum(hist))


def compute_all_pcfs_fast(
    grid: np.ndarray,
    max_distance: Optional[float] = None,
    n_bins: int = 50,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """
    Compute all three species Pair Correlation Functions (PCFs) using cell-list acceleration.

    This function calculates the spatial auto-correlations (Prey-Prey,
    Predator-Predator) and the cross-correlation (Prey-Predator) for a given
    simulation grid. It identifies particle positions and leverages
    Numba-accelerated cell lists to handle the computations efficiently.

    Parameters
    ----------
    grid : np.ndarray
        2D integer array where 1 represents prey and 2 represents predators.
    max_distance : float, optional
        The maximum radial distance for the correlation. Defaults to 1/4
        of the minimum grid dimension if not provided.
    n_bins : int, optional
        Number of distance bins for the histogram. Default is 50.

    Returns
    -------
    results : dict
        A dictionary with keys 'prey_prey', 'pred_pred', and 'prey_pred'.
        Each value is a tuple containing:
        - bin_centers (np.ndarray): Radial distances.
        - pcf_values (np.ndarray): Normalized g(r) values.
        - pair_count (int): Total number of pairs found.

    Notes
    -----
    The PCF provides insight into the spatial organization of the system.
    g(r) > 1 at short distances indicates aggregation (clustering),
    while g(r) < 1 indicates exclusion or repulsion.
    """
    rows, cols = grid.shape
    if max_distance is None:
        max_distance = min(rows, cols) / 4.0

    prey_pos = np.argwhere(grid == 1)
    pred_pos = np.argwhere(grid == 2)

    results = {}

    dist, pcf, n = compute_pcf_periodic_fast(
        prey_pos,
        prey_pos,
        (rows, cols),
        max_distance,
        n_bins,
        self_correlation=True,
    )
    results["prey_prey"] = (dist, pcf, n)

    dist, pcf, n = compute_pcf_periodic_fast(
        pred_pos,
        pred_pos,
        (rows, cols),
        max_distance,
        n_bins,
        self_correlation=True,
    )
    results["pred_pred"] = (dist, pcf, n)

    dist, pcf, n = compute_pcf_periodic_fast(
        prey_pos,
        pred_pos,
        (rows, cols),
        max_distance,
        n_bins,
        self_correlation=False,
    )
    results["prey_pred"] = (dist, pcf, n)

    return results


# ============================================================================
# WARMUP & BENCHMARKS
# ============================================================================


def warmup_numba_kernels(grid_size: int = 100, directed_hunting: bool = False):
    """
    Pre-compile all Numba-accelerated kernels to avoid first-run latency.

    This function executes a single step of the simulation and each analysis
    routine on a dummy grid. Because Numba uses Just-In-Time (JIT) compilation,
    the first call to a decorated function incurs a compilation overhead.
    Running this warmup ensures that subsequent experimental runs are timed
    accurately and perform at full speed.

    Parameters
    ----------
    grid_size : int, optional
        The side length of the dummy grid used for warmup (default 100).
    directed_hunting : bool, optional
        If True, also warms up the directed behavior update kernel (default False).

    Returns
    -------
    None

    Notes
    -----
    This function checks for `NUMBA_AVAILABLE` before execution. It warms up
    the `PPKernel` (random and optionally directed), as well as the
    spatial analysis functions (`compute_all_pcfs_fast`, `detect_clusters_fast`, etc.).
    """
    if not NUMBA_AVAILABLE:
        return

    set_numba_seed(0)

    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    grid[::3, ::3] = 1
    grid[::5, ::5] = 2

    prey_death_arr = np.full((grid_size, grid_size), 0.05, dtype=np.float64)
    prey_death_arr[grid != 1] = np.nan

    # Always warmup random kernel
    kernel_random = PPKernel(grid_size, grid_size, directed_hunting=False)
    kernel_random.update(grid.copy(), prey_death_arr.copy(), 0.2, 0.05, 0.2, 0.1)

    # Warmup directed kernel if requested
    if directed_hunting:
        kernel_directed = PPKernel(grid_size, grid_size, directed_hunting=True)
        kernel_directed.update(grid.copy(), prey_death_arr.copy(), 0.2, 0.05, 0.2, 0.1)

    # Warmup analysis functions
    _ = compute_all_pcfs_fast(grid, max_distance=20.0, n_bins=20)
    _ = measure_cluster_sizes_fast(grid, 1)
    _ = detect_clusters_fast(grid, 1)
    _ = get_cluster_stats_fast(grid, 1)


def benchmark_kernels(grid_size: int = 100, n_runs: int = 20):
    """
    Benchmark the execution performance of random vs. directed update kernels.

    This utility measures the average time per simulation step for both the
    stochastic (random neighbor) and heuristic (directed hunting/reproduction)
    update strategies. It accounts for the computational overhead introduced
    by the "intelligent" search logic used in directed mode.

    Parameters
    ----------
    grid_size : int, optional
        The side length of the square simulation grid (default 100).
    n_runs : int, optional
        The number of iterations to perform for averaging performance (default 20).

    Returns
    -------
    t_random : float
        Average time per step for the random kernel in milliseconds.
    t_directed : float
        Average time per step for the directed kernel in milliseconds.

    Notes
    -----
    The function ensures a fair comparison by:
    1. Using a fixed seed for reproducible initial grid states.
    2. Warming up Numba kernels before timing to exclude JIT compilation latency.
    3. Copying the grid and death arrays for each iteration to maintain
       consistent population densities throughout the benchmark.
    """
    import time

    print("=" * 60)
    print(f"KERNEL BENCHMARK ({grid_size}x{grid_size}, {n_runs} runs)")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("=" * 60)

    np.random.seed(42)
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    n_prey = int(grid_size * grid_size * 0.30)
    n_pred = int(grid_size * grid_size * 0.15)
    positions = np.random.permutation(grid_size * grid_size)
    for pos in positions[:n_prey]:
        grid[pos // grid_size, pos % grid_size] = 1
    for pos in positions[n_prey : n_prey + n_pred]:
        grid[pos // grid_size, pos % grid_size] = 2

    prey_death_arr = np.full((grid_size, grid_size), 0.05, dtype=np.float64)
    prey_death_arr[grid != 1] = np.nan

    print(f"Initial: {np.sum(grid == 1)} prey, {np.sum(grid == 2)} predators")

    # Warmup both kernels
    warmup_numba_kernels(grid_size, directed_hunting=True)

    # Benchmark random kernel
    kernel_random = PPKernel(grid_size, grid_size, directed_hunting=False)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        test_grid = grid.copy()
        test_arr = prey_death_arr.copy()
        kernel_random.update(test_grid, test_arr, 0.2, 0.05, 0.2, 0.1)
    t_random = (time.perf_counter() - t0) / n_runs * 1000

    # Benchmark directed kernel
    kernel_directed = PPKernel(grid_size, grid_size, directed_hunting=True)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        test_grid = grid.copy()
        test_arr = prey_death_arr.copy()
        kernel_directed.update(test_grid, test_arr, 0.2, 0.05, 0.2, 0.1)
    t_directed = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\nRandom kernel:   {t_random:.2f} ms/step")
    print(f"Directed kernel: {t_directed:.2f} ms/step")
    print(
        f"Overhead:        {t_directed - t_random:.2f} ms (+{100*(t_directed/t_random - 1):.1f}%)"
    )

    return t_random, t_directed


def benchmark_cluster_detection(grid_size: int = 100, n_runs: int = 20):
    """
    Benchmark the performance of different cluster detection and analysis routines.

    This function evaluates three levels of spatial analysis:
    1. Size measurement only (fastest, no label map).
    2. Full detection (returns label map and size dictionary).
    3. Comprehensive statistics (calculates distributions, means, and order parameters).

    Parameters
    ----------
    grid_size : int, optional
        Side length of the square grid for benchmarking (default 100).
    n_runs : int, optional
        Number of iterations to average for performance results (default 20).

    Returns
    -------
    stats : dict
        The result dictionary from the final comprehensive statistics run.

    Notes
    -----
    The benchmark uses a fixed prey density of 30% to ensure a representative
    distribution of clusters. It pre-warms the Numba kernels to ensure that
    the measurements reflect execution speed rather than compilation time.
    """
    import time

    print("=" * 60)
    print(f"CLUSTER DETECTION BENCHMARK ({grid_size}x{grid_size})")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("=" * 60)

    np.random.seed(42)
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    n_prey = int(grid_size * grid_size * 0.30)
    positions = np.random.permutation(grid_size * grid_size)[:n_prey]
    for pos in positions:
        grid[pos // grid_size, pos % grid_size] = 1

    print(f"Prey cells: {np.sum(grid == 1)}")

    # Warmup
    _ = measure_cluster_sizes_fast(grid, 1)
    _ = detect_clusters_fast(grid, 1)
    _ = get_cluster_stats_fast(grid, 1)

    # Benchmark sizes only
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sizes = measure_cluster_sizes_fast(grid, 1)
    t_sizes = (time.perf_counter() - t0) / n_runs * 1000
    print(f"\nmeasure_cluster_sizes_fast: {t_sizes:.2f} ms  ({len(sizes)} clusters)")

    # Benchmark full detection
    t0 = time.perf_counter()
    for _ in range(n_runs):
        labels, size_dict = detect_clusters_fast(grid, 1)
    t_detect = (time.perf_counter() - t0) / n_runs * 1000
    print(f"detect_clusters_fast:       {t_detect:.2f} ms  ({len(size_dict)} clusters)")

    # Benchmark full stats
    t0 = time.perf_counter()
    for _ in range(n_runs):
        stats = get_cluster_stats_fast(grid, 1)
    t_stats = (time.perf_counter() - t0) / n_runs * 1000
    print(f"get_cluster_stats_fast:     {t_stats:.2f} ms")

    print(
        f"\nOverhead for labels: {t_detect - t_sizes:.2f} ms (+{100*(t_detect/t_sizes - 1):.0f}%)"
    )

    return stats


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NUMBA-OPTIMIZED PP MODULE - BENCHMARKS")
    print("=" * 60 + "\n")

    # Run kernel benchmarks
    benchmark_kernels(100)

    print("\n")

    # Run cluster benchmarks
    stats = benchmark_cluster_detection(100)
    print(
        f"\nSample stats: largest={stats['largest']}, "
        f"largest_fraction={stats['largest_fraction']:.3f}, "
        f"n_clusters={stats['n_clusters']}"
    )
