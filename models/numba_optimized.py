#!/usr/bin/env python3
"""
Numba-optimized kernels for predator-prey cellular automaton.

ENHANCED VERSION: Added full cluster detection with labels + percolation detection.

Key additions:
- detect_clusters_fast(): Returns (labels, sizes_dict) like Hoshen-Kopelman
- get_cluster_stats_fast(): Full statistics including largest_fraction
- get_percolating_cluster_fast(): Percolation detection for phase transitions

Optimizations:
1. Cell-list PCF: O(N) average instead of O(N²) brute force
2. Pre-allocated work buffers for async kernel
3. Consistent dtypes throughout
4. cache=True for persistent JIT compilation

Usage:
    from numba_optimized_enhanced import (
        PPKernel,
        compute_all_pcfs_fast,
        measure_cluster_sizes_fast,      # Sizes only (fastest)
        detect_clusters_fast,            # Labels + sizes dict
        get_cluster_stats_fast,          # Full statistics
        get_percolating_cluster_fast,    # Percolation detection
        NUMBA_AVAILABLE
    )
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
    """Seed Numba's internal RNG from within a JIT context."""
    np.random.seed(seed)


# ============================================================================
# PREDATOR-PREY KERNELS
# ============================================================================
@njit(cache=True)
def _pp_async_kernel_fast(
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
    """Partially synchronous predator-prey update kernel."""
    rows, cols = grid.shape
    n_shifts = len(dr_arr)
    grid_copy = grid.copy()
    prey_death_arr_copy = prey_death_arr.copy()

    prey_death = np.random.random(size=grid.shape)
    grid_copy[(grid == 1) & (prey_death < prey_death_arr)] = 0
    prey_death_arr_copy[(grid == 1) & (prey_death < prey_death_arr)] = np.nan

    pred_death = np.random.random(size=grid.shape)
    grid_copy[(grid == 2) & (pred_death < pred_death_val)] = 0

    count = np.count_nonzero(grid)
    indices = np.random.permutation(count)
    rs = indices // cols
    cs = indices % cols

    nb = np.random.randint(0, n_shifts, size=count)
    nrs = (rs + dr_arr[nb]) % rows
    ncs = (cs + dc_arr[nb]) % cols

    for r, c, nr, nc in zip(rs, cs, nrs, ncs):
        state = grid[r, c]
        nstate = grid[nr, nc]

        if state == 1 and nstate == 0 and np.random.random() < p_birth_val:
            grid_copy[nr, nc] = 1
            parent_val = prey_death_arr[r, c]
            if not evolution_stopped:
                child_val = parent_val + np.random.normal(0, evolve_sd)
                prey_death_arr_copy[nr, nc] = np.clip(child_val, evolve_min, evolve_max)
            else:
                prey_death_arr_copy[nr, nc] = parent_val

        elif state == 2 and nstate == 1 and np.random.random() < pred_birth_val:
            grid_copy[nr, nc] = 2
            prey_death_arr_copy[nr, nc] = np.nan

    grid = grid_copy
    prey_death_arr = prey_death_arr_copy

    return grid

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
    """Asynchronous predator-prey update kernel."""
    rows, cols = grid.shape
    n_shifts = len(dr_arr)
    
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
        occupied_buffer[i, 0], occupied_buffer[j, 0] = occupied_buffer[j, 0], occupied_buffer[i, 0]
        occupied_buffer[i, 1], occupied_buffer[j, 1] = occupied_buffer[j, 1], occupied_buffer[i, 1]
    
    for i in range(count):
        r = occupied_buffer[i, 0]
        c = occupied_buffer[i, 1]
        
        state = grid[r, c]
        if state == 0:
            continue

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
    """Async predator-prey update kernel with directed hunting."""
    rows, cols = grid.shape
    n_shifts = len(dr_arr)
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                occupied_buffer[count, 0] = r
                occupied_buffer[count, 1] = c
                count += 1
    
    for i in range(count - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        occupied_buffer[i, 0], occupied_buffer[j, 0] = occupied_buffer[j, 0], occupied_buffer[i, 0]
        occupied_buffer[i, 1], occupied_buffer[j, 1] = occupied_buffer[j, 1], occupied_buffer[i, 1]
    
    for i in range(count):
        r = occupied_buffer[i, 0]
        c = occupied_buffer[i, 1]
        
        state = grid[r, c]
        if state == 0:
            continue

        if state == 1:  # PREY
            nbi = np.random.randint(0, n_shifts)
            nr = (r + dr_arr[nbi]) % rows
            nc = (c + dc_arr[nbi]) % cols
            
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

    return grid


class PPKernel:
    """Wrapper for predator-prey kernel with pre-allocated buffers."""
    
    def __init__(self, rows: int, cols: int, neighborhood: str = "moore", 
                 directed_hunting: bool = False):
        self.rows = rows
        self.cols = cols
        self.directed_hunting = directed_hunting
        self._occupied_buffer = np.empty((rows * cols, 2), dtype=np.int32)
        
        if neighborhood == "moore":
            self._dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
            self._dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        else:
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
        if self.directed_hunting:
            return _pp_async_kernel_directed(
                grid, prey_death_arr,
                prey_birth, prey_death, pred_birth, pred_death,
                self._dr, self._dc,
                evolve_sd, evolve_min, evolve_max,
                evolution_stopped,
                self._occupied_buffer,
            )
        else:
            return _pp_async_kernel_random(
                grid, prey_death_arr,
                prey_birth, prey_death, pred_birth, pred_death,
                self._dr, self._dc,
                evolve_sd, evolve_min, evolve_max,
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
    """Stack-based flood fill with configurable neighborhood and periodic BC."""
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
    """Measure all cluster sizes for a species (sizes only)."""
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


@njit(cache=True)
def _check_percolation(
    labels: np.ndarray,
    sizes: np.ndarray,
    direction: int,
) -> Tuple[bool, int, int]:
    """
    Check for percolating clusters.
    
    Args:
        direction: 0=horizontal, 1=vertical, 2=both
    
    Returns:
        percolates, perc_label, perc_size
    """
    rows, cols = labels.shape
    max_label = len(sizes)
    
    touches_left = np.zeros(max_label + 1, dtype=np.bool_)
    touches_right = np.zeros(max_label + 1, dtype=np.bool_)
    touches_top = np.zeros(max_label + 1, dtype=np.bool_)
    touches_bottom = np.zeros(max_label + 1, dtype=np.bool_)
    
    for i in range(rows):
        if labels[i, 0] > 0:
            touches_left[labels[i, 0]] = True
        if labels[i, cols - 1] > 0:
            touches_right[labels[i, cols - 1]] = True
    
    for j in range(cols):
        if labels[0, j] > 0:
            touches_top[labels[0, j]] = True
        if labels[rows - 1, j] > 0:
            touches_bottom[labels[rows - 1, j]] = True
    
    best_label = 0
    best_size = 0
    
    for label in range(1, max_label + 1):
        percolates_h = touches_left[label] and touches_right[label]
        percolates_v = touches_top[label] and touches_bottom[label]
        
        is_percolating = False
        if direction == 0:
            is_percolating = percolates_h
        elif direction == 1:
            is_percolating = percolates_v
        else:
            is_percolating = percolates_h or percolates_v
        
        if is_percolating:
            cluster_size = sizes[label - 1]
            if cluster_size > best_size:
                best_size = cluster_size
                best_label = label
    
    return best_label > 0, best_label, best_size


# ============================================================================
# PUBLIC API - CLUSTER DETECTION
# ============================================================================

def measure_cluster_sizes_fast(
    grid: np.ndarray, 
    species: int,
    neighborhood: str = "moore",
) -> np.ndarray:
    """
    Measure cluster sizes only (fastest method).
    
    Use when you only need size statistics, not the label array.
    ~25x faster than pure Python.
    
    Args:
        grid: 2D array of cell states
        species: Target species value (1=prey, 2=predator)
        neighborhood: 'moore' (8-connected) or 'neumann' (4-connected)
    
    Returns:
        1D array of cluster sizes
    """
    grid_int = np.asarray(grid, dtype=np.int32)
    moore = (neighborhood == "moore")
    return _measure_clusters(grid_int, np.int32(species), moore)


def detect_clusters_fast(
    grid: np.ndarray,
    species: int,
    neighborhood: str = "moore",
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Full cluster detection with labels (Numba-accelerated).
    
    Returns both the label array and size dictionary for richer analysis.
    
    Args:
        grid: 2D array of cell states
        species: Target species value (1=prey, 2=predator)
        neighborhood: 'moore' (8-connected) or 'neumann' (4-connected)
    
    Returns:
        labels: 2D array where each cell has its cluster ID (0 = non-target)
        sizes: Dict mapping cluster_id -> cluster_size
    
    Example:
        >>> labels, sizes = detect_clusters_fast(grid, species=1)
        >>> largest_id = max(sizes, key=sizes.get)
        >>> largest_size = sizes[largest_id]
    """
    grid_int = np.asarray(grid, dtype=np.int32)
    moore = (neighborhood == "moore")
    labels, sizes_arr = _detect_clusters_numba(grid_int, np.int32(species), moore)
    sizes_dict = {i + 1: int(sizes_arr[i]) for i in range(len(sizes_arr))}
    return labels, sizes_dict


def get_cluster_stats_fast(
    grid: np.ndarray,
    species: int,
    neighborhood: str = "moore",
) -> Dict:
    """
    Compute comprehensive cluster statistics (Numba-accelerated).
    
    Args:
        grid: 2D array of cell states
        species: Target species value
        neighborhood: 'moore' or 'neumann'
    
    Returns:
        Dictionary with keys:
        - 'n_clusters': Total number of clusters
        - 'sizes': Array of sizes (sorted descending)
        - 'largest': Size of largest cluster
        - 'largest_fraction': S_max / N (order parameter for percolation)
        - 'mean_size': Mean cluster size
        - 'size_distribution': Dict[size -> count]
        - 'labels': Cluster label array
        - 'size_dict': Dict[label -> size]
    """
    labels, size_dict = detect_clusters_fast(grid, species, neighborhood)
    
    if len(size_dict) == 0:
        return {
            'n_clusters': 0,
            'sizes': np.array([], dtype=np.int32),
            'largest': 0,
            'largest_fraction': 0.0,
            'mean_size': 0.0,
            'size_distribution': {},
            'labels': labels,
            'size_dict': size_dict,
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
        'n_clusters': len(size_dict),
        'sizes': sizes_sorted,
        'largest': largest,
        'largest_fraction': float(largest) / total_pop if total_pop > 0 else 0.0,
        'mean_size': float(np.mean(sizes)),
        'size_distribution': size_dist,
        'labels': labels,
        'size_dict': size_dict,
    }


def get_percolating_cluster_fast(
    grid: np.ndarray,
    species: int,
    neighborhood: str = "moore",
    direction: str = "both",
) -> Tuple[bool, int, int, np.ndarray]:
    """
    Detect percolating (spanning) clusters (Numba-accelerated).
    
    A percolating cluster connects opposite edges of the grid,
    indicating a phase transition in percolation theory.
    
    Args:
        grid: 2D array of cell states
        species: Target species value
        neighborhood: 'moore' or 'neumann'
        direction: 'horizontal', 'vertical', or 'both'
    
    Returns:
        percolates: True if a spanning cluster exists
        cluster_label: Label of the percolating cluster (0 if none)
        cluster_size: Size of the percolating cluster (0 if none)
        labels: Full cluster label array
    
    Example:
        >>> percolates, label, size, labels = get_percolating_cluster_fast(grid, 1)
        >>> if percolates:
        >>>     print(f"Prey percolates with {size} cells!")
    """
    grid_int = np.asarray(grid, dtype=np.int32)
    moore = (neighborhood == "moore")
    labels, sizes_arr = _detect_clusters_numba(grid_int, np.int32(species), moore)
    
    dir_map = {'horizontal': 0, 'vertical': 1, 'both': 2}
    dir_int = dir_map.get(direction, 2)
    
    percolates, perc_label, perc_size = _check_percolation(labels, sizes_arr, dir_int)
    return percolates, int(perc_label), int(perc_size), labels


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
    """Build cell list for spatial hashing."""
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
    r1: float, c1: float,
    r2: float, c2: float,
    L_row: float, L_col: float,
) -> float:
    """Squared periodic distance."""
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
    """Compute PCF histogram using cell lists."""
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
    """Cell-list accelerated PCF computation."""
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
    
    indices_j, offsets_j, counts_j, cell_size_r, cell_size_c = \
        _build_cell_list(pos_j, n_cells, L_row, L_col)
    
    hist = _pcf_cell_list(
        pos_i, pos_j,
        indices_j, offsets_j, counts_j,
        cell_size_r, cell_size_c,
        L_row, L_col,
        max_distance, n_bins,
        self_correlation, n_cells,
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
    """Compute all three PCFs using cell-list acceleration."""
    rows, cols = grid.shape
    if max_distance is None:
        max_distance = min(rows, cols) / 4.0
    
    prey_pos = np.argwhere(grid == 1)
    pred_pos = np.argwhere(grid == 2)
    
    results = {}
    
    dist, pcf, n = compute_pcf_periodic_fast(
        prey_pos, prey_pos, (rows, cols), max_distance, n_bins,
        self_correlation=True,
    )
    results['prey_prey'] = (dist, pcf, n)
    
    dist, pcf, n = compute_pcf_periodic_fast(
        pred_pos, pred_pos, (rows, cols), max_distance, n_bins,
        self_correlation=True,
    )
    results['pred_pred'] = (dist, pcf, n)
    
    dist, pcf, n = compute_pcf_periodic_fast(
        prey_pos, pred_pos, (rows, cols), max_distance, n_bins,
        self_correlation=False,
    )
    results['prey_pred'] = (dist, pcf, n)
    
    return results


# ============================================================================
# WARMUP & BENCHMARKS
# ============================================================================

def warmup_numba_kernels(grid_size: int = 100, directed_hunting: bool = False):
    """Pre-compile all Numba kernels."""
    if not NUMBA_AVAILABLE:
        return
    
    set_numba_seed(0)
    
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    grid[::3, ::3] = 1
    grid[::5, ::5] = 2
    
    prey_death_arr = np.full((grid_size, grid_size), 0.05, dtype=np.float64)
    prey_death_arr[grid != 1] = np.nan
    
    kernel_random = PPKernel(grid_size, grid_size, directed_hunting=False)
    kernel_random.update(grid.copy(), prey_death_arr.copy(), 0.2, 0.05, 0.2, 0.1)
    
    if directed_hunting:
        kernel_directed = PPKernel(grid_size, grid_size, directed_hunting=True)
        kernel_directed.update(grid.copy(), prey_death_arr.copy(), 0.2, 0.05, 0.2, 0.1)
    
    _ = compute_all_pcfs_fast(grid, max_distance=20.0, n_bins=20)
    _ = measure_cluster_sizes_fast(grid, 1)
    _ = detect_clusters_fast(grid, 1)
    _ = get_cluster_stats_fast(grid, 1)
    _ = get_percolating_cluster_fast(grid, 1)


def benchmark_cluster_detection(grid_size: int = 100, n_runs: int = 20):
    """Benchmark cluster detection methods."""
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
    _ = get_percolating_cluster_fast(grid, 1)
    
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
    
    # Benchmark percolation
    t0 = time.perf_counter()
    for _ in range(n_runs):
        perc, label, size, _ = get_percolating_cluster_fast(grid, 1)
    t_perc = (time.perf_counter() - t0) / n_runs * 1000
    print(f"get_percolating_cluster_fast: {t_perc:.2f} ms  (percolates={perc})")
    
    print(f"\nOverhead for labels: {t_detect - t_sizes:.2f} ms (+{100*(t_detect/t_sizes - 1):.0f}%)")
    
    return stats


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ENHANCED NUMBA MODULE BENCHMARKS")
    print("=" * 60 + "\n")
    
    warmup_numba_kernels()
    stats = benchmark_cluster_detection(100)
    print(f"\nSample stats: largest={stats['largest']}, "
          f"largest_fraction={stats['largest_fraction']:.3f}, "
          f"n_clusters={stats['n_clusters']}")