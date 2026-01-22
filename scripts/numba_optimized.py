
#!/usr/bin/env python3
"""
Numba-optimized kernels for predator-prey cellular automaton.

Optimizations:
1. Cell-list PCF: O(N) average instead of O(N²) brute force
2. Pre-allocated work buffers for async kernel
3. Consistent dtypes throughout
4. cache=True for persistent JIT compilation

Usage:
    from scripts.numba_optimized import (
        PPKernel,
        compute_all_pcfs_fast,
        measure_cluster_sizes_fast,
        NUMBA_AVAILABLE
    )
    
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
    # Create dummy decorators so code doesn't crash
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
    
@njit(cache=True)
def _pp_async_kernel(
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
    occupied_buffer: np.ndarray,  # Pre-allocated: (rows*cols, 2), int32
) -> np.ndarray:
    """
    Asynchronous predator-prey update kernel.
    
    Args:
        grid: (rows, cols) int32 array - 0=empty, 1=prey, 2=predator
        prey_death_arr: (rows, cols) float64 array - evolved prey death rates
        occupied_buffer: Pre-allocated work buffer for cell coordinates
        
    Returns:
        Updated grid (modified in-place)
    """
    rows, cols = grid.shape
    n_shifts = len(dr_arr)
    
    # Collect occupied cells into pre-allocated buffer
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
        # Swap
        occupied_buffer[i, 0], occupied_buffer[j, 0] = occupied_buffer[j, 0], occupied_buffer[i, 0]
        occupied_buffer[i, 1], occupied_buffer[j, 1] = occupied_buffer[j, 1], occupied_buffer[i, 1]
    
    # Process active cells
    for i in range(count):
        r = occupied_buffer[i, 0]
        c = occupied_buffer[i, 1]
        
        state = grid[r, c]
        if state == 0:
            continue

        # Pick random neighbor
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


class PPKernel:
    """
    Wrapper for predator-prey kernel with pre-allocated buffers.
    
    Creates buffers once at initialization, reuses for all updates.
    This avoids allocation overhead inside the hot loop.
    
    Usage:
        kernel = PPKernel(100, 100, neighborhood="moore")
        for step in range(1000):
            kernel.update(grid, prey_death_arr, ...)
    """
    def __init__(self, rows: int, cols: int, neighborhood: str = "moore"):
        self.rows = rows
        self.cols = cols
        
        # Pre-allocate work buffer
        self._occupied_buffer = np.empty((rows * cols, 2), dtype=np.int32)
        
        # Neighbor offsets
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
        """Update grid one step."""
        return _pp_async_kernel(
            grid, prey_death_arr,
            prey_birth, prey_death, pred_birth, pred_death,
            self._dr, self._dc,
            evolve_sd, evolve_min, evolve_max,
            evolution_stopped,
            self._occupied_buffer,
        )
        
@njit(cache=True)
def _build_cell_list(
    positions: np.ndarray,
    n_cells: int,
    L_row: float,
    L_col: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Build cell list for spatial hashing.
    
    Returns:
        indices: Particle indices sorted by cell
        offsets: Starting index for each cell
        counts: Number of particles in each cell
        cell_size_r, cell_size_c: Cell dimensions
    """
    n_pos = len(positions)
    cell_size_r = L_row / n_cells
    cell_size_c = L_col / n_cells
    
    # Count particles per cell
    cell_counts = np.zeros((n_cells, n_cells), dtype=np.int32)
    for i in range(n_pos):
        cr = int(positions[i, 0] / cell_size_r) % n_cells
        cc = int(positions[i, 1] / cell_size_c) % n_cells
        cell_counts[cr, cc] += 1
    
    # Build cumulative offsets
    offsets = np.zeros((n_cells, n_cells), dtype=np.int32)
    running = 0
    for cr in range(n_cells):
        for cc in range(n_cells):
            offsets[cr, cc] = running
            running += cell_counts[cr, cc]
    
    # Fill particle indices
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
    """Squared periodic distance (avoids sqrt in inner loop)."""
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
    Compute PCF histogram using cell lists.
    
    Only checks neighboring cells within max_distance.
    O(N * k) where k is average particles per cell neighborhood.
    """
    n_i = len(pos_i)
    bin_width = max_distance / n_bins
    max_dist_sq = max_distance * max_distance
    
    # How many cells to check in each direction
    cells_to_check = int(np.ceil(max_distance / min(cell_size_r, cell_size_c))) + 1
    
    hist = np.zeros(n_bins, dtype=np.int64)
    
    for i in prange(n_i):
        local_hist = np.zeros(n_bins, dtype=np.int64)
        r1, c1 = pos_i[i, 0], pos_i[i, 1]
        
        # Find which cell this particle is in
        cell_r = int(r1 / cell_size_r) % n_cells
        cell_c = int(c1 / cell_size_c) % n_cells
        
        # Check neighboring cells
        for dcr in range(-cells_to_check, cells_to_check + 1):
            for dcc in range(-cells_to_check, cells_to_check + 1):
                ncr = (cell_r + dcr) % n_cells
                ncc = (cell_c + dcc) % n_cells
                
                # Iterate particles in this cell
                start = offsets_j[ncr, ncc]
                end = start + counts_j[ncr, ncc]
                
                for idx in range(start, end):
                    j = indices_j[idx]
                    
                    # Skip self-pairs for auto-correlation
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
    
    # Double count for auto-correlation (we only computed upper triangle)
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
    Cell-list accelerated PCF computation.
    
    O(N * k) average case where k is particles per cell neighborhood,
    instead of O(N * M) for brute force. 10-50x faster for typical grids.
    
    Args:
        positions_i: (N, 2) array of (row, col) positions for species i
        positions_j: (M, 2) array of (row, col) positions for species j
        grid_shape: (rows, cols) shape of the grid
        max_distance: maximum distance to compute PCF
        n_bins: number of distance bins
        self_correlation: if True, computing C_ii (exclude self-pairs)
    
    Returns:
        bin_centers: array of distance bin centers
        pcf: C_ij(r) values (1.0 = random, >1 = clustering, <1 = segregation)
        n_pairs: total number of pairs counted
    """
    rows, cols = grid_shape
    L_row, L_col = float(rows), float(cols)
    area = L_row * L_col
    
    # Handle empty arrays
    bin_width = max_distance / n_bins
    bin_centers = np.linspace(bin_width / 2, max_distance - bin_width / 2, n_bins)
    
    if len(positions_i) == 0 or len(positions_j) == 0:
        return bin_centers, np.ones(n_bins), 0
    
    # Choose cell size ~ max_distance for optimal performance
    n_cells = max(4, int(min(rows, cols) / max_distance))
    
    # Ensure contiguous float64 arrays
    pos_i = np.ascontiguousarray(positions_i, dtype=np.float64)
    pos_j = np.ascontiguousarray(positions_j, dtype=np.float64)
    
    # Build cell list for positions_j
    indices_j, offsets_j, counts_j, cell_size_r, cell_size_c = \
        _build_cell_list(pos_j, n_cells, L_row, L_col)
    
    # Compute histogram
    hist = _pcf_cell_list(
        pos_i, pos_j,
        indices_j, offsets_j, counts_j,
        cell_size_r, cell_size_c,
        L_row, L_col,
        max_distance, n_bins,
        self_correlation, n_cells,
    )
    
    # Normalization
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
    Compute all three PCFs using cell-list acceleration.
    
    Returns dict with keys: 'prey_prey', 'pred_pred', 'prey_pred'
    Each value is (distances, pcf, n_pairs).
    """
    rows, cols = grid.shape
    if max_distance is None:
        max_distance = min(rows, cols) / 4.0
    
    prey_pos = np.argwhere(grid == 1)
    pred_pos = np.argwhere(grid == 2)
    
    results = {}
    
    # Prey auto-correlation (C_rr)
    dist, pcf, n = compute_pcf_periodic_fast(
        prey_pos, prey_pos, (rows, cols), max_distance, n_bins,
        self_correlation=True,
    )
    results['prey_prey'] = (dist, pcf, n)
    
    # Predator auto-correlation (C_cc)
    dist, pcf, n = compute_pcf_periodic_fast(
        pred_pos, pred_pos, (rows, cols), max_distance, n_bins,
        self_correlation=True,
    )
    results['pred_pred'] = (dist, pcf, n)
    
    # Cross-correlation (C_cr)
    dist, pcf, n = compute_pcf_periodic_fast(
        prey_pos, pred_pos, (rows, cols), max_distance, n_bins,
        self_correlation=False,
    )
    results['prey_pred'] = (dist, pcf, n)
    
    return results



@njit(cache=True)
def _flood_fill(
    grid: np.ndarray,
    visited: np.ndarray,
    start_r: int,
    start_c: int,
    target: int,
    rows: int,
    cols: int,
) -> int:
    """Stack-based flood fill for cluster detection (4-connected)."""
    max_stack = rows * cols
    stack_r = np.empty(max_stack, dtype=np.int32)
    stack_c = np.empty(max_stack, dtype=np.int32)
    stack_ptr = 0
    
    stack_r[stack_ptr] = start_r
    stack_c[stack_ptr] = start_c
    stack_ptr += 1
    visited[start_r, start_c] = True
    
    size = 0
    dr = np.array([-1, 1, 0, 0], dtype=np.int32)
    dc = np.array([0, 0, -1, 1], dtype=np.int32)
    
    while stack_ptr > 0:
        stack_ptr -= 1
        r = stack_r[stack_ptr]
        c = stack_c[stack_ptr]
        size += 1
        
        for k in range(4):
            nr = r + dr[k]
            nc = c + dc[k]
            
            if 0 <= nr < rows and 0 <= nc < cols:
                if not visited[nr, nc] and grid[nr, nc] == target:
                    visited[nr, nc] = True
                    stack_r[stack_ptr] = nr
                    stack_c[stack_ptr] = nc
                    stack_ptr += 1
    
    return size


@njit(cache=True)
def _measure_clusters(grid: np.ndarray, species: int) -> np.ndarray:
    """Measure all cluster sizes for a species."""
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=np.bool_)
    
    max_clusters = rows * cols
    sizes = np.empty(max_clusters, dtype=np.int32)
    n_clusters = 0
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == species and not visited[r, c]:
                size = _flood_fill(grid, visited, r, c, species, rows, cols)
                sizes[n_clusters] = size
                n_clusters += 1
    
    return sizes[:n_clusters]


def measure_cluster_sizes_fast(grid: np.ndarray, species: int) -> np.ndarray:
    """
    Numba-accelerated cluster measurement.
    
    5-10x faster than scipy.ndimage.label.
    """
    grid_int = np.asarray(grid, dtype=np.int32)
    return _measure_clusters(grid_int, np.int32(species))



def warmup_numba_kernels(grid_size: int = 100):
    """
    Pre-compile all Numba kernels.
    
    Call once at startup to avoid JIT overhead during parallel execution.
    """
    if not NUMBA_AVAILABLE:
        return
    
    set_numba_seed(0)
    
    # Dummy data
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    grid[::3, ::3] = 1  # Sparse prey
    grid[::5, ::5] = 2  # Sparse predators
    
    prey_death_arr = np.full((grid_size, grid_size), 0.05, dtype=np.float64)
    prey_death_arr[grid != 1] = np.nan
    
    # Warmup kernel
    kernel = PPKernel(grid_size, grid_size)
    kernel.update(grid.copy(), prey_death_arr.copy(), 0.2, 0.05, 0.2, 0.1)
    
    # Warmup PCF
    _ = compute_all_pcfs_fast(grid, max_distance=20.0, n_bins=20)
    
    # Warmup clusters
    _ = measure_cluster_sizes_fast(grid, 1)
    

def benchmark_pcf(grid_size: int = 100, n_runs: int = 10):
    """Benchmark PCF computation."""
    import time
    
    print("=" * 60)
    print(f"PCF BENCHMARK (grid={grid_size}x{grid_size})")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("=" * 60)
    
    np.random.seed(42)
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    n_prey = int(grid_size * grid_size * 0.30)
    n_pred = int(grid_size * grid_size * 0.15)
    
    positions = np.random.permutation(grid_size * grid_size)
    for pos in positions[:n_prey]:
        grid[pos // grid_size, pos % grid_size] = 1
    for pos in positions[n_prey:n_prey + n_pred]:
        grid[pos // grid_size, pos % grid_size] = 2
    
    print(f"Prey: {np.sum(grid == 1)}, Predators: {np.sum(grid == 2)}")
    
    # Warmup
    print("\nWarming up (JIT compilation)...")
    t0 = time.perf_counter()
    _ = compute_all_pcfs_fast(grid, max_distance=20, n_bins=20)
    print(f"Warmup: {time.perf_counter() - t0:.2f}s")
    
    # Benchmark
    print(f"\nBenchmarking {n_runs} runs...")
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = compute_all_pcfs_fast(grid, max_distance=20, n_bins=20)
    elapsed = time.perf_counter() - start
    
    print(f"Cell-list PCF: {elapsed/n_runs*1000:.1f} ms/call")
    return elapsed / n_runs * 1000


def benchmark_kernel(grid_size: int = 100, n_steps: int = 500):
    """Benchmark simulation kernel."""
    import time
    
    print("=" * 60)
    print(f"KERNEL BENCHMARK ({n_steps} steps, {grid_size}x{grid_size})")
    print("=" * 60)
    
    np.random.seed(42)
    grid = np.random.choice([0, 1, 2], size=(grid_size, grid_size), 
                            p=[0.55, 0.30, 0.15]).astype(np.int32)
    prey_death = np.full((grid_size, grid_size), 0.05, dtype=np.float64)
    prey_death[grid != 1] = np.nan
    
    kernel = PPKernel(grid_size, grid_size)
    
    # Warmup
    g = grid.copy()
    p = prey_death.copy()
    kernel.update(g, p, 0.2, 0.05, 0.2, 0.1)
    
    # Benchmark
    g = grid.copy()
    p = prey_death.copy()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        kernel.update(g, p, 0.2, 0.05, 0.2, 0.1, evolution_stopped=False)
    elapsed = (time.perf_counter() - t0) * 1000
    
    print(f"Total: {elapsed:.1f}ms for {n_steps} steps")
    print(f"Per step: {elapsed/n_steps:.3f}ms")
    return elapsed / n_steps


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NUMBA OPTIMIZATION BENCHMARKS")
    print("=" * 60 + "\n")
    
    benchmark_kernel()
    print()
    benchmark_pcf()