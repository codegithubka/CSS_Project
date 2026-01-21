
"""
NOTE: First call to each function is slow due to JIT compilation.
      Subsequent calls are fast. cache=True persists compilation to disk.
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
    
@njit(cache=True)
def _pp_async_kernel(
    grid, 
    prey_death_arr, 
    p_birth_val, p_death_val, pred_birth_val, pred_death_val,
    dr_arr, dc_arr,
    evolve_sd, evolve_min, evolve_max,
    evolution_stopped
):
    rows, cols = grid.shape
    n_shifts = len(dr_arr)
    
    # 1. PRE-ALLOCATE COORDINATE BUFFER
    # Instead of a list, we use a fixed-size array. 
    # Max possible occupied cells is rows * cols.
    occupied = np.empty((rows * cols, 2), dtype=np.int32)
    count = 0
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                occupied[count, 0] = r
                occupied[count, 1] = c
                count += 1
    
    # 2. IN-PLACE FISHER-YATES SHUFFLE
    # We only shuffle the part of the buffer we actually filled (up to 'count')
    for i in range(count - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        # Swap row index
        r_temp = occupied[i, 0]
        occupied[i, 0] = occupied[j, 0]
        occupied[j, 0] = r_temp
        # Swap col index
        c_temp = occupied[i, 1]
        occupied[i, 1] = occupied[j, 1]
        occupied[j, 1] = c_temp
        
    # 3. PROCESS ACTIVE CELLS
    for i in range(count):
        r = occupied[i, 0]
        c = occupied[i, 1]
        
        state = grid[r, c]
        if state == 0: 
            continue 

        # Pick random neighbor
        nbi = np.random.randint(0, n_shifts)
        nr = (r + dr_arr[nbi]) % rows
        nc = (c + dc_arr[nbi]) % cols

        if state == 1: # PREY
            # Death logic
            if np.random.random() < prey_death_arr[r, c]:
                grid[r, c] = 0
                prey_death_arr[r, c] = np.nan
            # Birth logic
            elif grid[nr, nc] == 0:
                if np.random.random() < p_birth_val:
                    grid[nr, nc] = 1
                    parent_val = prey_death_arr[r, c]
                    if not evolution_stopped:
                        child_val = parent_val + np.random.normal(0, evolve_sd)
                        # Manual clip is faster than np.clip in Numba
                        if child_val < evolve_min: child_val = evolve_min
                        if child_val > evolve_max: child_val = evolve_max
                        prey_death_arr[nr, nc] = child_val
                    else:
                        prey_death_arr[nr, nc] = parent_val

        elif state == 2: # PREDATOR
            # Death logic
            if np.random.random() < pred_death_val:
                grid[r, c] = 0
            # Birth logic (eating prey)
            elif grid[nr, nc] == 1:
                if np.random.random() < pred_birth_val:
                    grid[nr, nc] = 2
                    prey_death_arr[nr, nc] = np.nan
                    
    return grid

@njit(cache=True)
def _periodic_distance(r1, c1, r2, c2, L_row, L_col):
    """Compute distance with periodic boundary conditions."""
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    
    # Periodic wrapping - use shorter path
    if dr > L_row * 0.5:
        dr = L_row - dr
    if dc > L_col * 0.5:
        dc = L_col - dc
    
    return np.sqrt(dr * dr + dc * dc)


@njit(parallel=True, cache=True)
def _compute_distance_histogram(
    pos_i, 
    pos_j, 
    L_row, 
    L_col,
    max_distance, 
    n_bins, 
    self_correlation
):
    """
    Compute histogram of pairwise distances - THE expensive operation.
    
    This is O(N*M) but parallelized across the first array.
    For self-correlation, only computes upper triangle to avoid double-counting.
    """
    N = pos_i.shape[0]
    M = pos_j.shape[0]
    bin_width = max_distance / n_bins
    
    # Final histogram
    hist = np.zeros(n_bins, dtype=np.int64)
    
    # Parallel over first position array
    for i in prange(N):
        # Thread-local histogram to minimize race conditions
        local_hist = np.zeros(n_bins, dtype=np.int64)
        
        r1, c1 = pos_i[i, 0], pos_i[i, 1]
        
        # For self-correlation, only count each pair once (upper triangle)
        start_j = i + 1 if self_correlation else 0
        
        for j in range(start_j, M):
            r2, c2 = pos_j[j, 0], pos_j[j, 1]
            
            d = _periodic_distance(r1, c1, r2, c2, L_row, L_col)
            
            if 0 < d < max_distance:
                bin_idx = int(d / bin_width)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                local_hist[bin_idx] += 1
        
        # Accumulate into global histogram
        for b in range(n_bins):
            hist[b] += local_hist[b]
    
    # For self-correlation, we computed upper triangle only
    # Double count to get full pair count (excluding diagonal)
    if self_correlation:
        for b in range(n_bins):
            hist[b] = hist[b] * 2
    
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
    Numba-accelerated PCF computation.
    
    Drop-in replacement for compute_pcf_periodic().
    10-20x faster for typical grid sizes (100x100 with 500+ individuals).
    
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
    
    # Handle empty position arrays
    if len(positions_i) == 0 or len(positions_j) == 0:
        bin_width = max_distance / n_bins
        bin_centers = np.linspace(bin_width/2, max_distance - bin_width/2, n_bins)
        return bin_centers, np.ones(n_bins), 0
    
    # Ensure contiguous float64 arrays for Numba
    pos_i = np.ascontiguousarray(positions_i, dtype=np.float64)
    pos_j = np.ascontiguousarray(positions_j, dtype=np.float64)
    
    # Compute histogram using Numba-accelerated function
    hist = _compute_distance_histogram(
        pos_i, pos_j, L_row, L_col,
        max_distance, n_bins, self_correlation
    )
    
    # Compute expected counts (normalization) - this is cheap, no need to optimize
    bin_width = max_distance / n_bins
    bin_centers = np.linspace(bin_width/2, max_distance - bin_width/2, n_bins)
    
    n_i, n_j = len(positions_i), len(positions_j)
    
    if self_correlation:
        # Auto-correlation: N*(N-1) total pairs
        density_product = n_i * (n_i - 1) / (area * area)
    else:
        # Cross-correlation: N*M total pairs
        density_product = n_i * n_j / (area * area)
    
    # Expected count in each annulus: density * annulus_area * total_area
    expected = np.zeros(n_bins)
    for i in range(n_bins):
        r = bin_centers[i]
        annulus_area = 2 * np.pi * r * bin_width
        expected[i] = density_product * annulus_area * area
    
    # PCF = observed / expected
    pcf = np.ones(n_bins)
    mask = expected > 1.0  # Avoid division by tiny numbers
    pcf[mask] = hist[mask] / expected[mask]
    
    n_pairs = int(np.sum(hist))
    
    return bin_centers, pcf, n_pairs


def compute_all_pcfs_fast(
    grid: np.ndarray,
    max_distance: Optional[float] = None,
    n_bins: int = 50,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """
    Compute all three PCFs using Numba acceleration.
    
    Drop-in replacement for compute_all_pcfs().
    """
    rows, cols = grid.shape
    if max_distance is None:
        max_distance = min(rows, cols) / 4.0
    
    # Extract positions
    prey_pos = np.argwhere(grid == 1)
    pred_pos = np.argwhere(grid == 2)
    
    results = {}
    
    # Prey auto-correlation (C_rr)
    dist, pcf, n = compute_pcf_periodic_fast(
        prey_pos, prey_pos, (rows, cols), max_distance, n_bins, 
        self_correlation=True
    )
    results['prey_prey'] = (dist, pcf, n)
    
    # Predator auto-correlation (C_cc)
    dist, pcf, n = compute_pcf_periodic_fast(
        pred_pos, pred_pos, (rows, cols), max_distance, n_bins,
        self_correlation=True
    )
    results['pred_pred'] = (dist, pcf, n)
    
    # Cross-correlation (C_cr)
    dist, pcf, n = compute_pcf_periodic_fast(
        prey_pos, pred_pos, (rows, cols), max_distance, n_bins,
        self_correlation=False
    )
    results['prey_pred'] = (dist, pcf, n)
    
    return results

@njit(cache=True)
def _flood_fill_numba(grid, visited, start_r, start_c, target, rows, cols):
    """
    Stack-based flood fill for cluster detection.
    
    Uses manual stack instead of recursion (Numba-safe).
    4-connected neighbors (von Neumann neighborhood).
    """
    # Manual stack (Numba doesn't support Python lists efficiently)
    max_stack_size = rows * cols
    stack_r = np.empty(max_stack_size, dtype=np.int32)
    stack_c = np.empty(max_stack_size, dtype=np.int32)
    stack_ptr = 0
    
    # Push starting cell
    stack_r[stack_ptr] = start_r
    stack_c[stack_ptr] = start_c
    stack_ptr += 1
    visited[start_r, start_c] = True
    
    size = 0
    
    # 4-connected neighbors (up, down, left, right)
    dr = np.array([-1, 1, 0, 0], dtype=np.int32)
    dc = np.array([0, 0, -1, 1], dtype=np.int32)
    
    while stack_ptr > 0:
        # Pop
        stack_ptr -= 1
        r = stack_r[stack_ptr]
        c = stack_c[stack_ptr]
        size += 1
        
        # Check 4 neighbors
        for k in range(4):
            nr = r + dr[k]
            nc = c + dc[k]
            
            # Bounds check (no periodic boundary for clusters)
            if 0 <= nr < rows and 0 <= nc < cols:
                if not visited[nr, nc] and grid[nr, nc] == target:
                    visited[nr, nc] = True
                    stack_r[stack_ptr] = nr
                    stack_c[stack_ptr] = nc
                    stack_ptr += 1
    
    return size


@njit(cache=True)
def _measure_clusters_numba(grid, species):
    """
    Numba-accelerated cluster measurement.
    
    Returns array of cluster sizes for the given species.
    """
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=np.bool_)
    
    # Pre-allocate for maximum possible clusters
    max_clusters = rows * cols
    sizes = np.empty(max_clusters, dtype=np.int32)
    n_clusters = 0
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == species and not visited[r, c]:
                size = _flood_fill_numba(grid, visited, r, c, species, rows, cols)
                sizes[n_clusters] = size
                n_clusters += 1
    
    return sizes[:n_clusters]


def measure_cluster_sizes_fast(grid: np.ndarray, species: int) -> np.ndarray:
    """
    Numba-accelerated cluster measurement.
    
    Drop-in replacement for measure_cluster_sizes().
    5-10x faster than scipy.ndimage.label for typical grids.
    
    Args:
        grid: 2D array with species codes (0=empty, 1=prey, 2=predator)
        species: which species to measure (1 or 2)
    
    Returns:
        Array of cluster sizes
    """
    # Ensure correct dtype for Numba
    grid_int = np.asarray(grid, dtype=np.int32)
    return _measure_clusters_numba(grid_int, np.int32(species))


def benchmark_pcf(grid_size=100, n_prey=500, n_pred=200, n_runs=10):
    """
    Benchmark PCF computation.
    
    Run this to verify Numba is working and see speedup.
    """
    import time
    
    print("=" * 60)
    print(f"PCF BENCHMARK (grid={grid_size}x{grid_size})")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("=" * 60)
    
    # Create test grid
    np.random.seed(42)
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    # Place prey and predators randomly
    all_positions = np.random.permutation(grid_size * grid_size)
    prey_positions = all_positions[:n_prey]
    pred_positions = all_positions[n_prey:n_prey + n_pred]
    
    for pos in prey_positions:
        grid[pos // grid_size, pos % grid_size] = 1
    for pos in pred_positions:
        grid[pos // grid_size, pos % grid_size] = 2
    
    actual_prey = np.sum(grid == 1)
    actual_pred = np.sum(grid == 2)
    print(f"Prey: {actual_prey}, Predators: {actual_pred}")
    
    # Warm up Numba (first call compiles)
    print("\nWarming up (JIT compilation)...")
    t0 = time.perf_counter()
    _ = compute_all_pcfs_fast(grid, max_distance=20, n_bins=20)
    warmup_time = time.perf_counter() - t0
    print(f"Warmup time: {warmup_time:.2f}s (includes compilation)")
    
    # Benchmark
    print(f"\nBenchmarking {n_runs} runs...")
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = compute_all_pcfs_fast(grid, max_distance=20, n_bins=20)
    elapsed = time.perf_counter() - start
    
    ms_per_call = (elapsed / n_runs) * 1000
    print(f"Numba PCF: {ms_per_call:.1f} ms/call")
    
    return ms_per_call


def benchmark_clusters(grid_size=100, density=0.3, n_runs=10):
    """
    Benchmark cluster measurement.
    """
    import time
    from scipy import ndimage
    
    print("=" * 60)
    print(f"CLUSTER BENCHMARK (grid={grid_size}x{grid_size})")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("=" * 60)
    
    # Create test grid with random prey
    np.random.seed(42)
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    n_prey = int(grid_size * grid_size * density)
    positions = np.random.permutation(grid_size * grid_size)[:n_prey]
    for pos in positions:
        grid[pos // grid_size, pos % grid_size] = 1
    
    print(f"Prey cells: {np.sum(grid == 1)}")
    
    # Warm up
    print("\nWarming up...")
    _ = measure_cluster_sizes_fast(grid, 1)
    
    # Benchmark Numba version
    print(f"\nBenchmarking Numba ({n_runs} runs)...")
    start = time.perf_counter()
    for _ in range(n_runs):
        sizes_numba = measure_cluster_sizes_fast(grid, 1)
    numba_time = (time.perf_counter() - start) / n_runs * 1000
    
    # Benchmark scipy version
    print(f"Benchmarking scipy ({n_runs} runs)...")
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    start = time.perf_counter()
    for _ in range(n_runs):
        binary_mask = (grid == 1).astype(int)
        labeled, n = ndimage.label(binary_mask, structure=structure)
        if n > 0:
            sizes_scipy = np.array(ndimage.sum(binary_mask, labeled, range(1, n + 1)), dtype=int)
    scipy_time = (time.perf_counter() - start) / n_runs * 1000
    
    print(f"\nResults:")
    print(f"  Numba:  {numba_time:.2f} ms/call, found {len(sizes_numba)} clusters")
    print(f"  Scipy:  {scipy_time:.2f} ms/call, found {n} clusters")
    print(f"  Speedup: {scipy_time/numba_time:.1f}x")
    
    return numba_time, scipy_time


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("NUMBA OPTIMIZATION BENCHMARKS")
    print("=" * 60 + "\n")
    
    benchmark_pcf(grid_size=100, n_prey=500, n_pred=200)
    print()
    benchmark_clusters(grid_size=100, density=0.3)
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
    


if __name__ == "__main__":
    run_all_benchmarks()