"""Performance tests for PP: compare runs with and without visualization.

Usage examples:
  python -m scripts.performance_tests_pp --grid 100 --steps 200 --repeat 3
  python -m scripts.performance_tests_pp --grid 100 --steps 200 --with-viz --repeat 3
  python -m scripts.performance_tests_pp --benchmark-neighbors --grid 200 --repeats 50
"""
import argparse
import time
import statistics
import os

import numpy as np


def run_pp_trial(rows, cols, steps, with_viz=False, open_window=False, interval=5, seed=42):
    # Importing here so we can set matplotlib backend before visualize if needed
    if with_viz and not open_window:
        # use Agg for headless rendering (still measures drawing cost)
        import matplotlib

        matplotlib.use("Agg")

    from models.CA import PP

    pp = PP(rows=rows, cols=cols, densities=(0.2, 0.1), neighborhood="moore", seed=seed)

    if with_viz:
        # enable visualization (will use Agg if open_window is False)
        pp.visualize(interval=interval, figsize=(10, 6), pause=0.001, show_cell_params=True)

    t0 = time.perf_counter()
    pp.run(steps)
    t1 = time.perf_counter()

    return t1 - t0


def benchmark_neighbors(rows, cols, repeats=100, seed=1):
    from models.CA import PP

    pp = PP(rows=rows, cols=cols, densities=(0.2, 0.1), neighborhood="moore", seed=seed)
    # fill some prey/predator randomly
    rng = np.random.default_rng(seed)
    flat = rng.choice(rows * cols, size=int(rows * cols * 0.3), replace=False)
    pp.grid.ravel()[flat[: int(len(flat) * 0.6)]] = 1
    pp.grid.ravel()[flat[int(len(flat) * 0.6) :]] = 2

    # Warm-up
    pp.count_neighbors()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pp.count_neighbors()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


def benchmark_sync_async(rows, cols, repeats=100, seed=1):
    """Benchmark one-step cost of update_sync vs update_async.

    For fairness we create two PP instances (synchronous and asynchronous)
    with the same initial grid distribution and then repeatedly restore
    that initial grid before calling the update method so each timed call
    starts from the same state.
    """
    from models.CA import PP

    pp_sync = PP(rows=rows, cols=cols, densities=(0.2, 0.1), neighborhood="moore", seed=seed, synchronous=True)
    pp_async = PP(rows=rows, cols=cols, densities=(0.2, 0.1), neighborhood="moore", seed=seed, synchronous=False)

    # Save an initial grid snapshot for restoring before each timed call
    grid0_sync = pp_sync.grid.copy()
    grid0_async = pp_async.grid.copy()

    # Warm-up a single call to avoid startup costs
    pp_sync.update_sync()
    pp_async.update_async()

    times_sync = []
    times_async = []
    for _ in range(repeats):
        pp_sync.grid[:] = grid0_sync
        t0 = time.perf_counter()
        pp_sync.update_sync()
        t1 = time.perf_counter()
        times_sync.append(t1 - t0)

        pp_async.grid[:] = grid0_async
        t0 = time.perf_counter()
        pp_async.update_async()
        t1 = time.perf_counter()
        times_async.append(t1 - t0)

    return times_sync, times_async


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=150)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--with-viz", action="store_true")
    p.add_argument("--open-window", action="store_true", help="Allow opening an interactive window for visualization")
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--benchmark-neighbors", action="store_true")
    p.add_argument("--benchmark-sync-async", action="store_true")
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()

    rows = cols = args.grid

    if args.benchmark_neighbors:
        print(f"Benchmarking count_neighbors on {rows}x{cols} for {args.repeats} iterations...")
        times = benchmark_neighbors(rows, cols, repeats=args.repeats)
        print(f"min {min(times):.6f}s mean {statistics.mean(times):.6f}s median {statistics.median(times):.6f}s max {max(times):.6f}s")
        return

    if args.benchmark_sync_async:
        print(f"Benchmarking update_sync vs update_async on {rows}x{cols} for {args.repeats} iterations...")
        ts_sync, ts_async = benchmark_sync_async(rows, cols, repeats=args.repeats)
        print("Synchronous update: ")
        print(f"  min {min(ts_sync):.6f}s mean {statistics.mean(ts_sync):.6f}s median {statistics.median(ts_sync):.6f}s max {max(ts_sync):.6f}s")
        print("Asynchronous update: ")
        print(f"  min {min(ts_async):.6f}s mean {statistics.mean(ts_async):.6f}s median {statistics.median(ts_async):.6f}s max {max(ts_async):.6f}s")
        return

    # Run trials: without visualization and with visualization (headless unless --open-window)
    no_viz_times = []
    viz_times = []
    print(f"Running {args.repeat} trials: grid={rows} steps={args.steps}")
    for i in range(args.repeat):
        print(f"Trial {i+1}/{args.repeat} (no viz)")
        t = run_pp_trial(rows, cols, args.steps, with_viz=False)
        no_viz_times.append(t)
        print(f"  time: {t:.3f}s, steps/sec: {args.steps / t:.2f}")

        if args.with_viz:
            print(f"Trial {i+1}/{args.repeat} (with viz, open_window={args.open_window})")
            t2 = run_pp_trial(rows, cols, args.steps, with_viz=True, open_window=args.open_window, interval=args.interval)
            viz_times.append(t2)
            print(f"  viz time: {t2:.3f}s, steps/sec: {args.steps / t2:.2f}")

    print("Summary:")
    print(f"No-viz: mean {statistics.mean(no_viz_times):.3f}s (steps/sec {args.steps / statistics.mean(no_viz_times):.2f})")
    if viz_times:
        print(f"With-viz: mean {statistics.mean(viz_times):.3f}s (steps/sec {args.steps / statistics.mean(viz_times):.2f})")


if __name__ == "__main__":
    main()
