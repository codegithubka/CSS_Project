import cProfile, pstats
from pathlib import Path
import sys

# Ensure we can find our modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from pp_analysis import Config, run_single_simulation

# 1. Setup a single simulation configuration
cfg = Config()
cfg.default_grid = 150
cfg.warmup_steps = 200
cfg.measurement_steps = 300

# 2. Profile the function
profiler = cProfile.Profile()
profiler.enable()

# Run a single simulation (no parallelization)
run_single_simulation(0.2, 0.05, 150, 42, True, cfg)

profiler.disable()

# 3. Print the top 15 time-consumers
stats = pstats.Stats(profiler).sort_stats('tottime')
stats.print_stats(15)