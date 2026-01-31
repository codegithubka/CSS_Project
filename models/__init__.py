"""
Predator-Prey Cellular Automaton Models
=======================================

This package provides a Numba-accelerated cellular automaton framework
for simulating predator-prey dynamics with spatial structure.

Main Components
---------------
- `PP` : Predator-Prey cellular automaton model
- `Config` : Configuration dataclass for experiments
- `PPKernel` : Low-level Numba-optimized update kernel

Spatial Analysis
----------------
- `get_cluster_stats_fast` : Comprehensive cluster statistics
- `detect_clusters_fast` : Cluster detection with labels
- `measure_cluster_sizes_fast` : Fast cluster size measurement
- `compute_all_pcfs_fast` : Pair correlation functions

Example
-------
```python
from models import PP, Config

# Create a model with default parameters
model = PP(rows=100, cols=100, seed=42)

# Run simulation
for _ in range(1000):
    model.update()

# Or use the run method
model.run(steps=1000)
```

For experiments, use the configuration system:

```python
from models import Config, get_phase_config

# Use predefined phase config
cfg = get_phase_config(1)

# Or create custom config
cfg = Config(grid_size=200, n_replicates=10)
```
"""

# Core model classes
from models.CA import CA, PP

# Configuration
from models.config import *

# Numba-optimized components
from models.numba_optimized import *

__all__ = [
    # Core
    "CA",
    "PP",
    # Config
    "Config",
    "get_phase_config",
    "PHASE_CONFIGS",
    "PHASE1_CONFIG",
    "PHASE2_CONFIG",
    "PHASE3_CONFIG",
    "PHASE4_CONFIG",
    "PHASE5_CONFIG",
    # Numba kernel
    "PPKernel",
    # Cluster analysis
    "measure_cluster_sizes_fast",
    "detect_clusters_fast",
    "get_cluster_stats_fast",
    # PCF analysis
    "compute_pcf_periodic_fast",
    "compute_all_pcfs_fast",
    # Utilities
    "set_numba_seed",
    "warmup_numba_kernels",
    "NUMBA_AVAILABLE",
]

__version__ = "1.0.0"

from .CA import CA, PP

