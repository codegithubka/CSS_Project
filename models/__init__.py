"""
# Models Package

This package contains the core simulation logic for the Predator-Prey 
Cellular Automata.

## Key Components
- `CA`: The base Cellular Automata class.
- `experiment`: Tools for running batches and collecting data.
- `numba_optimized`: High-performance kernels for HPC execution.

## Example
```python
from models.CA import PredatorPreyCA
sim = PredatorPreyCA(rows=100, cols=100)
"""

from .CA import CA, PP

