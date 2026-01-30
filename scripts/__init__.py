"""
Execution and Orchestration Suite for Predator-Prey Hydra Effect Research.

This module provides the entry points for running multi-phase simulation 
experiments. It is designed for high-performance computing (HPC) environments,
utilizing parallel processing and Numba-accelerated kernels to analyze 
self-organized criticality and finite-size scaling.

## Experimental Phases
The suite is divided into five sequential research phases:
1.  **Critical Point Identification**: Parameter sweeps to locate 
    bifurcation points and phase transitions.
2.  **Self-Organization Analysis**: Observations of evolutionary drift 
    toward critical states.
3.  **Finite-Size Scaling (FSS)**: Analysis of cluster size cutoffs 
    across varying grid dimensions ($L$).
4.  **Sensitivity Analysis**: Comprehensive 4D parameter sweeps across 
    survival regimes.
5.  **Directed Hunting Comparisons**: Sensitivity analysis using 
    non-random predator search kernels.

## Performance Features
- **Parallelization**: Automated CPU core detection and SLURM integration 
  via `joblib`.
- **Reproducibility**: Deterministic seed generation using SHA-256 hashing 
  of parameter states.
- **Incremental Persistence**: Results are saved in JSON Lines (JSONL) 
  format to ensure data recovery during long-running batches.

## Usage
Experiments should be invoked from the project root:
```bash
python scripts/experiments.py --phase 1 --output results/
"""

from models.numba_optimized import *