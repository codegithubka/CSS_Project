## Predator-Prey Cellular Automaton

### Overview

This project impelments a spatial predator-prey cellular automaton (CA) to investigate the Hydra effect, a counterintuitive ecological phenomenon where increased prey mortality paradoxically leads to higher prey population densities. The model explores the rise of emergent population dynamics through spatial structure and local interactions that differe fundemntally from well-mixed (mean-field) predictions.

The codebase uses Numba JIT compilation for computationally intensive kernels and is tuned for high performance computing environments. It supporrs parameter sweeps, finite size scaling analysis, and self-organized criticality exploration.

---

### Project Structure

The repository is organized to separate model logic, high-performance execution scripts, and data analysis:

```text
.
├── models/               # Core simulation logic
│   ├── CA.py             # Base Cellular Automaton class
│   ├── config.py         # Phase-specific experiment configurations
│   └── numba_optimized.py # JIT kernels and cluster detection
├── scripts/              # HPC execution scripts
│   └── run_phase{1..5}.sh # Bash scripts for Slurm/SGE jobs
├── notebooks/            # Data analysis and visualization
│   └── plots.ipynb       # Results plotting and Hydra effect analysis
├── tests/                # Pytest suite for model validation
├── data/                 # Local storage for simulation outputs (JSONL)
└── requirements.txt      # Project dependencies
```
---

### Background

#### The Hydra Effect

In classical Lotka-Volterra dynamics, increasing prey mortality always reduces the prey population. However, theoretical work has identified conditions where the opposite effect can be observed. The Hydra effect emerges in spatially structured systems where

1. Local interactions prevent predators from immediately exploiting all available prey.
2. Spatial refugia from naturally through predator-prey dynamics
3. Increased prey mortality can disrupt predator populations disproportionally.

This study uses a cellular automaton framework toi study how spatial strcuture generates the Hydra effecr and whether the system exhibits signatures of self-organized criticality at the transition point.

#### Self-Organized Criticality (SOC)

SOC refers to systems that naturally evolve toward a critical state without external tuning. At criticality, such systems exhibit:

- Power-law cluster size distributions: $P(s) \sim s^{-\tau}$
- Scale-free correlations: No characteristic length scale dominates
- Critical slowing down: Perturbations decay slowly near the critical point
- Fractal spatial patterns: Self-similar structure across scales

In the predator-prey context, SOC would manifest as the system self-tuning toward a critical prey mortality rate where population dynamics become scale-free and the Hydra effect is maximized.

---

### Model Description

The model uses a 2D lattice with periodic boundary conditions. Each cell occupies one of three states:

| State | Value | Description |
|-------|-------|-------------|
| Empty | 0 | Unoccupied cell |
| Prey | 1 | Prey organism |
| Predator | 2 | Predator organism |

The model uses asynchronous updates: cells are processed in random order each timestep, with state changes taking effect immediately. This prevents artificial synchronization artifacts common in parallel update schemes.

For each prey cell, in order:

- Death: With probability `prey_death`, the prey dies and the cell becomes empty
- Reproduction: If alive and a randomly selected neighbor is empty, with probability `prey_birth`, a new prey is placed in that neighbor cell

For each predator cell, in order:

- Death: With probability `predator_death`, the predator dies (starvation)
- Hunting/Reproduction: If alive and a randomly selected neighbor contains prey, with probability `predator_birth`, the predator consumes the prey and reproduces into that cell

The model supports both neighborhood types:

- Moore neighborhood: (default): 8 adjacent cells (including diagonals)
- von Neumann neighborhood: 4 adjacent cells (cardinal directions only)

Moore neighborhoods are used throughout the experiments as they provide more realistic movement and interaction patterns.

---

### Hunting Modes

The model implements two distinct neighbor selection strategies that qualitatively affect dynamics.

#### Random Neighbor Selection (Default)

In the standard mode, each organism selects a single random neighbor for interaction:

```python
# Prey: pick random neighbor, reproduce if empty
neighbor_idx = np.random.randint(0, n_neighbors)
if grid[neighbor] == EMPTY and random() < prey_birth:
    grid[neighbor] = PREY

# Predator: pick random neighbor, hunt if prey
if grid[neighbor] == PREY and random() < predator_birth:
    grid[neighbor] = PREDATOR
```
This creates a blind interaction model where organisms are not aware of their surroundings.

#### Directed Hunting Mode

The directed mode implements "intelligent" neighbor selection:

- Prey: Scan all neighbors for empty cells, then randomly select one empty cell for reproduction
- Predators: Scan all neighbors for prey, then randomly select one prey cell to hunt

```python
# Directed prey reproduction
empty_neighbors = [n for n in neighbors if grid[n] == EMPTY]
if empty_neighbors and random() < prey_birth:
    target = random.choice(empty_neighbors)
    grid[target] = PREY

# Directed predator hunting
prey_neighbors = [n for n in neighbors if grid[n] == PREY]
if prey_neighbors and random() < predator_birth:
    target = random.choice(prey_neighbors)
    grid[target] = PREDATOR
```

This increases the effective reproduction and predation rates without changing the nominal probability parameters.

---

### Implementation Architecture

#### Class Hierarchy

```
CA (base class)
 └── PP (predator-prey specialization)
      └── PPKernel (Numba-optimized update kernel)
```

The `CA` base class provides:
- Grid initialization with specified densities
- Neighborhood definitions (Moore/von Neumann)
- Per-cell parameter evolution framework
- Validation and run loop infrastructure

The `PP` class adds:
- Predator-prey specific parameters and defaults
- Synchronous/asynchronous update dispatch
- Integration with Numba-optimized kernels

#### Basic Usage

```python
from models.CA import PP

# Initialize model
model = PP(
    rows=100,
    cols=100,
    densities=(0.30, 0.15),  # (prey_fraction, predator_fraction)
    params={
        "prey_birth": 0.2,
        "prey_death": 0.05,
        "predator_birth": 0.8,
        "predator_death": 0.05,
    },
    seed=42,
    synchronous=False,  # Always False for this study
    directed_hunting=False,
)

# Run simulation
for step in range(1000):
    model.update()
    
# Access grid state
prey_count = np.sum(model.grid == 1)
pred_count = np.sum(model.grid == 2)
```

#### Evolution Mode

The model supports per-cell parameter evolution, where offspring inherit (with mutation) their parent's parameter values:

```python
# Enable evolution of prey_death parameter
model.evolve(
    "prey_death",
    sd=0.01,      # Mutation standard deviation
    min_val=0.0,  # Minimum allowed value
    max_val=0.20, # Maximum allowed value
)
```

When evolution is enabled, each prey cell maintains its own `prey_death` value. Upon reproduction, offspring inherit the parent's value plus Gaussian noise. This allows investigation of whether the population self-organizes toward a critical mortality rate.

---

### Numba Optimization

The computational bottleneck is the update kernel, which must process every occupied cell each timestep. For a 1000×1000 grid with 50% occupancy, this means ~500,000 cell updates per step.

Key optimizations:

- **JIT Compilation**: Core kernels use `@njit(cache=True)` for ahead-of-time compilation
2. **Pre-allocated Buffers**: The `PPKernel` class maintains reusable arrays to avoid allocation overhead
3. **Efficient Shuffling**: Fisher-Yates shuffle implemented in Numba for random cell ordering
4. **Cell Lists for PCF**: Pair correlation functions use spatial hashing for O(N) instead of O(N²) complexity

#### PPKernel Class

```python
class PPKernel:
    """Wrapper for predator-prey kernel with pre-allocated buffers."""
    
    def __init__(self, rows, cols, neighborhood="moore", directed_hunting=False):
        self.rows = rows
        self.cols = cols
        self.directed_hunting = directed_hunting
        self._occupied_buffer = np.empty((rows * cols, 2), dtype=np.int32)
        
        # Neighbor offset arrays
        if neighborhood == "moore":
            self._dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
            self._dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        else:
            self._dr = np.array([-1, 1, 0, 0], dtype=np.int32)
            self._dc = np.array([0, 0, -1, 1], dtype=np.int32)
```
---

### Cluster Detection

Clusters are contiguous groups of same-species cells (using Moore connectivity). The implementation uses stack-based flood fill with periodic boundary conditions.

```python
from models.numba_optimized import get_cluster_stats_fast

stats = get_cluster_stats_fast(grid, species=1)  # Prey clusters
print(f"Number of clusters: {stats['n_clusters']}")
print(f"Largest cluster: {stats['largest']} cells")
print(f"Largest fraction: {stats['largest_fraction']:.3f}")
```

Metrics collected:
- Cluster size distribution: Power-law indicates criticality
- Largest cluster fraction: Order parameter for percolation transition
---

### Configuration System

The `Config` dataclass centralizes all experimental parameters:

```python
from config import PHASE1_CONFIG, Config

# Use predefined phase config
cfg = PHASE1_CONFIG

# Or create custom config
cfg = Config(
    grid_size=500,
    n_replicates=20,
    warmup_steps=1000,
    measurement_steps=2000,
    collect_pcf=True,
    pcf_sample_rate=0.5,
)

# Access parameters
print(f"Grid: {cfg.grid_size}x{cfg.grid_size}")
print(f"Estimate: {cfg.estimate_runtime(n_cores=32)}")
```

Each phase has a predefined configuration (`PHASE1_CONFIG` through `PHASE5_CONFIG`) with appropriate defaults for that analysis.

---

### Output Format

Results are saved in JSONL format (one JSON object per line) for efficient streaming and parallel processing:

```json
{"prey_birth": 0.2, "prey_death": 0.05, "seed": 12345, "final_prey": 28500, ...}
{"prey_birth": 0.2, "prey_death": 0.06, "seed": 12346, "final_prey": 27200, ...}
```

Each result dictionary contains:
- Input parameters
- Final population counts
- Cluster statistics
- Evolution statistics (if enabled)
- Time series (if enabled)

Metadata files (`phase{N}_metadata.json`) accompany each results file with configuration details and runtime information.

---

### Testing

The project includes a pytest test suite covering all core modules.

#### Test Modules

| File | Coverage |
|------|----------|
| `tests/test_ca.py` | CA base class, PP model initialization, update mechanics, evolution, edge cases |
| `tests/test_numba_optimized.py` | Cluster detection, PCF computation, PPKernel updates, performance |
| `tests/test_experiments.py` | Utility functions, I/O operations, simulation runner, phase registration |
| `tests/test_config.py` | Configuration defaults, phase configs, helper methods |

#### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ca.py -v

# Run with coverage report
pytest tests/ --cov=models --cov-report=html

# Fast mode (stop on first failure)
pytest tests/ -x --tb=short
```

### Documentation

Full API documentation is available at: **[https://codegithubka.github.io/CSS_Project/](https://codegithubka.github.io/CSS_Project/)**


#### Generating Docs Locally
```bash
# Generate HTML documentation
pdoc --output-dir docs --docformat numpy --no-include-undocumented \
    models.CA models.config models.numba_optimized experiments.py

# View locally
open docs/index.html
```

Documentation is auto-generated from NumPy-style docstrings using [pdoc](https://pdoc.dev/).


### Getting Started

#### 1. Dependencies

Required:
- Python 3.8+
- NumPy
- Numba (for JIT compilation)
- tqdm (progress bars)
- joblib (parallelization)

Optional:
- matplotlib (visualization)
- scipy (additional analysis)

#### 2. Installation
Clone the repository and install the dependencies. It is recommended to use a virtual environment.

```bash
# Install dependencies
pip install -r requirements.txt
```

#### 3. Running simulations

The experiments are automated via bash-scripts in the ```scripts``` directory. These are configured for high-performance computing environments:

```bash
# Grant execution permissions
chmod +x scripts/*.sh

# Execute a specific phase (e.g., Phase 1)
./scripts/run_phase1.sh
```
