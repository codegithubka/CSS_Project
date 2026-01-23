## Refactoring and Optimization Update (22/1)

---

### CA Class (```CA.py```)

Method: ```update_async``` method

This method is the primary interface for the async update mode in the PP model. Instead of iterating through the grid in Python, we prepare the data and leaves the heavy computations for the Numba kernel

Kernel execution: ```self._kernel.update(..)```  uses pre-allocated ```PPKernel``` to modify the gruid and parameter arrays on place.

We mutate the ```grid``` and the ```p_death_arr`` at the same time.

---

### Numba Optimization (```numba_optimization.py```)

Utility: ```set_numba_seed```

Synchronize the RNG used by Numba's JIT compiled functions with a specific seed. Becuase Numba compiles Python code into machine code, it maintains its own internal state for ```numpy.random```. We have global seeding this way under the ```@njit``` decorator

---

Core Logic: ```_pp_async_kernel_random```

This is the engine room of the sim. We do the stochastic async update of the entire grid in a single pass:

    1. Update Order with Fisher-Yates shuffle
    2. Cellular Interactions
    3. Evolutionary Inheritance

This function is called internally in the ```PPKernel.update``` method and is not intended to be called directly by the user.


We use the ```occupied_buffer``` which is a pre-allocated array. We recyle memory instead of creating a new list of active cells at each interation.

Core Logic: ```_pp_async_kernel_directed```

Similar to the previous function but also implements hunting logic. This kernel is activated by passing ```directed_hunting = True``` during the initialization of the ```PP``` model or the ```PPKernel``` wrapper

---

Class: ```PPKernel```

Handles the logistical overhead of the simulation.

- Memory pre-allocation: Initialize an ```occupied_buffer``` during setup. Used in each time step to store the coords of active cells. We don't need to constantly allocate and deallocate memory this way.

- Kernel dispatching: Directs simualtion data to to the appropriate specialized numba function (random or hunting)

---

Spatial Optimization ```_build_cell_list```

Oraganizes particles into a list of cells. We only compare particles to those in the same or adjacent cells.

---

Spatial Stats: ```_pcf_cell_list```

This kernel computes a histogram of distances between two sets of particles. It determines if species are clustered, segregated or randomly distributed.

If ```self_corrrelation = True``` it optimizes the process by only finding the upper triangle of the interaction matrix and doubling the result. We use ```prange``` to distribute the calculation across all available CPU cores.


---

Spatial Interpretation: ```compute_pcf_periodic_fast```

Wrapper that transforms the distance counts into a normalized PCF function. THis value represents the density of particles at a distance r relative to a completely random distrubution.

For the hydra effect, we look at the specific signature on the PCF.


---

Algorithm: ```_flood_fill```

Stack-based implementation that explores all neighborhoods of a starting cell. It marks cells as visited to avoid double counting and returns the total count of cells in a specific clsuter.

---

Kernel: ```_measure_clusters```

Iterate through the grid. When we find a cell of the target species that has not been visited yet, trigger a new flood fill to map out that entire cluster.

---

Wrapper: ```measure_cluster_sizes_fast```

Identify and measure every discrete cluster for a specific species on the grid. We ue a high speed flood fill to paint and count contiguous cells.


The hypothesis for the prey hydra effect is that increased mortality forces prey into tight more resilent clusters. We measure changes inm the average and max cluster size across a parameter sweep to test the spatial hypothesis.

---

### HPC Script Functionality (```pp_analysis.py```)

Class: ```Config```

Central configuration for analysis. Set grid dimensions, intial densities, resolution, death rate range, replications, warmup period, measurement window, stationarity, PCF sampling rates, mutation rate, fine size scaling grid sizes, and equilibrium scaling

Utility:```estimate_runtime```

HPC resource management tool. Estimates total simualtion runtime based on selected configurtion.

---

Data Packing Utility: ```save_sweep_binary```

This function transforms the in-memory Python objects into binary archive. We use array casting ```np.array(val)``` to store NumPy objects. 

Structural Reconstruction: ```load_sweep_binary```

Performs the inverse operatiion taking flat binary files and rebuilding the Python list of dictionaries required for plotting.

We use ```np.savez_compressed``` to reduce disk footprint for large arrays such as PCF results. 

---


Core Execution Unut: ```run_single_simulation```

Before updating the cell list, we set the seed and environment setup with RNG Synchronization, Model Instantiation and Evolution Activation.


We divide simulation into two stages to examine long-term behavior and avoid initialization bias. The model runs for ```cfg.warmup_steps``` and an additional ```cfg.measurement_steps``` to record data.

The script also measures cluster sizes using ```measure_cluster_sizes_fast``` and uses a ```pcf_sample_rate``` to claculate correlation functions on selected runs.

After each iteration, we flag survivals, evolved statistics, and clustering indices.

---

Finite size scaling Utility: ```run_fass```

Before running the scaling sweep, we perform a sanity check to make sure the system is in an intresting regime (meaning near a critical point).

We check if $\tau$ is near the critical value (2.05 approx.) after anchoring the simulation at a specific point in the parameter space.

Since larger systems usually take longer to each SS, we scale the warmup and measurement steps linearly. For each grid size, we record the mean population, the powetr law exponent and its SDE. 


--- 

Phase Space Utility: ```run_2d_sweep```

This is a high-throuput pipeline to generate raw data for boundary identification of the hydra effect and evo advantages.

We construct a massive list of individual sim jobs (tasks) to be executed in parallel.
Iterate through each combination of ```prey_births``` and ```prey_deaths``1.

    For each parameter coordinate, we create two jobs:
        1. Baseline (no evo)
        2. Experimental (with evo)

Before launchng the parallel threads, we warm up the kernels to avoid JIT (Just in Time) Numba Compilation overhead.

Each task calls ```run_single_simulation``` indepdendetly and returns a dict of results collected into a master list.

We compress the results efficiently using ```save_sweep_binary``` to keep a managable output size. We log the metatdat by generating a ```sweep_metadat.json```.

---


Plotting Utility: ```generate_plots```

We resahpe the flat list of results into a grid based on the ```prey_birth``` and ```prey_death``` parameters. We get the mean population across replicates for eevry point and filter out random noise of individual runs. To get derivatives effectively, we use a Gaussian filter to make the Hydra calculation reliable.

The Hydra effect is quantified as follows:
- Use a numerical gradient acorss the smoothed population grid
- Idenitfy the region where the derivative is positive that marks counter-intuitive   ecosystem dynamics.
- Compare evo with no-evo sets.

The function also computes spatial and criticality analysis as follows:
- Plot the power law exponent to show near phase transition regime
- Visualize PCF results. A low segregation index indicates predator and prey spatial separation, which indicates the Hydra effect in the CA model
- Overlay the Hydra boundary on top of the segreation heatmap to show correlation between spatial structure and population response (if existing)


We also calculate a relative advantage score to quantify the benifit of adaptation. We highlight regions where the baseline population went extinct but the evo population survived to show "evolutionary rescue".

----
Usage of the analysis script is recommended as follows:

bash
```
    python pp_analysis.py --mode full          # Run everything
    python pp_analysis.py --mode sweep         # Only 2D sweep
    python pp_analysis.py --mode sensitivity   # Only evolution sensitivity
    python pp_analysis.py --mode fss           # Only finite-size scaling
    python pp_analysis.py --mode plot          # Only generate plots from saved data
    python pp_analysis.py --mode debug         # Interactive visualization (local only)
    python scripts/pp_analysis.py --dry-run            # Estimate runtime without running
```

### Benchmark Results

The HPC script was optimized using JIT compilation.

Numba Kernel Accelaration:

- Speedup: 58.7x performance increase for a 50x50 grid
- Throughput: 1,216 steps per second

Spatial Metrics Refactoring:

- PCF: The cell-list algorithm resulted in a 562.5 speedup on a 75x75 grid
- Cluster metrics: Numba flood-fill algo gave us 24.6x speedup

Directed Hunting Overhead
- Negative ovderhead for larger grids (30-56% less!)

As a resultl, we can probably use a 1000x1000 grid for our HPC simulation!

## Testing and HPC Run Update (23/1)

HPC Run Estimate (we are using 32 cores).

1000 x 1000 grid -> 1 million cells

At each step: 500 million operations per simulation

This is multiplied by the number of replicates. 50 reps will result in 22,500 simulations.

By the benchmark, we have 1,182 steps per second (throuput) for a 100x100 grid. If we use a 1000x1000 grid, that implies 11.8 steps/second. So 1000x1000 grid with 50 reps

8.26 hours (not ideal!)

### Tests

```test_pp_analysis```

We have 58 test cases (one might fail to to messing with the grid size init default):

```
Run with:
    pytest test_pp_analysis.py -v
    pytest test_pp_analysis.py -v -x  # stop on first failure
```

Do not be alarmed by this if I forgot to change the test case to match the default value. The final grid size should be solidified by Friday.

```
def test_config_default_values(self, default_config):
        """Config should have sensible defaults."""
>       assert default_config.default_grid == 100
E       assert 1000 == 100

```
---

```test_numba_optimized```

Run with:
```
    pytest test_numba_optimized.py -v
    pytest test_numba_optimized.py -v --tb=short  # shorter traceback
    python test_numba_optimized.py  # without pytest
```


We have 48 tests cases validating the folloiwing:

- Imports
- Kernel Initialization
- Buffer allocation
- Async updates
- Evolution
- Directed and undirected kernel methods
- PCF behavior
- Cluster metrics
- Warmup
- Edge cases with extreme parameter values


## Issues to be resolved

1. Grid size for HPC run
2. Number of replicates for statistal power
3. Directed and/or undirected runs 
4. Evolving and non-evolving runs?
5. Mean field baseline or non evolving basiline
6. Warmup period and measurement steps (i.e how many steps do we need to avoid init bias?)
7. Measurement frequency for statistical accuracy
8. Default parameters (Need Storm's input on this one).

Options:

1. Asymmetric repliates for non-evolving runs
2. Coarse initial parameter sweep grid
3. Discard non-evo runs and use mean field baseline instead or the opposite

NOTE: Without the optimization kernels for a 1000x1000 grid the simulation (using 50 reps for statistical power) would run for 548 hours (approximately 23 days)






