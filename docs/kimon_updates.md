## Refactoring and Optimization Update

---

### CA Class (Simulation Engine)

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

Class: ```Config``

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

### Tests
