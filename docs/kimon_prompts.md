### Mean Field class

1. Create a baseline mean-field class based on the attached research paper on predator-prey dynamics. The class should adhere to the papers specifications. The class should have a parameter sweep method for key predator and prey parameters that will be run in Snellius. Also include a method for equilibrium analysis. Make sure to justify the logic for this method. Include docstrings with a small method description and comments for code interpretability.

2. Justify initialization parameter values for a small test expiriment. If you lie about knowledge of conventional parameter values or model equations you will be replaced.

3. Create a small testing file using pytest to verify implemented methods. Make sure to cover edge cases and list them after the .py file output for me please. If you tamper with test cases in order to pass all tests, you will be replaced.

4. We are now ready to plot some of the results of the mean fielf baseline. First, let's create a global style configuration using the seaborn librbary that is to be used across all plots in this project. Make sure the legend is at the bottom of each plot.

5. Plot the phase portait to confirm the system spiral into a stable point. Show the nullclines as well. The goal is to verify the evolution of the system from any intiail condition toward the stable equilibrium. 

6. Create a time series analysis plot of the evolution of prey and predator density vs. time. Make sure enough time steps all visible to see how the system eventually stabilizes.

7. Create a bifuracation diagram to confirm the monotonic relationship for a varying prey death rate vs. equilibrium density. 

---

### Testing CA class

1. Create a comprehensive testing suite for the CA and PP classes. Test initialization, async update changes, synchronous update changes, prey growth in isolation behavior, predator starvation, parameter evolution and long run dynamics. Also make sur ethe test_viz mehtod works as desired

---

### Parameter Sweep and PP Class Analysis

2. Create a skeletal version of a .py script that will be subimtted into Snellius for parameter analysis. The purpose of this script should be to identify power law distribution with the consideration of finite size scaling, the hydra effect, and suitable parameter configurtaions for time series analysis for model evolution. Compare the non-evo to the evo model.

3. Create a config class adjustable depending on the CPU budget. We want to run a prey_birth vs. predator_death parameter sweep (2D), quantify the hydra effect using the derivative, search for the critical point (power law relartionship paramameter), quantify evolution sensitivity and analyze finite grid size scaling. Include script options for cost optimal runs as well. Make sure to have a summary of collected data stored for reference and future usage.


4. Add configuration option to run the asynchronous version of the CA class. The synchronous functionality should also be preserved. Provide me with a small smoke test to see if the updated file runs as expected.

5. Create a minimal bash script for Snellius. Use the rome configiuration.

6. Fix predator-prey analysis script so that the hydra effect focuses on the prey hydra effect as a result of the increasing prey death rate.


7. Add PCF analysis functonality for prey auto, predator auto and cross correlation. Also, integrate the snapshot method from the CA clas as an optional functionality of the analysis module. Add the folowing plots: 1. phase diagrams showing segregation, prey-clusterin, and predator clustering. Scatter plots tetsing if Hydra effect correlates with spatial segregation, and CA-style snapshots, neighbor histogram, and evolution trajectory.


8. Help me create a testing module for the analysis file. Use unittest.mock to create a mock model for testing. If you lie or falsify tests so that they pass my script, you will be replaced.


9. Add a larger scale simulation in the testing file to verify plots are as desired.

---

### Script Optimization\


1. I am considering using numba for optimization and faster runs in the HPC. Outline an implementation plan, practical considerations, and feasibility within a logical timeframe.

2. Walk me through modifying tg