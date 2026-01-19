### Create base CA class

Create a cellular automaton class named CA with an init function, a count neighbors function, an update function, and a run. The CA should consist of a numpy array called grid, a string containing the neighborhood type, and a numpy random number generator called generator. Use this generator for all random number generation inside the class. The CA class should also contain a dictionary called params.

The init function should take arguments for the grid size (rows, columns, both ints), the initial density of each species (a tuple of floats), the neighborhood type ("neumann" or "moore"), the parameters in the form of a dictionary, and the seed for the generator. It should assign the parameters dictionary to the params variable, create the generator object and assign it to the generator variable, as well as create the 2D array of zeros based on the grid size and assign it to the grid variable. This grid should then be filled with states dependent on the density tuple. Iterate over the elements i of this tuple, filling grid_size * density[i] elements of the grid with state i+1. Non-zero cell states should not be overwritten, ensuring that the specified percentage of the grid is filled with that state. It should also check if the neighborhood argument corresponds with a known neighborhood and return an error otherwise.

The count neighbors function should return a tuple of matrices (one for each defined non-zero state) containing the amount of neighbors of that state for each cell. It should use the neighborhood defined in the class. Ensure the logic works for both "neumann" and "moore". Use periodic boundaries.

The update function can remain empty, so fill it with a pass statement.

The run function should take a steps (int) argument. It should then run the CA for steps interations, calling the update function each time.

Finally, make sure to add an expected type for each argument and define the return types. Add this information, as well as a short description of the function to the docstring. Also add assert statements to ensure arguments "make sense". For example, the sum of densities should not exceed 1 and the rows, cols, densities should all be positive, and the neighborhood should be either "neumann" or "moore".


### Mean Field class

1. Create a baseline mean-field class based on the attached research paper on predator-prey dynamics. The class should adhere to the papers specifications. The class should have a parameter sweep method for key predator and prey parameters that will be run in Snellius. Also include a method for equilibrium analysis. Make sure to justify the logic for this method. Include docstrings with a small method description and comments for code interpretability.

2. Justify initialization parameter values for a small test expiriment. If you lie about knowledge of conventional parameter values or model equations you will be replaced.

3. Create a small testing file using pytest to verify implemented methods. Make sure to cover edge cases and list them after the .py file output for me please. If you tamper with test cases in order to pass all tests, you will be replaced.

4. We are now ready to plot some of the results of the mean fielf baseline. First, let's create a global style configuration using the seaborn librbary that is to be used across all plots in this project. Make sure the legend is at the bottom of each plot.

5. Plot the phase portait to confirm the system spiral into a stable point. Show the nullclines as well. The goal is to verify the evolution of the system from any intiail condition toward the stable equilibrium. 

6. Create a time series analysis plot of the evolution of prey and predator density vs. time. Make sure enough time steps all visible to see how the system eventually stabilizes.

7. Create a bifuracation diagram to confirm the monotonic relationship for a varying prey death rate vs. equilibrium density. 

###