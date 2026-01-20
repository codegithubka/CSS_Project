import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from models.CA import PP


def test_visualize_headless_runs():
    pp = PP(rows=30, cols=30, densities=(0.2, 0.05), neighborhood='moore', seed=1)
    pp.evolve('prey_death', sd=0.01, min=0.001, max=0.1)
    # should not raise
    pp.visualize(interval=1, figsize=(4, 4), pause=0.001, show_cell_params=True, show_neighbors=False)
    pp.run(3)
    plt.close('all')
