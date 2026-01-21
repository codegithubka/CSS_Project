"""Run and visualize the PP model with evolving prey death rates.

Creates a 250x250 grid, enables per-cell evolution for `prey_death`, and
visualizes the grid every 5 iterations while running for 2500 steps.
"""
from models.CA import PP


def main():
    params = {
        "prey_birth": 0.2,
        "prey_death": 0.05,
        "predator_birth": 0.8,
        "predator_death": 0.045,
    }

    pp = PP(
        rows=250,
        cols=250,
        densities=(0.2, 0.05),
        neighborhood="moore",
        params=params,
        cell_params=None,
        seed=12345,
        synchronous=True,
    )

    # Enable per-cell evolution for prey death rates. Use a small sd and
    # reasonable clipping bounds so values remain in (0.001, 0.2).
    pp.evolve("prey_death", sd=0.1, min_val=0.001, max_val=0.2)

    # Start interactive visualization: update every 10 iterations
    # Do not show neighbor histogram/percentile plots to reduce overhead
    pp.visualize(interval=10, figsize=(12, 8), pause=0.1, show_cell_params=True, show_neighbors=True)

    # Run the simulation (ensure the plot stays open afterwards)
    try:
        pp.run(2500)
    finally:
        # Block and show the final figure so the user can inspect it.
        # Turn off interactive mode (visualize() enabled it) and show blocking.
        import matplotlib.pyplot as plt

        try:
            plt.ioff()
        except Exception:
            pass
        plt.show(block=True)


if __name__ == "__main__":
    main()
