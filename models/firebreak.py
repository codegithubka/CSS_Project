""""
Altruistic Suicide Firebreak Model

Concept:

- Standard Prey: Dies of old age or being eaten
- Firebreak prey: Has a sensor to detect if a predator is nearby. If so, then it has
    the option to protect the colony by commiting suicde before the predator can reproduce
    into it
- Result: Creates empty cells around the predator cluster. Prevents percolation through
the cluster


TO DO:

1. Override update_sync and update_async logic.
    New Logic:
        1. Calculate predator_neighbors for every cell
        2. Create a death rate grid where every cell gets assigned a death rate based on neighbors
        3. Calc the mask against the dynamic grid
"""

import numpy as np
from models.CA import PP


class Firebreak(PP):
    """
    PP CA where prey commit suicide when threatend to creaty empty firebreaks
    that starve predators
    """

    def __init__(
        self,
        rows,
        cols,
        densities,
        neighborhood="moore",
        params=None,
        cell_params=None,
        seed=None,
        synchronous=True,
    ):
        # get firebreak specific parameter
        self.alt_dr = 0.5  # FIXME: Random default value
        clean_params = dict(params) if params else {}
        if "altruistic_dr" in clean_params:
            self.alt_dr = float(clean_params.pop("altruistic_dr"))

        super().__init__(
            rows,
            cols,
            densities,
            neighborhood,
            clean_params,
            cell_params,
            seed,
            synchronous,
        )
        self.params["altruistic_dr"] = self.alt_dr

    def update_sync(self) -> None:
        """Override syncrhonous update. Similar to PP update except Prey Death Logic."""
        pass
