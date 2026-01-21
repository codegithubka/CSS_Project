import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from scipy.integrate import solve_ivp
from scipy.integrate import odeint


class MeanFieldModel:
    """
    Mean-field (non-spatial) predator-prey model.

    Equations:
        dR/dt = R * (b - d_r - c*C - e*R)
        dC/dt = C * (a*R - d_c - q*C)

    where:
        R: Prey population density
        C: Predator population density
        b: Prey birth rate
        d_r: Prey death rate
        c: Consumption rate of prey by predators
        e: Intraspecific competition among prey
        a: Conversion efficiency of prey into predator offspring
        d_c: Predator death rate
        q: Intraspecific competition among predators
    """

    def __init__(
        self,
        birth: float = 0.2,
        consumption: float = 0.8,
        predator_death: float = 0.045,
        conversion: float = 1.0,
        prey_competition: float = 0.1,
        predator_competition: float = 0.05,
    ):
        """
        Initialize the mean-field model with given parameters.
        Args:
            birth (float): Prey birth rate (b)
            consumption (float): Consumption rate of prey by predators (c)
            predator_death (float): Predator death rate (d_c)
            conversion (float): Conversion efficiency of prey into predator offspring (a)
            prey_competition (float): Intraspecific competition among prey (e)
            predator_competition (float): Intraspecific competition among predators (q)
        """
        self.birth = birth
        self.consumption = consumption
        self.predator_death = predator_death
        self.conversion = conversion
        self.pred_benifit = self.consumption * self.conversion
        self.prey_competition = prey_competition
        self.predator_competition = predator_competition

    def ode_system(self, Z: np.ndarray, t: float, prey_death: float) -> list:
        """
        Mean-field ODE system for predator prey dynamics.
        """
        R, C = Z

        R = np.maximum(R, 0)
        C = np.maximum(C, 0)

        # Net prey growth rate
        r = self.birth - prey_death

        # Prey dynamics: growth - predation - competition
        dR = R * (r - self.consumption * C - self.prey_competition * R)

        # Predator dynamics: growth from predation - death - competition
        dC = C * (
            self.conversion * self.consumption * R
            - self.predator_death
            - self.predator_competition * C
        )

        return [dR, dC]

    def solve(
        self,
        prey_death: float = 0.5,
        R0: float = 0.5,
        C0: float = 0.2,
        t_max: float = 500,
        n_points: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the mean-field ODE system.

        Args:
            prey_death (float): Prey death rate (d)
            Z0 (Tuple[float, float]): Initial conditions (prey, predator)
            t_max (float): Maximum time
            n_points (int): Number of time points

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time points and solution array
        """
        t = np.linspace(0, t_max, n_points)
        Z0 = [R0, C0]

        sol = odeint(self.ode_system, Z0, t, args=(prey_death,))

        return t, sol

    def equilibrium(self, prey_death: float) -> Tuple[float, float]:
        """
        Calculate the equilibrium densities of the system.

        Args:
            prey_death (float): Prey death rate (d)

        Returns:
            Tuple[float, float]: Equilibrium populations (prey, predator)
        """
        r = self.birth - prey_death
        c = self.consumption
        a = self.pred_benifit
        e = self.prey_competition
        q = self.predator_competition
        d_c = self.predator_death

        if r <= 0:
            return (0.0, 0.0)

        R_prey = r / e

        # Check if predator can invade
        predator_invasion_fitness = a * R_prey - d_c
        if predator_invasion_fitness <= 0:
            return (R_prey, 0.0)  # Predator cannot persist

        # Coexistence equilibrium
        R_n = r * q + d_c * c
        R_d = c * a + e * q

        if R_d <= 0:
            return (R_prey, 0.0)

        R_star = R_n / R_d
        C_star = (a * R_star - d_c) / q

        if R_star < 0 or C_star < 0:
            if r > 0:
                return (R_prey, 0.0)
            else:
                return (0.0, 0.0)

        return (R_star, C_star)

    def equilibrium_numerical(
        self, prey_death: float, t_max: float = 1000
    ) -> Tuple[float, float]:
        """
        Find equilibrium densities numerically by solving ODEs over a long time.
        """
        t, Z = self.solve(prey_death=prey_death, t_max=t_max)
        R_eq = max(0, np.mean(Z[-100:, 0]))
        C_eq = max(0, np.mean(Z[-100:, 1]))
        return (R_eq, C_eq)

    def sweep_death_rate(
        self, d_r_values: np.ndarray, method: str = "analytical"
    ) -> Dict[str, np.ndarray]:
        """
        Sweep prey death rate and record equilibrium densities.
        """
        n = len(d_r_values)
        R_eq = np.zeros(n)
        C_eq = np.zeros(n)

        for i, d_r in enumerate(d_r_values):
            if method == "analytical":
                R_eq[i], C_eq[i] = self.equilibrium(d_r)
            else:
                R_eq[i], C_eq[i] = self.equilibrium_numerical(d_r)

        return {
            "d_r": d_r_values,
            "R_eq": R_eq,
            "C_eq": C_eq,
            "net_growth": self.birth - d_r_values,
        }

if __name__ == "__main__":
    print("Mean-Field Model Module")
    mf = MeanFieldModel()

    print("Model Parameters:")
    print(f"Birth rate: {mf.birth}")
    print(f"Consumption rate: {mf.consumption}")
    print(f"Predator death rate: {mf.predator_death}")
    print(f"Conversion efficiency: {mf.conversion}")
    print(f"Prey competition: {mf.prey_competition}")
    print(f"Predator competition: {mf.predator_competition}")

    d_r_values = np.linspace(0.01, 0.15, 50)
    results = mf.sweep_death_rate(d_r_values)
