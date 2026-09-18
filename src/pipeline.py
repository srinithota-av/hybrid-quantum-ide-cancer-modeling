"""
Minimal end-to-end example corresponding to the paper’s workflow.
"""

import numpy as np
from .ide_model import solve_ide, positivity_bounds
from .parameter_estimation import estimate_parameters

def main():
    # True parameters used to generate synthetic data (as in the paper)
    theta_true = np.array([0.6, 0.3, 0.2, 0.5])
    t, T, M = solve_ide(theta_true, T0=0.3, t_span=(0, 20), h=0.05)

    # Add modest Gaussian noise
    rng = np.random.default_rng(42)
    T_obs = T + 0.01 * rng.standard_normal(len(T))

    # Parameter bounds used in the paper
    bounds = [(0.1, 1.0), (0.1, 1.0), (0.0, 0.5), (0.1, 1.0)]

    theta_est, cost = estimate_parameters(t, T_obs, bounds)
    print("Estimated parameters:", theta_est)
    print("Final cost J(θ):", cost)
    print("Positivity bounds:", positivity_bounds(theta_est))

if __name__ == "__main__":
    main()
