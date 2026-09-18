"""
Nonlinear least-squares parameter estimation (Eq. 8).
"""

import numpy as np
from scipy.optimize import least_squares
from .ide_model import solve_ide

def residual(theta, t_obs, T_obs, T0=0.3):
    _, T_sim, _ = solve_ide(theta, T0=T0, t_span=(t_obs[0], t_obs[-1]), h=t_obs[1]-t_obs[0])
    T_sim_interp = np.interp(t_obs, np.linspace(t_obs[0], t_obs[-1], len(T_sim)), T_sim)
    return T_sim_interp - T_obs

def estimate_parameters(t_obs, T_obs, bounds, theta0=None, method="trf"):
    if theta0 is None:
        theta0 = np.array([0.5, 0.3, 0.2, 0.5])
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    res = least_squares(residual, theta0, args=(t_obs, T_obs),
                        bounds=(lb, ub), method=method, verbose=0)
    return res.x, res.cost
