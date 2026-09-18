"""
Memory-dependent tumour progression model (Eqs. 1–5 of the paper).
"""

import numpy as np
from scipy.integrate import solve_ivp

def exponential_kernel_system(t, y, a, b, c, lam):
    """Coupled ODE system (5)."""
    T, M = y
    dT = a * T - b * T**2 - c * T * M
    dM = T - lam * M
    return [dT, dM]

def solve_ide(theta, T0=0.3, t_span=(0.0, 20.0), h=0.01, method="RK45"):
    """Solve the auxiliary system for parameters theta = (a, b, c, λ)."""
    a, b, c, lam = theta
    t_eval = np.arange(t_span[0], t_span[1] + h, h)
    sol = solve_ivp(
        exponential_kernel_system,
        t_span,
        [T0, 0.0],
        args=(a, b, c, lam),
        t_eval=t_eval,
        method=method,
        rtol=1e-8,
        atol=1e-8,
    )
    return sol.t, sol.y[0], sol.y[1]

def positivity_bounds(theta, T0=0.3):
    """Proposition 1: a-priori bounds."""
    a, b, _, lam = theta
    Tmax = max(T0, a / b)
    Mmax = Tmax / lam
    return Tmax, Mmax
