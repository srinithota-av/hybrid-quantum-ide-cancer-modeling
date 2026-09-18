"""
Quantum-compatible pathway: discretisation → QUBO/Ising → QAOA (Section 3.3).
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

def build_ising_from_quadratic(Q, q, C=0.0):
    """Convert quadratic form to Ising Hamiltonian (paper mapping)."""
    n = len(q)
    pauli_list = [(f"Z{i}", float(q[i])) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Q[i, j]) > 1e-12:
                pauli_list.append((f"Z{i}Z{j}", float(Q[i, j])))
    return SparsePauliOp.from_list(pauli_list)

def run_qaoa(H_ising, p=2, shots=1024):
    """QAOA with the configuration given in the paper (4 qubits, p=2)."""
    optimizer = COBYLA(maxiter=100)
    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p)
    result = qaoa.compute_minimum_eigenvalue(H_ising)
    return result
