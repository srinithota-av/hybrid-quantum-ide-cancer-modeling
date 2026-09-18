# Hybrid Quantum-Assisted Integro-Differential Framework for Cancer Progression Modeling

Code companion to the paper:

> Srinivasarao Thota & Boukabcha Brahim,  
> “A Hybrid Quantum-Assisted Integro-Differential Framework for Cancer Progression Modeling and Clinical Decision Support”.

## Overview
This repository implements the modular framework described in the paper:

1. Nonlinear Volterra integro-differential equation (IDE) with exponential memory kernel for tumour-burden dynamics.
2. Equivalent auxiliary ODE system that preserves non-negativity and boundedness.
3. Numerical discretization (trapezoidal + predictor-corrector) and nonlinear least-squares parameter estimation.
4. Quantum-compatible pathway: bounded parameter discretization → quadratic surrogate → QUBO/Ising → QAOA (Qiskit).
5. Hybrid CNN–QNN image classifier (ResNet-18 backbone + shallow variational quantum circuit) with fusion of simulation-derived mechanistic features.
6. Ablation study configurations M1–M5 and statistical evaluation utilities.

**Important disclaimer**  
This is a reproducible computational proof-of-concept.  
- IDE parameters are recovered from *synthetic* trajectories only.  
- Image features come from public benchmarks (BreakHis, Brain Tumor MRI).  
- No patient-specific longitudinal data or clinical claims are made.  
- Quantum experiments use the ideal Aer simulator; no quantum advantage is claimed.

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/hybrid-quantum-ide-cancer-modeling.git
cd hybrid-quantum-ide-cancer-modeling
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
