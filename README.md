# Continuum leader–follower model of cranial neural crest migration

## Overview
This repository contains Python code implementing the continuum cell–cell adhesion
model described in Chapter 6 and Appendix A of the PhD thesis:

Mathematical models of single and collective cell migration in the cranial neural crest

The code numerically solves a coupled system of non-local advection–diffusion PDEs
for leader and follower cranial neural crest cell (CNCC) densities in one spatial
dimension. Leaders migrate at a fixed speed, while followers undergo diffusion and
non-local attraction to both leaders and other followers subject to a volume-filling
constraint.

The implementation is designed to reproduce the simulations and parameter sweeps
presented in Chapter 6 of the thesis.

### Code Author
- Samuel Johnson

### Date
- 09/01/2026

### Requirements
- numpy==2.2.4
- scipy==1.15.2

In particular, the implementation relies on FFT-based convolution from
scipy.signal.

The required libraries can be installed using pip:

```bash
pip install numpy==2.2.4 scipy==1.15.2
```

### Code Structure

cellCellAdhesion.py

This script contains the full numerical implementation of the continuum
leader–follower cell–cell adhesion model.

The code includes:
- explicit forward Euler time integration
- integer-cell shifting algorithm for leader advection
- second-order finite-difference Laplacian with homogeneous Neumann boundary conditions
- conservative finite-volume upwind discretisation of the taxis flux
- non-local attraction terms evaluated using FFT-based linear convolution
- zero-flux boundary conditions for follower diffusion and taxis

### Model Description

Leaders:
- advect at constant speed v0
- zero inflow at the left boundary and free outflow at the right boundary

Followers:
- diffuse with diffusivity Df
- undergo non-local attraction to leaders and other followers
- subject to a volume-filling constraint with carrying capacity kappa

Non-local interactions:
- truncated Gaussian kernels with interaction lengths xi_ff and xi_fl
- linear convolution with zero extension beyond the computational domain

### Execution

The code is executed from the command line, with parameters given as command line arguments:

```bash
python cellCellAdhesion.py mu_fl mu_ff xi_fl xi_ff
```

- **mu_fl** - Follower–leader interaction strength (integer)
- **mu_ff** - Follower–follower interaction strength (integer)
- **xi_fl** - Follower–leader interaction length in microns (integer)
- **xi_ff** - Follower–follower interaction length in microns (integer)


### Outputs

The simulation produces two text files:

Y_mu_fl*_mu_ff*_xi_fl*_xi_ff*.txt  
Contains the leader and follower density profiles at the specified output times.
The first N rows correspond to leader density and the next N rows correspond to
follower density.

T_mu_fl*_mu_ff*_xi_fl*_xi_ff*.txt  
Contains the time points at which the solution is recorded.
