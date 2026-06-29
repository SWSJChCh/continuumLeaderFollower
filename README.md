# Continuum leader–follower models of cranial neural crest migration

## Overview
This repository contains Python code implementing the continuum models described
in Chapter 6 of the PhD thesis:

Mathematical models of single and collective cell migration in the cranial neural crest

The code numerically solves continuum models for leader and follower cranial
neural crest cell (CNCC) migration in one spatial dimension. Two related models
are included:

1. A continuum cell–cell adhesion model, in which leaders migrate at a fixed
   speed and followers undergo diffusion and non-local attraction to leaders and
   other followers subject to a volume-filling constraint.

2. A continuum non-local alignment model, in which followers are structured by
   both spatial position and cell polarisation. Followers move according to their
   polarisation, diffuse in space and polarisation, and undergo non-local
   alignment of their polarisation with nearby leaders and followers.

The implementations are designed to reproduce the simulations and parameter
sweeps presented in Chapter 6 of the thesis.

### Code Author
- Samuel Johnson

### Date
- 09/01/2026

### Requirements
- numpy==2.2.4
- scipy==1.15.2
- numba

The cell–cell adhesion implementation relies on FFT-based convolution from
`scipy.signal`.

The non-local alignment implementation relies on Numba JIT compilation for the
explicit time-stepping loops.

The required libraries can be installed using pip:

```bash
pip install numpy==2.2.4 scipy==1.15.2 numba
```

### Code Structure

#### cellCellAdhesion.py

This script contains the full numerical implementation of the continuum
leader–follower cell–cell adhesion model.

The code includes:
- explicit forward Euler time integration
- integer-cell shifting algorithm for leader advection
- second-order finite-difference Laplacian with homogeneous Neumann boundary conditions
- conservative finite-volume upwind discretisation of the taxis flux
- non-local attraction terms evaluated using FFT-based linear convolution
- zero-flux boundary conditions for follower diffusion and taxis

#### nonLocalAlignment.py

This script contains the full numerical implementation of the continuum
leader–follower non-local alignment model.

The code includes:
- explicit forward Euler time integration
- accumulated-displacement integer-cell shifting algorithm for leader advection
- follower densities structured by physical position and polarisation
- spatial advection of followers with velocity determined by polarisation
- diffusion in both physical and polarisation space
- conservative upwind discretisation of spatial and polarisation advection fluxes
- non-local polarity alignment evaluated using a compact Gaussian interaction kernel
- zero-flux boundary conditions for follower spatial and polarisation fluxes
- sparse recording of the full follower–leader state
- Numba-accelerated time stepping

### Model Description

#### Cell–cell adhesion model

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

#### Non-local alignment model

Leaders:
- advect at fixed speed determined by v0 and the fixed leader polarisation pLead
- have fixed polarisation pLead = 1
- zero inflow at the left boundary and free outflow at the right boundary

Followers:
- are described by a density f(x, p, t), where x is physical position and p is
  cell polarisation
- move in physical space with velocity v0 p
- diffuse in physical space with diffusivity Df
- diffuse in polarisation space with diffusivity Dp
- align their polarisation non-locally with nearby leaders and followers

Non-local alignment:
- interactions are mediated by a truncated Gaussian kernel
- the alignment direction is calculated from a non-local weighted mean
  polarisation
- the polarisation drift is proportional to lambda
- the factor 1 - p^2 confines polarisation dynamics to the interval [-1, 1]

### Execution

#### Cell–cell adhesion model

The cell–cell adhesion code is executed from the command line, with parameters
given as command line arguments:

```bash
python cellCellAdhesion.py mu_fl mu_ff xi_fl xi_ff
```

- **mu_fl** - Follower–leader interaction strength (integer)
- **mu_ff** - Follower–follower interaction strength (integer)
- **xi_fl** - Follower–leader interaction length in microns (integer)
- **xi_ff** - Follower–follower interaction length in microns (integer)

#### Non-local alignment model

The non-local alignment code is executed from the command line without command
line arguments:

```bash
python nonLocalAlignment.py
```

The script runs a built-in parameter sweep over:

```python
lmbdVals = [2, 6, 10]
interactionVals = [10, 50, 100]
```

- **lambda** - Non-local polarisation-alignment strength
- **interaction length** - Alignment interaction length in microns

To change the parameter sweep, edit `lmbdVals` and `interactionVals` at the top
of `nonLocalAlignment.py`.

### Outputs

#### Cell–cell adhesion model

The simulation produces two text files:

Y_mu_fl*_mu_ff*_xi_fl*_xi_ff*.txt  
Contains the leader and follower density profiles at the specified output times.
The first N rows correspond to leader density and the next N rows correspond to
follower density.

T_mu_fl*_mu_ff*_xi_fl*_xi_ff*.txt  
Contains the time points at which the solution is recorded.

#### Non-local alignment model

For each parameter pair in the sweep, the simulation produces two text files:

solution_Y_numba_euler_lam_*_xi_*.txt  
Contains the recorded follower and leader states at the specified output times.
The first nx * nPol rows correspond to the follower density f(x, p, t), stored
in row-major order over physical position and polarisation. The final nx rows
correspond to the leader density.

solution_T_numba_euler_lam_*_xi_*.txt  
Contains the time points at which the solution is recorded.

For example, the parameter pair lambda = 2 and interaction length = 10 produces:

```text
solution_Y_numba_euler_lam_2_xi_10.txt
solution_T_numba_euler_lam_2_xi_10.txt
```
