# PolyKin

[![Test](https://github.com/HugoMVale/polykin/actions/workflows/test.yml/badge.svg)](https://github.com/HugoMVale/polykin/actions)
[![codecov](https://codecov.io/gh/HugoMVale/polykin/branch/main/graph/badge.svg?token=QfqQLX2rHx)](https://codecov.io/gh/HugoMVale/polykin)
[![Latest Commit](https://img.shields.io/github/last-commit/HugoMVale/polykin)](https://img.shields.io/github/last-commit/HugoMVale/polykin)

PolyKin is an open-source polymerization kinetics library for Python. It is still at an early
development stage, but the following modules can already be used:

- Activity coefficient models
  - [x] Ideal solution
  - [x] Flory-Huggins
  - [x] NRTL
  - [ ] Poly-NRTL
  - [x] UNIQUAC
  - [x] Wilson
- Copolymerization
  - [x] Implicit penultimate model
  - [x] Penultimate model
  - [x] Terminal model
  - [x] Mayo-Lewis equation (binary, ternary and multicomponent)
  - [x] Monomer drift equation (binary and multicomponent)
  - [x] Fitting methods
- Equations of state
  - [50%] Cubic (Redlich-Kwong, Soave, Peng-Robinson)
  - [x] Ideal gas
  - [ ] Sanchez-Lacombe
  - [x] Virial equation
- [ ] Database
- Distributions
  - [x] Flory
  - [x] Log-normal
  - [x] Poison
  - [x] Schulz-Zimm
  - [x] Weibull-Nycander-Gold
- Kinetics
  - [x] Arrhenius
  - [x] Eyring
  - [x] Propagation half-length
  - [x] Termination composite model
- Math
  - [x] Joint confidence regions
- [ ] Models
- Physical property correlations
  - [x] Antoine
  - [x] DIPPR
  - [x] Wagner
  - [x] Yaws
- Step-growth polymerization
  - [x] Analytical solutions for $M_n$ and $M_w$
- Transport properties (estimation methods, mixing rules, etc.)
  - Diffusivity
    - [x] Binary gas mixtures
    - [x] Binary liquid mixtures
    - [x] Binary polymer solutions
    - [ ] Multicomponent polymer solutions  
  - Thermal conductivity
    - [x] Gases
    - [x] Liquids
    - [ ] Polymer solutions
  - Viscosity
    - [x] Gases
    - [x] Liquids
    - [ ] Polymer solutions

## Documentation

Please refer to the package [homepage](https://hugomvale.github.io/polykin/).

## Tutorials

The main features of PolyKin are explained and illustrated through a series of [tutorials](https://hugomvale.github.io/polykin/tutorials/) based on Jupyter [notebooks](https://github.com/HugoMVale/polykin/tree/main/docs/tutorials),
which can be launched online via Binder.

<p align="center">
  <a href="https://github.com/HugoMVale/polykin">
  <img src="https://raw.githubusercontent.com/HugoMVale/polykin/8e54e0b492b4dd782c2fe92b52f617dda71a29b3/docs/deconvolution.svg" width=600 alt="MWD of a polymer blend">
  </a>
</p>

## Installation

PolyKin currently runs on Python 3.9+. You can install it from PyPI via `pip` (or `poetry`):

```console
pip install polykin
# poetry add polykin
```

Alternatively, you may install it directly from the source code repository:

```console
git clone https://github.com/HugoMVale/polykin.git
cd polykin
pip install . 
# poetry install
```
