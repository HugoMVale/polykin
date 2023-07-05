# PolyKin

[![Test](https://github.com/HugoMVale/polykin/actions/workflows/test.yml/badge.svg)](https://github.com/HugoMVale/polykin/actions)
[![codecov](https://codecov.io/gh/HugoMVale/polykin/branch/main/graph/badge.svg?token=QfqQLX2rHx)](https://codecov.io/gh/HugoMVale/polykin)
[![Latest Commit](https://img.shields.io/github/last-commit/HugoMVale/polykin)](https://img.shields.io/github/last-commit/HugoMVale/polykin)

PolyKin is an open-source polymerization kinetics library for Python. It is still at an early
development stage, but the following modules can already be used:

- [x] distributions
- [x] coefficients
- [ ] copolymerization
- [ ] kinetics
- [ ] database
- [ ] models

## Documentation

Please refer to the package [homepage](https://hugomvale.github.io/polykin/).

## Tutorials

The main features of PolyKin are explained and illustrated through a series of [tutorials](https://hugomvale.github.io/polykin/tutorials/distributions/) based on Jupyter [notebooks](https://github.com/HugoMVale/polykin/tree/main/docs/tutorials),
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
