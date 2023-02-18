# PolyKin

[![CI](https://github.com/HugoMVale/polykin/actions/workflows/CI.yml/badge.svg)](https://github.com/HugoMVale/polykin/actions)
[![codecov](https://codecov.io/gh/HugoMVale/polykin/branch/main/graph/badge.svg?token=QfqQLX2rHx)](https://codecov.io/gh/HugoMVale/polykin)
[![Latest Commit](https://img.shields.io/github/last-commit/HugoMVale/polykin)](https://img.shields.io/github/last-commit/HugoMVale/polykin)

PolyKin is an open-source polymerization kinetics library for Python. It is still at an early
development stage, but the following modules can already be used:

- [x] distributions
- [ ] copolymerization  

## Documentation

Please refer to the package [homepage](https://hugomvale.github.io/polykin/).

## Tutorials

The main features of PolyKin are explained and illustrated through a series of [tutorials](https://hugomvale.github.io/polykin/tutorials/distributions/) based on Jupyter [notebooks](https://github.com/HugoMVale/polykin/tree/main/docs/tutorials),
which can be launched online via Binder.

## Installation

PolyKin currently runs on Python>=3.9. You can install it from PyPI via `pip` (or `poetry`):

```sh
pip install polykin
# poetry add polykin
```

Alternatively, you may install it directly from the source code repository:

```sh
git clone https://github.com/HugoMVale/polykin.git
cd polykin
pip install . 
# poetry install
```

<p align="center">
  <img src="https://github.com/HugoMVale/polykin/blob/main/docs/dist.png" width=600 alt="MWD of a polymer blend">
</p>