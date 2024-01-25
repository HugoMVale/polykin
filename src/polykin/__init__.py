# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""
# PolyKin: A polymerization kinetics library for Python.

Documentation is available in the docstrings and online at https://hugomvale.github.io/polykin.

## Subpackages

Using any of these subpackages requires an explicit import. For example,
`import polykin.distributions`.

* copolymerization = copolymerization models
* distributions    = chain-length distributions
* kinetics         = kinetic coefficients
* properties       = physical property methods and models

"""
import importlib.metadata

from polykin.properties.equations import plotequations

__version__ = importlib.metadata.version('polykin')
