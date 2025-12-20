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
* flow             = fluid dynamics and rheology
* hmt              = heat and mass transfer
* kinetics         = kinetic coefficients
* math             = mathematical algorithms
* properties       = physical property methods and models
* reactors         = reaction engineering models
* stepgrowth       = step-growth polymerization formulas
* thermo           = thermodynamic models
"""

import importlib.metadata

__version__ = importlib.metadata.version("polykin")
