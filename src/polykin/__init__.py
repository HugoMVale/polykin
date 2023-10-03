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
* diffusion        = diffusion models
* kinetics         = kinetic coefficients
* physprops        = physical property equations

"""
from polykin.physprops.propertyequation import plotequations

__version__ = "0.2.0"

from polykin._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
