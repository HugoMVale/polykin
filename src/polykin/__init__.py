"""
# PolyKin: A polymerization kinetics library for Python.

Documentation is available in the docstrings and online at https://hugomvale.github.io/polykin.

## Subpackages

Using any of these subpackages requires an explicit import. For example,
`import polykin.distributions`.

* distributions = chain-length distributions

"""

from polykin.coefficients import Arrhenius, Eyring, \
    CompositeModelTermination

__version__ = "0.2.0"
