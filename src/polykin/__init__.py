"""
# PolyKin: A polymerization kinetics library for Python.

Documentation is available in the docstrings and online at https://hugomvale.github.io/polykin.

## Subpackages

Using any of these subpackages requires an explicit import. For example,
`import polykin.distributions`.

* distributions = chain-length distributions

"""

from polykin.coefficients import Arrhenius, Eyring, \
    TerminationCompositeModel
from pathlib import Path

__version__ = (Path(__file__).resolve().parent / "_version.txt").read_text(
    encoding="utf-8")
