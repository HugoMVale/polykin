# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.properties.pvt import Z_virial

import pytest
import numpy as np


def test_Z_virial():
    "Example 3-1, p. 35"
    Z = Z_virial(T=366.5, P=20.67, Tc=385.0, Pc=41.4, w=0.204)
    assert np.isclose(Z, 0.75, rtol=0.1)
