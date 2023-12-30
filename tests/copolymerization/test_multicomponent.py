# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from numpy import all, isclose

from polykin.copolymerization import convert_Qe_to_r


def test_convert_Qe():
    Qe1 = (1., -0.8)     # Sty
    Qe2 = (0.78, 0.4)    # MMA
    Qe3 = (0.026, -0.88)  # VAc
    r = convert_Qe_to_r([Qe1, Qe2, Qe3])
    assert all(isclose(r.diagonal(), [1., 1., 1.]))
    assert all(isclose(r[0, :], [1., 0.5, 40.], rtol=0.1))
