# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from numpy import all, isclose

from polykin.copolymerization import fit_Finemann_Ross

# %% Terminal


def test_fit_Finemann_Ross():
    f1 = [0.186, 0.299, 0.527, 0.600, 0.700, 0.798]
    F1 = [0.196, 0.279, 0.415, 0.473, 0.542, 0.634]
    result = fit_Finemann_Ross(np.array(f1), np.array(F1))
    assert all(isclose(result, (0.25, 0.79), atol=0.03))
