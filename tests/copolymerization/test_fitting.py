# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from numpy import all, isclose

from polykin.copolymerization import (CopoDataset_Ff, fit_copo_data,
                                      fit_Finemann_Ross)

# %% Terminal


def test_fit_Finemann_Ross():
    "Scott & Penlidis (2019). p. 9."
    f1 = [0.70, 0.50, 0.30, 0.10, 0.05, 0.010]
    F1 = [0.66, 0.42, 0.23, 0.06, 0.03, 0.005]
    result = fit_Finemann_Ross(f1, F1)
    assert all(isclose(result, (1.06, 1.83), atol=0.01))
    result = fit_Finemann_Ross(1. - np.array(f1), 1. - np.array(F1))
    assert all(isclose(result, (2.03, 1.96), atol=0.01))


def test_fit_reactivity_ratios_1():
    "van Herk & Dröge (1997). p. 9."
    name = "van Herk & Dröge"
    f1 = np.array([0.100, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800])
    F1 = np.array([0.059, 0.243, 0.364, 0.486, 0.583, 0.721, 0.824])
    # const error
    for method in ['NLLS', 'ODR']:
        data = CopoDataset_Ff(name, f1, F1, 1e-10, 0.05)
        result = fit_copo_data([data], method=method)  # type: ignore
        assert result
        r1 = result.r1
        r2 = result.r2
        assert all(isclose([r1, r2], [1.43, 1.67], rtol=0.01))
    # relative error
    for method in ['NLLS', 'ODR']:
        data = CopoDataset_Ff(name, f1, F1, 1e-10, 0.05*F1)
        result = fit_copo_data([data], method=method)  # type: ignore
        assert result
        r1 = result.r1
        r2 = result.r2
        assert all(isclose([r1, r2], [1.71, 1.95], rtol=0.01))


def test_fit_reactivity_ratios_2():
    "Scott & Penlidis (2019). p. 9."
    name = "Scott & Penlidis"
    f1 = np.array([0.70, 0.50, 0.30, 0.10, 0.05, 0.010])
    F1 = np.array([0.66, 0.42, 0.23, 0.06, 0.03, 0.005])
    for method in ['NLLS', 'ODR']:
        data = CopoDataset_Ff(name, f1, F1, 0.01*f1, 0.1*F1)
        result = fit_copo_data([data], method=method)  # type: ignore
        assert result
        r1 = result.r1
        r2 = result.r2
        assert all(isclose([r1, r2], [1.19, 1.87], rtol=0.01))
