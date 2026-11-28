# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from numpy import allclose

from polykin.copolymerization import (
    CopoDataset_Ff,
    CopoDataset_Fx,
    CopoDataset_fx,
    fit_copo_data,
    fit_Finemann_Ross,
    inst_copolymer_binary,
    monomer_drift_binary,
)


def test_fit_Finemann_Ross():
    """Scott & Penlidis (2019). p. 9."""
    f1 = [0.70, 0.50, 0.30, 0.10, 0.05, 0.010]
    F1 = [0.66, 0.42, 0.23, 0.06, 0.03, 0.005]
    result = fit_Finemann_Ross(f1, F1)
    assert allclose(result, (1.06, 1.83), atol=0.01)
    result = fit_Finemann_Ross(1.0 - np.array(f1), 1.0 - np.array(F1))
    assert allclose(result, (2.03, 1.96), atol=0.01)


def test_fit_reactivity_ratios_1():
    """Van Herk & Dröge (1997). p. 9."""
    name = "van Herk & Dröge"
    f1 = np.array([0.100, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800])
    F1 = np.array([0.059, 0.243, 0.364, 0.486, 0.583, 0.721, 0.824])
    # const error
    for method in ["NLLS", "ODR"]:
        data = CopoDataset_Ff(name, f1, F1, 1e-10, 0.05)
        result = fit_copo_data([data], method=method)  # type: ignore
        assert result
        r1 = result.r1
        r2 = result.r2
        assert allclose([r1, r2], [1.43, 1.67], rtol=0.01)
    # relative error
    for method in ["NLLS", "ODR"]:
        data = CopoDataset_Ff(name, f1, F1, 1e-10, 0.05 * F1)
        result = fit_copo_data([data], method=method)  # type: ignore
        assert result
        r1 = result.r1
        r2 = result.r2
        assert allclose([r1, r2], [1.71, 1.95], rtol=0.01)


def test_fit_reactivity_ratios_2():
    """Scott & Penlidis (2019). p. 9."""
    name = "Scott & Penlidis"
    f1 = np.array([0.70, 0.50, 0.30, 0.10, 0.05, 0.010])
    F1 = np.array([0.66, 0.42, 0.23, 0.06, 0.03, 0.005])
    for method in ["NLLS", "ODR"]:
        data = CopoDataset_Ff(name, f1, F1, 0.01 * f1, 0.1 * F1)
        result = fit_copo_data([data], method=method)  # type: ignore
        assert result
        r1 = result.r1
        r2 = result.r2
        assert allclose([r1, r2], [1.19, 1.87], rtol=0.01)


def test_fit_mixed_reactivity_data():
    """Fit weighted instantaneous and composition drift residuals together."""
    r_true = (1.5, 0.6)
    f1_Ff = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    F1_Ff = inst_copolymer_binary(f1_Ff, *r_true)

    f10 = 0.4
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    f1_x = monomer_drift_binary(f10, x, *r_true)
    F1_x = f1_x + (f10 - f1_x) / (x + 1e-10)

    result = fit_copo_data(
        [CopoDataset_Ff("Ff", f1_Ff, F1_Ff, 1.0, 0.02, 0.5)],
        [CopoDataset_fx("fx", f10, x, f1_x, 0.03, 2.0)],
        [CopoDataset_Fx("Fx", f10, x, F1_x, 0.04, 1.5)],
        r_guess=(1.0, 1.0),
        plot_data=False,
        JCR_linear=False,
    )

    assert allclose([result.r1, result.r2], r_true, rtol=1e-3)
    assert result.cov is not None
    assert np.all(np.isfinite(result.cov))
