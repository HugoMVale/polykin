# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

import pytest
from numpy import isclose

from polykin.properties.pvt import VL_Rackett


def test_VL_Rackett():
    """Example 3-5, p. 68, Reid-Prausnitz-Poling."""
    Tc = 408.2
    Pc = 36.5e5
    w = 0.183
    ZRA = 0.2754
    vL1 = VL_Rackett(310.93, Tc, Pc, ZRA=ZRA)
    assert isclose(vL1, 108.9e-6, rtol=1e-3)
    vL2 = VL_Rackett(310.93, Tc, Pc, w=w)
    assert isclose(vL2, vL1, rtol=1e-2)
    # Invalid input, ZRA and w
    with pytest.raises(ValueError):
        _ = VL_Rackett(310.93, Tc, Pc, ZRA=ZRA, w=w)
