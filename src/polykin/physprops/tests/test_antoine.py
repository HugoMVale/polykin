# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.physprops import Antoine


# import pytest
import numpy as np


def test_Antoine():
    """Pvap of water
    Parameters: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=4&Type=ANTOINE&Plot=on
    """
    p = Antoine(A=4.6543, B=1435.264, C=-64.848,
                Tmin=255.9, Tmax=373., unit='bar')
    assert np.isclose(p(100., 'C'), 1.01325, rtol=2e-2)
