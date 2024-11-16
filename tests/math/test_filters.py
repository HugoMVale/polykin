# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
import pytest
from numpy import all, isclose

from polykin.math.filters import simplify_polyline


def test_polyline():
    p = np.array([(0.0, 0.0),
                  (1.0, 0.1),
                  (2.0, -0.1),
                  (3.0, 5.0),
                  (4.0, 6.0),
                  (5.0, 7.0),
                  (6.0, 8.1),
                  (7.0, 9.0),
                  (8.0, 9.0),
                  (9.0, 9.0)])
    res = simplify_polyline(p, tol=1.0)
    sol = np.array([(0, 0), (2, -0.1), (3, 5), (7, 9), (9, 9)])
    assert all(isclose(res, sol))

    assert all(isclose(simplify_polyline(p[:2, :], tol=1.0), p[:2, :]))

    with pytest.raises(ValueError):
        _ = simplify_polyline(p[:, 0], tol=1.0)

    with pytest.raises(ValueError):
        _ = simplify_polyline(p[0, :], tol=1.0)
