# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numba import njit

from polykin.utils.types import FloatMatrix

__all__ = ['simplify_polyline']


def simplify_polyline(p: FloatMatrix,
                      tol: float
                      ) -> FloatMatrix:
    r"""Simplify an N-dimensional polyline using the [Ramer-Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm).

    Parameters
    ----------
    p : FloatMatrix
        Matrix of point coordinates.
    tol : float
        Point-to-edge distance tolerance. Points closer than `tol` are removed.

    Returns
    -------
    FloatMatrix
        Simplified polyline.

    Examples
    --------
    Simplify a 2D polyline.
    >>> from polykin.math import simplify_polyline
    >>> p = np.array([[0.0, 0.0],
    ...               [1.0, 0.1],
    ...               [2.0, -0.1],
    ...               [3.0, 5.0],
    ...               [4.0, 6.0],
    ...               [5.0, 7.0],
    ...               [6.0, 8.1],
    ...               [7.0, 9.0],
    ...               [8.0, 9.0],
    ...               [9.0, 9.0]])
    >>> simplify_polyline(p, tol=1.0)
    array([[ 0. ,  0. ],
           [ 2. , -0.1],
           [ 3. ,  5. ],
           [ 7. ,  9. ],
           [ 9. ,  9. ]])
    """

    # Check inputs
    nvert, ndims = p.shape
    if nvert < 2:
        raise ValueError("Polyline must have at least 2 points.")
    if ndims < 2:
        raise ValueError("Polyline must have at least 2 dimensions.")

    # Find row indices that need to be kept
    idx = [i for i in range(nvert)]
    idx = _simplify_rdh(idx, p, tol)

    return p[idx, :]


@njit
def _simplify_rdh(idx: list[int],
                  p: np.ndarray,
                  tol: float
                  ) -> list[int]:
    """Find indexes of points that need to be kept.

    Parameters
    ----------
    idx : list[int]
        List of point indices.
    p : np.ndarray
        Matrix of point coordinates.
    tol : float
        Simplification tolerance.

    Returns
    -------
    list[int]
        Indices of points that need to be kept.
    """
    # Find most distant point
    dmax = 0.
    imax = 0
    for i in range(1, len(idx) - 2):
        d = _perpdist(p[idx[i], :], p[idx[0], :], p[idx[-1], :])
        if d > dmax:
            dmax = d
            imax = i

    # Recursively simplify
    if dmax > tol:
        line1 = _simplify_rdh(idx[:imax+1], p, tol)
        line2 = _simplify_rdh(idx[imax:], p, tol)
        res = line1 + line2[1:]
    else:
        res = [idx[0], idx[-1]]

    return res


@njit
def _perpdist(pt: np.ndarray,
              line_start: np.ndarray,
              line_end: np.ndarray
              ) -> float:
    r"""Perpendicular distance between a point and a line in hyperspace.

    Parameters
    ----------
    pt : np.ndarray
        Point coordinates.
    line_start : np.ndarray
        Line start coordinates.
    line_end : np.ndarray
        Line end coordinates.

    Returns
    -------
    float
        Perpendicular distance.
    """
    d = line_end - line_start
    d /= np.linalg.norm(d)
    pv = pt - line_start
    pvdot = np.dot(d, pv)
    ds = pvdot * d
    return np.linalg.norm(pv - ds)  # type: ignore
