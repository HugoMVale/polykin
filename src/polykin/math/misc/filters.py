# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numba import njit

from polykin.utils.types import FloatMatrix, FloatVector

__all__ = ['simplify_polyline']


def simplify_polyline(points: FloatMatrix,
                      tol: float
                      ) -> FloatMatrix:
    r"""Simplify an N-dimensional polyline using the Ramer-Douglas-Peucker
    algorithm.

    The Ramer-Douglas-Peucker algorithm is considered the best global polyline
    simplification algorithm. This particular implementation is based on the
    recursive version of the method. 

    **References**

    * Ramer, Urs. "An iterative procedure for the polygonal approximation of
    plane curves." Computer graphics and image processing 1.3 (1972): 244-256.
    * Douglas, David H., and Thomas K. Peucker. "Algorithms for the reduction
    of the number of points required to represent a digitized line or its
    caricature." Cartographica: the international journal for geographic
    information and geovisualization 10.2 (1973): 112-122.

    Parameters
    ----------
    points : FloatMatrix
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

    nvert, ndims = points.shape

    if ndims < 2:
        raise ValueError("Polyline must have at least 2 dimensions.")

    if nvert < 2:
        raise ValueError("Polyline must have at least 2 points.")
    elif nvert == 2:
        return points
    else:
        idx = _ramer_douglas_peucker(points, 0, nvert - 1, tol)
        return points[idx, :]


@njit
def _ramer_douglas_peucker(points: FloatMatrix,
                           istart: int,
                           iend: int,
                           tol: float
                           ) -> list[int]:
    """Recursively find indexes of points that should be kept.

    Parameters
    ----------
    points : FloatMatrix
        Matrix of point coordinates.
    istart : int
        Start index of polyline segment.
    iend : int
        End index of polyline segment.
    tol : float
        Point-to-edge distance tolerance.

    Returns
    -------
    list[int]
        Indices of points that should be kept.
    """
    # Find farthest point
    imax, dmax = _farthest_point(points, istart, iend)

    # Recursively simplify
    if dmax > tol:
        line1 = _ramer_douglas_peucker(points, istart, imax, tol)
        line2 = _ramer_douglas_peucker(points, imax, iend, tol)
        result = line1 + line2[1:]
    else:
        result = [istart, iend]

    return result


@njit
def _farthest_point(points: FloatVector,
                    istart: int,
                    iend: int
                    ) -> tuple[int, float]:
    """Farthest point from a line segment.

    Parameters
    ----------
    points : FloatVector
        Matrix of point coordinates.
    istart : int
        Start index of polyline segment.
    iend : int
        End index of polyline segment.

    Returns
    -------
    tuple[int, float]
        Index and distance of farthest point.
    """
    imax = 0
    dmax = 0.
    for i in range(istart + 1, iend):
        d = _perpendicular_distance(
            points[i, :], points[istart, :], points[iend, :])
        if d > dmax:
            imax = i
            dmax = d

    return (imax, dmax)


@njit
def _perpendicular_distance(point: FloatVector,
                            line_start: FloatVector,
                            line_end: FloatVector
                            ) -> float:
    """Perpendicular distance between a point and a line in hyperspace.

    Parameters
    ----------
    point : FloatVector
        Point coordinates.
    line_start : FloatVector
        Line start coordinates.
    line_end : FloatVector
        Line end coordinates.

    Returns
    -------
    float
        Perpendicular distance.
    """
    d = line_end - line_start
    d /= np.linalg.norm(d)
    pv = point - line_start
    pvdot = np.dot(d, pv)
    ds = pvdot * d
    return np.linalg.norm(pv - ds)  # type: ignore
