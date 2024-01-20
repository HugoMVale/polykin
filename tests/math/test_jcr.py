# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes._axes import Axes

from polykin.math import confidence_ellipse
from polykin.utils.exceptions import ShapeError


def test_confidence_ellipse():
    # Test Case 1: Basic test with valid inputs
    ax = plt.gca()
    center = (0, 0)
    cov = np.array([[1, 0.5], [0.5, 2]])
    ndata = 100
    alpha = 0.05
    color = 'black'
    confidence_ellipse(ax, center, cov, ndata, alpha, color)
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 1

    # Test Case 2: Test with different center and covariance matrix
    ax = plt.gca()
    center = (2, 3)
    cov = np.array([[2, 0.3], [0.3, 1]])
    ndata = 150
    alpha = 0.1
    color = 'blue'
    confidence_ellipse(ax, center, cov, ndata, alpha, color)
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 2

    # Test Case 3: Test with invalid center (wrong length)
    ax = plt.gca()
    center = (0,)
    cov = np.array([[1, 0.5], [0.5, 2]])
    ndata = 100
    alpha = 0.05
    color = 'black'
    with pytest.raises(ShapeError):
        confidence_ellipse(ax, center, cov, ndata, alpha, color)

    # Test Case 4: Test with invalid covariance matrix (wrong shape)
    ax = plt.gca()
    center = (0, 0)
    cov = np.array([[1, 0.5], [0.5, 2], [0.1, 0.2]])
    ndata = 100
    alpha = 0.05
    color = 'black'
    with pytest.raises(ShapeError):
        confidence_ellipse(ax, center, cov, ndata, alpha, color)

    # Test Case 5: Test with invalid alpha (below lower bound)
    ax = plt.gca()
    center = (0, 0)
    cov = np.array([[1, 0.5], [0.5, 2]])
    ndata = 100
    alpha = 0.0001
    color = 'black'
    with pytest.raises(ValueError):
        confidence_ellipse(ax, center, cov, ndata, alpha, color)

    # Test Case 6: Test with invalid ndata (below lower bound)
    ax = plt.gca()
    center = (0, 0)
    cov = np.array([[1, 0.5], [0.5, 2]])
    ndata = 2
    alpha = 0.05
    color = 'black'
    with pytest.raises(ValueError):
        confidence_ellipse(ax, center, cov, ndata, alpha, color)

    # Test Case 7: Check if the Ellipse is added to the Axes
    ax = plt.gca()
    center = (0, 0)
    cov = np.array([[1, 0.5], [0.5, 2]])
    ndata = 100
    alpha = 0.05
    color = 'black'
    confidence_ellipse(ax, center, cov, ndata, alpha, color)
    assert len(ax.patches) == 3
