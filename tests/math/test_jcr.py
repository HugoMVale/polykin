# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes._axes import Axes
from scipy.optimize import minimize

from polykin.math import confidence_ellipse, confidence_region
from polykin.utils.exceptions import ShapeError


def test_confidence_ellipse():
    _, ax = plt.subplots()
    # Test Case 1: Basic test with valid inputs
    center = (0, 0)
    cov = np.array([[1, 0.5], [0.5, 2]])
    ndata = 100
    alpha = 0.05
    color = 'black'
    confidence_ellipse(ax, center, cov, ndata, alpha, color)
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 1

    # Test Case 2: Test with different center and covariance matrix
    center = (2, 3)
    cov = np.array([[2, 0.3], [0.3, 1]])
    ndata = 150
    alpha = 0.1
    color = 'blue'
    confidence_ellipse(ax, center, cov, ndata, alpha, color)
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 2

    # Test Case 3: Test with invalid center (wrong length)
    center = (0,)
    cov = np.array([[1, 0.5], [0.5, 2]])
    ndata = 100
    alpha = 0.05
    color = 'black'
    with pytest.raises(ShapeError):
        confidence_ellipse(ax, center, cov, ndata, alpha, color)

    # Test Case 4: Test with invalid covariance matrix (wrong shape)
    center = (0, 0)
    cov = np.array([[1, 0.5], [0.5, 2], [0.1, 0.2]])
    ndata = 100
    alpha = 0.05
    color = 'black'
    with pytest.raises(ShapeError):
        confidence_ellipse(ax, center, cov, ndata, alpha, color)

    # Test Case 5: Test with invalid alpha (below lower bound)
    center = (0, 0)
    cov = np.array([[1, 0.5], [0.5, 2]])
    ndata = 100
    alpha = 0.0001
    color = 'black'
    with pytest.raises(ValueError):
        confidence_ellipse(ax, center, cov, ndata, alpha, color)

    # Test Case 6: Test with invalid ndata (below lower bound)
    center = (0, 0)
    cov = np.array([[1, 0.5], [0.5, 2]])
    ndata = 2
    alpha = 0.05
    color = 'black'
    with pytest.raises(ValueError):
        confidence_ellipse(ax, center, cov, ndata, alpha, color)

    # Test Case 7: Check if the Ellipse is added to the Axes
    center = (0, 0)
    cov = np.array([[1, 0.5], [0.5, 2]])
    ndata = 100
    alpha = 0.05
    color = 'black'
    confidence_ellipse(ax, center, cov, ndata, alpha, color)
    assert len(ax.patches) == 3


def test_confidence_region():
    # model and synthetic data
    def model(x, beta):
        return beta[0]*x**2 + beta[1]*x**5

    ndata = 101
    X = np.linspace(0., 1., ndata)
    beta = (0.15, -0.05)
    Y = model(X, beta) + np.random.normal(0., 0.05, len(X))

    def sse(beta) -> float:
        Ye = model(X, beta)
        return sum((Ye - Y)**2)

    sol = minimize(sse, (-1., 1.))
    beta_est = sol.x

    jcr = confidence_region(beta_est,
                            sse=sse,
                            ndata=ndata)
    assert len(jcr[0]) > 1
