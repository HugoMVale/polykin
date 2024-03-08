# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import exp, isclose

from polykin.math import derivative_centered, derivative_complex


def fnc(x): return 2*exp(x)


def test_derivative_centered():
    df, fx = derivative_centered(fnc, 2)
    assert isclose(df, fx)


def test_derivative_complex():
    df, fx = derivative_complex(fnc, 2)
    assert isclose(df, fx)
