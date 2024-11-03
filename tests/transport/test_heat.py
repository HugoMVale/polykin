# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import isclose

from polykin.transport.heat import (Nu_cylinder, Nu_drop, Nu_flatplate,
                                    Nu_sphere, Nu_tube, Nu_tank)


def test_Nu_tube():
    # turbulent
    Pr = 1
    Re = 1e5
    Nu = Nu_tube(Re, Pr)
    Nu_Colburn = 0.023*Re**0.8 * Pr**0.33
    assert isclose(Nu, Nu_Colburn, rtol=10e-2)
    # laminar
    Re = 1e3
    D_L = 0.1
    Nu_Walas = (3.66**3 + 1.61**3 * Re * Pr * D_L)**(1/3)
    Nu = Nu_tube(Re, Pr, D_L)
    assert isclose(Nu, Nu_Walas, rtol=10e-2)


def test_Nu_cylinder_cross_flow():
    "Incropera, Example 7.4,  p. 373"
    Pr = 0.7
    Re = 6071
    Nu = Nu_cylinder(Re, Pr)
    assert isclose(Nu, 40.6, rtol=1e-2)


def test_Nu_sphere():
    Pr = 0.709
    Re = 6510
    mur = 181.6/197.8
    Nu = Nu_sphere(Re, Pr, mur)
    assert isclose(Nu, 47.4, rtol=1e-2)


def test_Nu_drop():
    Pr = 1.
    Re = 1000.
    Nu_s = Nu_sphere(Re, Pr, 1.0)
    Nu_d = Nu_drop(Re, Pr)
    assert isclose(Nu_s, Nu_d, rtol=5e-2)


def test_Nu_flatplate():
    "Incropera, Example 7.1,  p. 360"
    Pr = 0.687
    Re = 9597
    Nu = Nu_flatplate(Re, Pr)
    assert isclose(Nu, 57.4, rtol=1e-2)
    Re = 6.84e5
    Nu = Nu_flatplate(Re, Pr)
    assert isclose(Nu, 753, rtol=1e-2)


def test_Nu_tank():
    "Handbook of Industrial Mixing, p. 882"
    # Example 14.1
    Re = 1.57e6
    Pr = 4.37
    Nu = Nu_tank('6BD', 'wall', Re, Pr, mur=1.)
    assert isclose(Nu, 16400, rtol=1e-2)
    # Example 14.2
    Re = 76
    Pr = 2.5e5
    Nu = Nu_tank('helical-ribbon', 'wall', Re, Pr, mur=1.)
    assert isclose(Nu, 337, rtol=1e-2)
