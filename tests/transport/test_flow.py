# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import isclose

from polykin.transport.flow import (Cd_sphere, fD_Colebrook, fD_Haaland,
                                    pressure_drop_pipe,
                                    terminal_velocity_sphere,
                                    terminal_velocity_Stokes)


def test_fD():
    "Noel de Nevers, p. 191"
    for fun in [fD_Colebrook, fD_Haaland]:
        f = fun(1e4, 0.0)
        assert isclose(f, 0.0075*4, rtol=5e-2)
        f = fun(1e5, 0.0)
        assert isclose(f, 0.0045*4, rtol=2e-2)
        f = fun(1e7, 0.0)
        assert isclose(f, 0.008, rtol=2e-2)
        f = fun(1e6, 0.006)
        assert isclose(f, 0.008*4, rtol=2e-2)
        f = fun(1e6, 0.04)
        assert isclose(f, 0.016*4, rtol=2e-2)


def test_pressure_drop_pipe():
    "Noel de Nevers, p. 194"
    Re = 2.05e5
    er = 0.0006
    fD = fD_Colebrook(Re, er)
    rho = 1.0
    DP = pressure_drop_pipe(3.068*2.54e-2, 8.68*0.3048, rho, fD, 2e3*0.3048)
    assert isclose(DP, 524., rtol=1e-2)


def test_Cd_sphere():
    "Noel de Nevers, p. 225"
    Re = 1e-3
    CD0 = Cd_sphere(Re)
    assert isclose(CD0, 24/Re, rtol=1e-2)
    Re = 5e3
    CD0 = Cd_sphere(Re)
    assert isclose(CD0, 0.4, rtol=2e-2)


def test_terminal_velocity_Stokes():
    "Noel de Nevers, p. 226"
    D = 1e-4*2.54e-2
    mu = 0.018e-3
    rho = (101325*29e-3)/(8.314*298.15)
    rhop = 1602.
    vt = terminal_velocity_Stokes(D, rhop, rho, mu)
    assert isclose(vt, 1e-3*0.3048, rtol=5e-2)


def test_terminal_velocity_sphere_1():
    "Noel de Nevers, p. 226"
    D = 1e-4*2.54e-2
    mu = 0.018e-3
    rho = (101325*29e-3)/(8.314*298.15)
    rhop = 1602.
    vt_stokes = terminal_velocity_Stokes(D, rhop, rho, mu)
    vt = terminal_velocity_sphere(D, rhop, rho, mu)
    assert isclose(vt_stokes, vt, rtol=5e-2)


def test_terminal_velocity_sphere_2():
    "Noel de Nevers, p. 227"
    D = 0.02
    mu = 1e-3
    rhop = 7.85e3
    rho = 1e3
    vt = terminal_velocity_sphere(D, rhop, rho, mu)
    assert isclose(vt, 1.9, rtol=5e-2)
