# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import isclose

from polykin.transport.flow import (Cd_sphere, DP_Darcy_Weisbach,
                                    DP_Hagen_Poiseuille, DP_tube, fD_Colebrook,
                                    fD_Haaland, terminal_velocity_sphere,
                                    terminal_velocity_Stokes,
                                    DP_packed_bed)


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


def test_DP_Hagen_Poiseuille():
    "Noel de Nevers, p. 186"
    Q = 100*3.78e-3/60
    L = 3e3*0.3048
    D = 3.068*2.54e-2
    mu = 100e-3
    DP = DP_Hagen_Poiseuille(Q, D, L, mu)
    assert isclose(DP, 641e3, rtol=1e-2)


def test_DP_Darcy_Weisbach():
    "Noel de Nevers, p. 194"
    Re = 2.05e5
    er = 0.0006
    fD = fD_Colebrook(Re, er)
    rho = 1.0
    DP = DP_Darcy_Weisbach(8.68*0.3048, 3.068*2.54e-2, 2e3*0.3048, rho, fD)
    assert isclose(DP, 524., rtol=1e-2)


def test_DP_tube():
    # Laminar flow
    Q = 1e-3
    L = 100
    D = 0.05
    mu = 0.1
    rho = 1e3
    DP_HP = DP_Hagen_Poiseuille(Q, D, L, mu)
    DP = DP_tube(Q, D, L, rho, mu)
    assert isclose(DP_HP, DP)
    # Turbulent flow
    Q = 200*3.78e-3/60
    L = 2e3*0.3048
    D = 3.068*2.54e-2
    er = 0.0006
    mu = 1e-3
    rho = 1e3
    DP = DP_tube(Q, D, L, rho, mu, er)
    assert isclose(DP, 524*rho, rtol=1e-2)


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


def test_DP_packed_bed():
    "Walas, p. 118"
    Dp = 5e-3
    G = 50.
    rho = 800.
    mu = 0.01
    eps = 0.4
    L = 1.
    DP = DP_packed_bed(G, L, Dp, eps, rho, mu)
    assert isclose(DP, 0.31e5, rtol=15e-2)
