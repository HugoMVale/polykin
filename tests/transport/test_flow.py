# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import isclose, pi

from polykin.transport.flow import (Cd_sphere, DP_Darcy_Weisbach,
                                    DP_GL_Lockhart_Martinelli, DP_GL_Mueller_Bonn,
                                    DP_Hagen_Poiseuille, DP_packed_bed,
                                    DP_tube, fD_Colebrook, fD_Haaland,
                                    vt_sphere, vt_Stokes)


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


def test_vt_Stokes():
    "Noel de Nevers, p. 226"
    D = 1e-4*2.54e-2
    mu = 0.018e-3
    rho = (101325*29e-3)/(8.314*298.15)
    rhop = 1602.
    vt = vt_Stokes(D, rhop, rho, mu)
    assert isclose(vt, 1e-3*0.3048, rtol=5e-2)


def test_vt_sphere_1():
    "Noel de Nevers, p. 226"
    D = 1e-4*2.54e-2
    mu = 0.018e-3
    rho = (101325*29e-3)/(8.314*298.15)
    rhop = 1602.
    vt_stokes = vt_Stokes(D, rhop, rho, mu)
    vt = vt_sphere(D, rhop, rho, mu)
    assert isclose(vt_stokes, vt, rtol=5e-2)


def test_vt_sphere_2():
    "Noel de Nevers, p. 227"
    D = 0.02
    mu = 1e-3
    rhop = 7.85e3
    rho = 1e3
    vt = vt_sphere(D, rhop, rho, mu)
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


def test_DP_GL_Lockhart_Martinelli():
    "Walas, p. 116"
    mdotL = 140e3*1.26e-4
    mdotG = 800*1.26e-4
    rhoL = 51.85*16.02
    rhoG = 0.142*16.02
    muL = 15e-3
    muG = 2.5e-7*47.88
    D = 0.2557*0.3048
    L = 1e2*0.3048
    er = 0.00059
    DP = DP_GL_Lockhart_Martinelli(
        mdotL, mdotG, D, L, rhoL, rhoG, muL, muG, er)
    assert isclose(DP, 36.8*6895, rtol=5e-2)
    # only liquid
    DP = DP_GL_Lockhart_Martinelli(mdotL, 0.0, D, L, rhoL, 0.0, muL, 0.0, er)
    assert DP > 0
    # only gas
    DP = DP_GL_Lockhart_Martinelli(0.0, mdotG, D, L, 0.0, rhoG, 0.0, muG, er)
    assert DP > 0


def test_DP_GL_Mueller_Bonn():
    "Mueller-Steinhagen, 1986, Figure 2"
    D = 0.036  # m
    L = 1     # m
    vm = 200  # kg/m²/s
    P = 1.4   # bar
    rhoL = 998
    muL = 1e-3
    rhoG = 1.2*(P/1.01325)
    muG = 1.8e-5
    mdot = vm*(pi/4*D**2)
    x = 0.8
    DP = DP_GL_Mueller_Bonn(mdot*(1-x), mdot*x, D, L, rhoL, rhoG, muL, muG)
    assert isclose(DP, 6e3, rtol=10e-2)
