# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import isclose

from polykin.transport.flow import fD_Colebrook, fD_Haaland, pressure_drop_pipe


def test_pressure_drop_pipe():
    Re = 2.05e5
    er = 0.0006
    fD = fD_Colebrook(Re, er)
    rho = 1.0
    DP = pressure_drop_pipe(3.068*2.54e-2, 8.68*0.3048, rho, fD, 2e3*0.3048)
    assert isclose(DP, 524., rtol=1e-2)


def test_f_Colebrook_White():
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
