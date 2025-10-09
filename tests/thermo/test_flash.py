import numpy as np
from numpy import allclose, isclose

from polykin.properties.equations import Antoine
from polykin.thermo.acm import UNIQUAC
from polykin.thermo.flash.vle import (flash2_PV, flash2_PT,
                                      residual_Rachford_Rice,
                                      solve_Rachford_Rice)

# %% Parameters for water(1) end EG(2)

# Activity coefficient model
N = 2
a = np.zeros((N, N))
b = np.zeros((N, N))
a[0, 1] = 3.974
a[1, 0] = -10.33
b[0, 1] = -1194.0
b[1, 0] = 3359.0
r = np.array([0.92, 2.41])
q = np.array([1.4, 2.25])
acm = UNIQUAC(N, q, r, a=a, b=b)

# Pure component vapor pressures
unit_conv = np.log10(101325/760)
Psat_water = Antoine(A=8.07131 + unit_conv,
                     B=1730.63,
                     C=(233.426 - 273.15),
                     unit="bar",
                     name="water")
Psat_EG = Antoine(A=(8.09083 + unit_conv),
                  B=2088.936,
                  C=(203.454 - 273.15),
                  unit="bar",
                  name="EG")


def Psat(T):
    return np.array([Psat_water(T), Psat_EG(T)])


def Kcalc(T, P, x, y):
    return acm.gamma(T, x)*Psat(T)/P


# %% Tests

def test_flash_residual():
    # Subcooled feed
    z = np.array([0.5, 0.25, 0.25])
    K = np.array([0.2, 0.75, 0.99])
    F, = residual_Rachford_Rice(0.0, K, z)
    assert F < 0.0
    # Superheated feed
    K = np.array([1.5, 2.0, 5.0])
    F, = residual_Rachford_Rice(1.0, K, z)
    assert F > 0.0
    # Saturated feed: Seader, Example 4.1
    z = np.array([0.1, 0.2, 0.3, 0.4])
    K = np.array([4.2, 1.75, 0.74, 0.34])
    F, = residual_Rachford_Rice(0.0, K, z)
    assert isclose(F, 0.128, atol=1e-3)
    F, = residual_Rachford_Rice(1.0, K, z)
    assert isclose(F, -0.720, atol=1e-3)
    F, dF = residual_Rachford_Rice(0.5, K, z, derivative=True)
    assert isclose(F, -0.2515, atol=1e-4)
    assert isclose(dF, -0.6259, atol=1e-4)


def test_solve_flash_residual():
    # Subcooled feed
    z = np.array([0.5, 0.25, 0.25])
    K = np.array([0.2, 0.75, 0.99])
    sol = solve_Rachford_Rice(K, z)
    assert sol.success
    assert isclose(sol.beta, 0.0)
    # Superheated feed
    K = np.array([1.5, 2.0, 5.0])
    sol = solve_Rachford_Rice(K, z)
    assert sol.success
    assert isclose(sol.beta, 1.0)
    # Saturated feed: Seader, Example 4.1
    z = np.array([0.1, 0.2, 0.3, 0.4])
    K = np.array([4.2, 1.75, 0.74, 0.34])
    for beta0 in [None, 0.4]:
        sol = solve_Rachford_Rice(K, z, beta0=beta0)
        assert sol.success
        assert isclose(sol.beta, 0.1219, atol=1e-4)


def test_flash2_PT():
    sol = flash2_PT(F=1.0,
                    z=np.array([0.5, 0.5]),
                    T=373.0,
                    P=0.4e5,
                    Kcalc=Kcalc)
    assert sol.success
    assert allclose(sol.x, [0.2910601, 0.7089399], rtol=1e-3)
    assert allclose(sol.y, [0.96042225, 0.03957775], rtol=1e-3)


def test_flash2_PV():
    sol = flash2_PV(F=1.0,
                    z=np.array([0.5, 0.5]),
                    P=0.4e5,
                    beta=0.2,
                    Kcalc=Kcalc,
                    T0=300.0)
    assert sol.success
    assert allclose(sol.x, [0.3814322, 0.6185678], rtol=1e-3)
    assert allclose(sol.y, [0.97427121, 0.02572879], rtol=1e-3)
    assert allclose(sol.T, 367.1325398903182, atol=0.01)


def test_flash2_PV_bubble():
    sol = flash2_PV(F=1.0,
                    z=np.array([0.5, 0.5]),
                    P=0.6e5,
                    beta=0.0,
                    Kcalc=Kcalc,
                    T0=300.0)

    assert isclose(sol.T, 372.7277, atol=0.01)
    assert allclose(sol.y, [0.9803498, 0.0196502], rtol=1e-4)


def test_flash2_PV_dew():
    sol = flash2_PV(F=1.0,
                    z=np.array([0.9803498, 0.0196502]),
                    P=0.6e5,
                    beta=1.0,
                    Kcalc=Kcalc,
                    T0=300.0)

    assert isclose(sol.T, 372.7277, atol=0.01)
    assert allclose(sol.x, [0.5, 0.5], rtol=1e-4)


# def test_bubble_P():
#     P, y, _ = bubble_P(T=373.15,
#                        x=np.array([0.5, 0.5]),
#                        Kcalc=Kcalc,
#                        P0=1e5)
#     assert isclose(P/1e5, 0.6088, rtol=1e-3)
#     assert allclose(y, [0.98019598, 0.01980402], rtol=1e-3)

# def test_dew_P():
#     P, x, K = dew_P(T=373.15,
#                     y=np.array([0.9, 0.1]),
#                     Kcalc=Kcalc,
#                     P0=1e5)
#     assert isclose(P/1e5, 0.1925, rtol=1e-3)
#     assert allclose(x, [0.11173842, 0.88826158], rtol=1e-3)
