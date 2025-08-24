from numpy import exp, isclose, inf

from polykin.reactors.rtd import *


def test_E_cstr():
    assert isclose(E_cstr(0, 1), 1)
    assert isclose(E_cstr(1e8, 1), 0)
    assert isclose(E_cstr(inf, 1), 0)
    assert isclose(E_cstr(1, 1), exp(-1))


def test_F_cstr():
    assert isclose(F_cstr(0, 1), 0)
    assert isclose(F_cstr(1e8, 1), 1)
    assert isclose(F_cstr(inf, 1), 1)
    assert isclose(F_cstr(1, 1), 0.631, rtol=0.01)


def test_E_tanks_series():
    # N = 1
    N = 1
    assert isclose(E_tanks_series(0, 1, N), 1)
    assert isclose(E_tanks_series(1e8, 1, N), 0)
    assert isclose(E_tanks_series(inf, 1, N), 0)
    assert isclose(E_tanks_series(1, 1, N), exp(-1))
    # N > 1
    for N in range(2, 10):
        assert isclose(E_tanks_series(0, 1, N), 0)
        assert isclose(E_tanks_series(1e8, 1, N), 0)
        assert isclose(E_tanks_series(inf, 1, N), 0)
    assert isclose(E_tanks_series(1, 1, 20), 1.8, atol=0.1)


def test_F_tanks_series():
    for N in range(1, 10):
        assert isclose(F_tanks_series(0, 1, N), 0)
        assert isclose(F_tanks_series(1e8, 1, N), 1)
        assert isclose(F_tanks_series(inf, 1, N), 1)
    assert isclose(F_tanks_series(1, 1, N=1), 0.631, rtol=0.01)
    assert isclose(F_tanks_series(0.85, 1, N=2), 0.5, atol=0.01)
    assert isclose(F_tanks_series(1, 1, N=20), 0.52, atol=0.01)
    assert isclose(F_tanks_series(0.5, 1, N=20), 0.0, atol=0.01)
    assert isclose(F_tanks_series(1.6, 1, N=20), 1.0, atol=0.01)


def test_E_laminar_flow():
    assert isclose(E_laminar_flow(0, 1), 0)
    assert isclose(E_laminar_flow(0.5-1e-8, 1), 0)
    assert isclose(E_laminar_flow(0.5, 1), 4)
    assert isclose(E_laminar_flow(1, 1), 0.5)
    assert isclose(E_laminar_flow(1e8, 1), 0)
    assert isclose(E_laminar_flow(inf, 1), 0)


def test_F_laminar_flow():
    assert isclose(F_laminar_flow(0, 1), 0)
    assert isclose(F_laminar_flow(0.5-1e-8, 1), 0)
    assert isclose(F_laminar_flow(0.5, 1), 0)
    assert isclose(F_laminar_flow(1, 1), 0.75)
    assert isclose(F_laminar_flow(1e8, 1), 1)
    assert isclose(F_laminar_flow(inf, 1), 1)


def test_E_dispersion_model():
    for Pe in [1, 10, 100, 1000]:
        assert isclose(E_dispersion_model(0, 1, Pe), 0)
        assert isclose(E_dispersion_model(1e8, 1, Pe), 0)
        assert isclose(E_dispersion_model(inf, 1, Pe), 0)
    assert isclose(E_dispersion_model(1, 1, 1/2e-4), 20, rtol=1e-2)
    assert isclose(E_dispersion_model(1, 1, 1/3.2e-3), 5, rtol=1e-2)
    assert isclose(E_dispersion_model(1, 1, 1/0.02), 2, atol=5e-2)
    assert isclose(E_dispersion_model(1, 1.5, 1/0.1), 0.48, atol=0.02)
    assert isclose(E_dispersion_model(0.5, 1, 1/0.2), 0.48, atol=0.02)


def test_F_dispersion_model():
    for Pe in [101, 200, 1000]:
        assert isclose(F_dispersion_model(0, 1, Pe), 0)
        assert isclose(F_dispersion_model(1, 1, Pe), 0.5)
        assert isclose(F_dispersion_model(1e8, 1, Pe), 1)
        assert isclose(F_dispersion_model(inf, 1, Pe), 1)
    assert isclose(F_dispersion_model(0.88, 1, 1/0.0099), 0.2, atol=0.01)
    assert isclose(F_dispersion_model(1.12, 1, 1/0.0099), 0.8, atol=0.01)

    for Pe in [1, 10, 99]:
        assert isclose(F_dispersion_model(0, 1, Pe), 0, atol=0.0)
        assert isclose(F_dispersion_model(inf, 1, Pe), 1, atol=0.01)

    assert isclose(F_dispersion_model(1, 1, 99.9), 0.5, atol=0.03)
    assert isclose(F_dispersion_model(10, 1, 99.9), 1, atol=0.01)


def test_Pe_tube():
    assert isclose(Pe_tube(1, 1), 1, rtol=1e-2)
    assert isclose(Pe_tube(2100, 1000), 1e-4, rtol=1e-1)
    # !!! test in turbulent is missing
