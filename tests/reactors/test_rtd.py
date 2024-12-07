from numpy import exp, isclose

from polykin.reactors.rtd import *


def test_cstr_differential():
    assert isclose(cstr_differential(0, 1), 1)
    assert isclose(cstr_differential(1e8, 1), 0)
    assert isclose(cstr_differential(1, 1), exp(-1))


def testcstr_cumulative():
    assert isclose(cstr_cumulative(0, 1), 0)
    assert isclose(cstr_cumulative(1e8, 1), 1)
    assert isclose(cstr_cumulative(1, 1), 0.63, rtol=0.01)


def test_tanks_series_differential():
    # N = 1
    N = 1
    assert isclose(tanks_series_differential(0, 1, N), 1)
    assert isclose(tanks_series_differential(1e8, 1, N), 0)
    assert isclose(tanks_series_differential(1, 1, N), exp(-1))
    # N > 1
    for N in range(2, 10):
        assert isclose(tanks_series_differential(0, 1, N), 0)
        assert isclose(tanks_series_differential(1e8, 1, N), 0)
    assert isclose(tanks_series_differential(1, 1, 20), 1.8, atol=0.1)


def test_tanks_series_cumulative():
    for N in range(1, 10):
        assert isclose(tanks_series_cumulative(0, 1, N), 0)
        assert isclose(tanks_series_cumulative(1e8, 1, N), 1)
    assert isclose(tanks_series_cumulative(1, 1, N=1), 0.63, rtol=0.01)
    assert isclose(tanks_series_cumulative(0.85, 1, N=2), 0.5, atol=0.01)
    assert isclose(tanks_series_cumulative(1, 1, N=20), 0.52, atol=0.01)
    assert isclose(tanks_series_cumulative(0.5, 1, N=20), 0.0, atol=0.01)
    assert isclose(tanks_series_cumulative(1.6, 1, N=20), 1.0, atol=0.01)
