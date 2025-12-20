import pytest
from numpy import isclose

from polykin.flow.rheology import (
    aT_WLF,
    mu_Carreau_Yasuda,
    mu_Cross,
    mu_Cross_modified,
    mu_PowerLaw,
)


def test_mu_PowerLaw():
    K = 10.0  # Pa·s^n
    n = 0.2
    assert isclose(mu_PowerLaw(1.0, K, n), K)


def test_mu_Carreau_Yasuda():
    mu0 = 1.0  # Pa·s
    muinf = 0.001  # Pa·s
    lmbda = 1.0  # s
    n = 0.2
    a = 2.0
    assert isclose(mu_Carreau_Yasuda(0, mu0, muinf, lmbda, n, a), mu0)
    assert isclose(mu_Carreau_Yasuda(1e99, mu0, muinf, lmbda, n, a), muinf)
    assert isclose(mu_Carreau_Yasuda(1e6, mu0, muinf, lmbda, n, a), 1e-3, rtol=0.1)
    assert isclose(mu_Carreau_Yasuda(1e2, mu0, muinf, lmbda, n, a), 2.5e-2, rtol=0.1)


def test_mu_Cross():
    mu0 = 1.0  # Pa·s
    lmbda = 1.0  # s
    n = 0.2
    assert isclose(mu_Cross(0, mu0, lmbda, n), mu0)
    assert isclose(mu_Cross(1e99, mu0, lmbda, n), 0.0)
    assert isclose(
        mu_Cross(10.0, mu0, lmbda, n), mu_Cross_modified(10.0, mu0, lmbda / mu0, n)
    )


def test_mu_Cross_modified():
    mu0 = 1.34e7  # Pa·s
    C = 2.2e-5  # 1/Pa
    n = 0.21
    assert isclose(mu_Cross_modified(0, mu0, C, n), mu0)
    assert isclose(mu_Cross_modified(1e99, mu0, C, n), 0.0)
    assert isclose(mu_Cross_modified(1e-2, mu0, C, n), 4e6, rtol=0.1)
    assert isclose(mu_Cross_modified(1e0, mu0, C, n), 1.5e5, rtol=0.1)


def test_aT_WLF():
    # Example 9, Stephen L. Rosen
    Tg = -70 + 273.15  # K
    isclose(aT_WLF(25 + 273.15, Tg), 5.01e-12, rtol=1e-2)
    isclose(aT_WLF(-20 + 273.15, Tg) / aT_WLF(25 + 273.15, Tg) * 10, 5140, rtol=1e-2)
    # User defined C1, C2
    isclose(aT_WLF(-20 + 273.15, Tg), aT_WLF(25 + 273.15, Tg, C1=17.44, C2=51.6))
    # Missing C1 or C2
    with pytest.raises(ValueError):
        _ = aT_WLF(320.0, 300.0, C1=15.0)
    with pytest.raises(ValueError):
        _ = aT_WLF(320.0, 300.0, C2=50.0)
