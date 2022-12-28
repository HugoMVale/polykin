from polykin.distributions import Flory, Poisson
import numpy as np


def test_init():
    distributions = {"flory": Flory(), "poisson": Poisson()}
    for name, item in distributions.items():
        for dist in ["number", "mass", "gpc"]:
            w = item(100, dist=dist)
            print(f"{name}, {dist}: {w}")
            assert w > 0


def test_Flory():
    N = 100
    d = Flory(N)
    assert d.DPn == N
    assert np.isclose(d.DPw, 2 * N, atol=1)
    assert np.isclose(d.PDI, 2, rtol=1e-2)
    d.show()
