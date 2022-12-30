from polykin.distributions import Flory, Poisson
import numpy as np


def test_properties():
    distributions = {"flory": Flory(), "poisson": Poisson()}
    for d in distributions.values():
        name = 'newname'
        DPn = 142
        M0 = 242
        d.name = name
        d.DPn = DPn
        d.M0 = M0
        assert (d.name[:len(name)] == name)
        assert (d.DPn == DPn)
        assert (np.isclose(d.M0, M0, rtol=1e-8))
        assert (np.isclose(d.Mn*d.Mw*d.Mz, M0**3*d.DPn*d.DPw*d.DPz, rtol=1e-8))


def test_pmf():
    DPn = 100
    x = [i for i in range(1, 20*DPn)]
    distributions = {"flory": Flory(DPn), "poisson": Poisson(DPn)}
    for d in distributions.values():
        for dist in ["number", "mass", "gpc"]:
            y = d(x, dist=dist, unit_x='chain_length')
            assert (np.isclose(sum(y), 1.0, rtol=1e-3))


def test_Flory():
    N = 123
    d = Flory(N)
    assert d.DPn == N
    assert np.isclose(d.DPw, 2 * N, atol=1)
    assert np.isclose(d.PDI, 2, rtol=1e-2)
