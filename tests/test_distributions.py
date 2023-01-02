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
        assert (np.isclose(d.DPn, d.moment(1), rtol=1e-8))
        assert (np.isclose(d.DPw, d.moment(2)/d.moment(1), rtol=1e-8))
        assert (np.isclose(d.DPz, d.moment(3)/d.moment(2), rtol=1e-8))


def test_pdf():
    DPn = 69
    x = [i for i in range(1, 20*DPn)]
    distributions = {"flory": Flory(DPn), "poisson": Poisson(DPn)}
    for d in distributions.values():
        for dist in ["number", "mass", "gpc"]:
            pdf = d.pdf(x, dist=dist, unit_size='chain_length')
            assert (np.isclose(sum(pdf), 1.0, rtol=1e-3))


def test_cdf():
    DPn = 51
    x = [i for i in range(1, DPn+1)]
    distributions = {"flory": Flory(DPn), "poisson": Poisson(DPn)}
    for d in distributions.values():
        for dist in ["number", "mass", "gpc"]:
            pdf = d.pdf(x, dist=dist)
            cdf = d.cdf(DPn, dist=dist)
            sum_pdf = sum(pdf)
            assert (np.isclose(sum_pdf, cdf, rtol=1e-8))


def test_random():
    DPn = 49
    num_samples = 10**6
    distributions = {"flory": Flory(DPn), "poisson": Poisson(DPn)}
    for d in distributions.values():
        x = d.random(num_samples)
        for i in range(1, 4):
            mom = np.sum(x**i)/num_samples
            # print(mom, d.moment(i))
            assert (np.isclose(mom, d.moment(i), rtol=1e-2))


def test_composite_1():
    distributions = [Flory(64, 50), Poisson(34, 67)]
    for d in distributions:
        cases = [1*d, d*1, 2.0*d, d*3.0, d + d + d, d + 2*d + 3.0*d]
        for s in cases:
            assert (np.isclose(s.DPn, d.DPn, rtol=1e-15))
            assert (np.isclose(s.DPw, d.DPw, rtol=1e-15))
            assert (np.isclose(s.DPz, d.DPz, rtol=1e-15))
            # assert (np.isclose(s.Mn, d.Mn, rtol=1e-15))
            # assert (np.isclose(s.Mw, d.Mw, rtol=1e-15))
            # assert (np.isclose(s.Mz, d.Mz, rtol=1e-15))


def test_composite_2():
    DPn = 34
    f = Flory(DPn, 50)
    p = Poisson(DPn, 67)
    cases = [f + p + f, 1*f + p*2, f*2.0 + 3.0*p + 0.5*f]
    for s in cases:
        assert (np.isclose(s.DPn, DPn, rtol=1e-15))


def test_composite_3():
    DPw = 98
    f = Flory((DPw + 1)/2, 50)
    p = Poisson(DPw - 1, 67)
    cases = [f + p, 1*f + p*2 + f, f*2.0 + 3.0*p + 2*p]
    for s in cases:
        assert (np.isclose(s.DPw, DPw, rtol=1e-4))


def test_Flory():
    N = 123
    d = Flory(N)
    assert d.DPn == N
    assert np.isclose(d.DPw, 2 * N, atol=1)
    assert np.isclose(d.PDI, 2, rtol=1e-2)
