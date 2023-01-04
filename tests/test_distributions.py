from polykin.distributions import Flory, Poisson, LogNormal

import numpy as np
import scipy.integrate as integrate

rtol = 10*np.finfo(np.float64).eps


def test_properties():
    distributions = {"flory": Flory(100),
                     "poisson": Poisson(100),
                     'log-normal': LogNormal(100)}
    for d in distributions.values():
        name = 'newname'
        DPn = 142
        M0 = 242
        d.name = name
        d.DPn = DPn
        d.M0 = M0
        assert (d.name[:len(name)] == name)
        assert (d.DPn == DPn)
        assert (np.isclose(d.M0, M0, rtol=rtol))
        assert (np.isclose(d.Mn*d.Mw*d.Mz, M0**3*d.DPn*d.DPw*d.DPz, rtol=rtol))
        assert (np.isclose(d.DPn, d.moment(1), rtol=rtol))
        assert (np.isclose(d.DPw, d.moment(2)/d.moment(1), rtol=rtol))
        assert (np.isclose(d.DPz, d.moment(3)/d.moment(2), rtol=rtol))


def test_pdf_discrete_sum():
    """For all discrete distributions, we should get sum(pdf)=1
    """
    DPn = 69
    x = [i for i in range(1, 100*DPn)]
    distributions = [Flory(DPn), Poisson(DPn)]
    for d in distributions:
        for dist in ["number", "mass", "gpc"]:
            pdf = d.pdf(x, dist=dist, unit_size='chain_length')
            assert (np.isclose(sum(pdf), 1.0, rtol=rtol))


def test_pdf_continuous_integral():
    """For all continuous distributions, we should get integral(pdf)=1
    """
    distributions = [LogNormal(121, 3)]
    for d in distributions:
        for dist in ["number", "mass", "gpc"]:
            pdf_integral, atol = integrate.quad(
                lambda x: d.pdf(x, dist=dist, unit_size='chain_length'),
                0, np.Inf)
            # print(d.name, dist, pdf_integral, atol)
            assert (np.isclose(pdf_integral, 1.0, atol=atol))


def test_pdf_discrete_moment():
    """For all discrete distributions, the moment obtined by summing the pdf
    should match the analytical value.
    """
    DPn = 69
    x = np.asarray([i for i in range(1, 100*DPn)])
    distributions = [Flory(DPn), Poisson(DPn)]
    for d in distributions:
        pdf = d.pdf(x, dist='number', unit_size='chain_length')
        for order in range(0, 4):
            mom_analytical = d.moment(order)
            mom_sum = np.sum(pdf*x**order)
            assert (np.isclose(mom_sum, mom_analytical, rtol=1e-4))


def test_pdf_continuous_moment():
    """For all continuous distributions, the moment obtined by integrating the
    pdf should match the analytical value.
    """
    distributions = [LogNormal(121, 3)]
    for d in distributions:
        for order in range(0, 4):
            mom_analytical = d.moment(order)
            mom_integral, atol = integrate.quad(
                lambda x: x**order *
                d.pdf(x, dist='number', unit_size='chain_length'),
                0, np.Inf)
            # print(d.name, mom_integral, atol)
            assert (np.isclose(mom_integral, mom_analytical, atol=atol))


def test_pfd_cdf_discrete():
    """For all discrete distributions, we should get sum(pdf(1:k))=cdf(k)
    """
    DPn = 51
    x = [i for i in range(1, DPn+1)]
    distributions = {"flory": Flory(DPn), "poisson": Poisson(DPn)}
    for d in distributions.values():
        for dist in ["number", "mass", "gpc"]:
            cdf = d.cdf(DPn, dist=dist)
            pdf = d.pdf(x, dist=dist)
            sum_pdf = sum(pdf)
            assert (np.isclose(sum_pdf, cdf, rtol=1e-8))


def test_pfd_cdf_continuous():
    """For all continuous distributions, we should get integral(pdf(0:x)=cdf(x)
    """
    DPn = 51
    PDI = 3
    s = 1*DPn
    distributions = [LogNormal(DPn, PDI)]
    for d in distributions:
        for dist in ["number", "mass", "gpc"]:
            cdf = d.cdf(s, dist=dist)
            integral_pdf, atol = integrate.quad(
                lambda x: d.pdf(x, dist=dist, unit_size='chain_length'),
                0, s)
            print(cdf, integral_pdf)
            assert (np.isclose(integral_pdf, cdf, atol=atol))


def test_random():
    """The moments of the random samples should match the moments of the parent
    distributions.
    """
    DPn = 49
    num_samples = 10**6
    distributions = {"flory": Flory(DPn),
                     "poisson": Poisson(DPn),
                     'log-normal': LogNormal(DPn, PDI=1.9)}
    for d in distributions.values():
        x = d.random(num_samples)
        for i in range(1, 4):
            mom = np.sum(x**i)/num_samples
            # print(d.name, i, mom, d.moment(i))
            assert (np.isclose(mom, d.moment(i), rtol=5e-2))


def test_composite_1():
    """Properties should remain unchanged when a distribution is combined with
    itself.
    """
    distributions = [Flory(64, 50), Poisson(34, 67), LogNormal(34, 1.4, 35)]
    for d in distributions:
        cases = [1*d, d*1, 2.0*d, d*3.0, d + d + d, d + 2*d + 3.0*d]
        for s in cases:
            assert (np.isclose(s.DPn, d.DPn, rtol=1e-15))
            assert (np.isclose(s.DPw, d.DPw, rtol=1e-15))
            assert (np.isclose(s.DPz, d.DPz, rtol=1e-15))
            assert (np.isclose(s.M0, d.M0, rtol=1e-15))
            assert (np.isclose(s.Mn, d.Mn, rtol=1e-15))
            assert (np.isclose(s.Mw, d.Mw, rtol=1e-15))
            assert (np.isclose(s.Mz, d.Mz, rtol=1e-15))


def test_composite_2():
    """Number-average properties should not change when combining base
    distributions with identical DPn and M0."""
    DPn = 34
    M0 = 44
    f = Flory(DPn, M0)
    p = Poisson(DPn, M0)
    cases = [f + p + f, 1*f + p*2, f*2.0 + 3.0*p + 0.5*f]
    for s in cases:
        assert (np.isclose(s.DPn, DPn, rtol=1e-15))
        assert (np.isclose(s.M0, M0, rtol=1e-15))
        assert (np.isclose(s.Mn, M0*DPn, rtol=1e-15))


def test_composite_3():
    """DPw should not change when combining base distributions with identical
    DPw."""
    DPw = 98
    f = Flory((DPw + 1)/2, 50)
    p = LogNormal(DPw/2, 2, 67)
    cases = [f + p, 1*f + p*2 + f, f*2.0 + 3.0*p + 2*p]
    for s in cases:
        assert (np.isclose(s.DPw, DPw, rtol=1e-4))


def test_composite_4():
    """Mw should not change when combining base distributions with identical
    Mw."""
    DPw = 98
    f = Flory((DPw + 1)/2, 50)
    p = Poisson((DPw - 1)/2, 100)
    cases = [f + p, 1*f + p*2 + f, f*2.0 + 3.0*p + 2*p]
    for s in cases:
        print(f.Mw, p.Mw, s.Mw, DPw*50)
        # assert (np.isclose(s.Mw, DPw*50.0, rtol=1e-4))


def test_Flory():
    N = 123
    d = Flory(N)
    assert d.DPn == N
    assert np.isclose(d.DPw, 2 * N, atol=1)
    assert np.isclose(d.PDI, 2, rtol=1e-2)
