from polykin.distributions import Flory, Poisson, LogNormal, SchulzZimm

import numpy as np
import scipy.integrate as integrate

# Default tolerance for "exact" comparisons
rtol = float(10*np.finfo(np.float64).eps)

# Default distributions for tests
dist1 = [Flory(90), Poisson(25)]
dist2 = [LogNormal(80, 3.0), SchulzZimm(90, 2.5)]


def test_properties():
    distributions = dist1 + dist2
    for d in distributions:
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
    distributions = dist1
    for d in distributions:
        d.DPn = DPn
        for dist in ["number", "mass", "gpc"]:
            pdf = d.pdf(x, dist=dist)
            assert (np.isclose(sum(pdf), 1.0, rtol=rtol))


def test_pdf_continuous_integral():
    """For all continuous distributions, we should get integral(pdf)=1
    """
    distributions = dist2
    for d in distributions:
        for dist in ["number", "mass", "gpc"]:
            pdf_integral, atol = integrate.quad(
                lambda x: d.pdf(x, dist=dist),
                0, np.Inf)
            # print(d.name, dist, pdf_integral, atol)
            assert (np.isclose(pdf_integral, 1.0, atol=atol))


def test_pdf_discrete_moment():
    """For all discrete distributions, the moment obtined by summing the pdf
    should match the analytical value.
    """
    DPn = 69
    x = np.asarray([i for i in range(1, 100*DPn)])
    distributions = dist1
    for d in distributions:
        d.DPn = DPn
        pdf = d.pdf(x, dist='number')
        for order in range(0, 4):
            mom_analytical = d.moment(order)
            mom_sum = np.sum(pdf*x**order)
            assert (np.isclose(mom_sum, mom_analytical, rtol=1e-4))


def test_pdf_continuous_moment():
    """For all continuous distributions, the moment obtined by integrating the
    pdf should match the analytical value.
    """
    distributions = dist2
    for d in distributions:
        for order in range(0, 4):
            mom_analytical = d.moment(order)
            mom_integral, atol = integrate.quad(
                lambda x: x**order *
                d.pdf(x, dist='number'),
                0, np.Inf)
            # print(d.name, mom_integral, atol)
            assert (np.isclose(mom_integral, mom_analytical, atol=atol))


def test_pfd_cdf_discrete():
    """For all discrete distributions, we should get sum(pdf(1:k))=cdf(k)
    """
    DPn = 51
    x = [i for i in range(1, DPn+1)]
    distributions = dist1
    for d in distributions:
        d.DPn = DPn
        for dist in ["number", "mass", "gpc"]:
            cdf = d.cdf(DPn, dist=dist)
            pdf = d.pdf(x, dist=dist)
            sum_pdf = sum(pdf)
            assert (np.isclose(sum_pdf, cdf, rtol=1e-8))


def test_pfd_cdf_continuous():
    """For all continuous distributions, we should get integral(pdf(0:s)=cdf(s)
    """
    DPn = 51
    PDI = 2.9
    s = 1*DPn
    distributions = dist2
    for d in distributions:
        d.DPn = DPn
        d.PDI = PDI
        for dist in ["number", "mass", "gpc"]:
            cdf = d.cdf(s, dist=dist)
            integral_pdf, atol = integrate.quad(
                lambda x: d.pdf(x, dist=dist),
                0, s)
            print(cdf, integral_pdf)
            assert (np.isclose(integral_pdf, cdf, atol=atol))


def test_random():
    """The moments of the random samples should match the analytical moments of
    the corresponding distributions.
    """
    DPn = 49
    PDI = 1.8
    num_samples = 10**6
    rtol = [None, 5e-3, 2e-2, 5e-2]
    distributions = dist1 + dist2
    for d in distributions:
        d.DPn = DPn
        try:
            d.PDI = PDI
        except AttributeError:
            pass
        x = d.random(num_samples)
        for order in range(1, 4):
            mom = np.sum(x**order)/num_samples
            print(type(d), order, mom, d.moment(order))
            assert (np.isclose(mom, d.moment(order), rtol=rtol[order]))


def test_composite_1():
    """Properties should remain unchanged when a distribution is combined with
    itself.
    """
    distributions = [Flory(64, 50), Poisson(34, 67), LogNormal(34, 1.4, 35)]
    for d in distributions:
        cases = [1*d, d*1, 2.0*d, d*3.0, d + d + d, d + 2*d + 3.0*d]
        for s in cases:
            assert (np.isclose(s.DPn, d.DPn, rtol=rtol))
            assert (np.isclose(s.DPw, d.DPw, rtol=rtol))
            assert (np.isclose(s.DPz, d.DPz, rtol=rtol))
            # assert (np.isclose(s.M0, d.M0, rtol=rtol))
            assert (np.isclose(s.Mn, d.Mn, rtol=rtol))
            assert (np.isclose(s.Mw, d.Mw, rtol=rtol))
            assert (np.isclose(s.Mz, d.Mz, rtol=rtol))


def test_composite_2a():
    """DPn should not change when combining distributions with same DPn."""
    DPn = 34
    f = Flory(DPn, 50)
    p = Poisson(DPn, 70)
    cases = [f + p + f, 1*f + p*2, f*2.0 + 3.0*p + 0.5*f]
    for s in cases:
        assert (np.isclose(s.DPn, DPn, rtol=rtol))


def test_composite_2b():
    """Mn should not change when combining distributions with same Mn."""
    Mn = 15000
    f = Flory(Mn/50, 50)
    p = Poisson(Mn/70, 70)
    cases = [f + p + f, 1*f + p*2, f*2.0 + 3.0*p + 0.5*f]
    for s in cases:
        assert (np.isclose(s.Mn, Mn, rtol=rtol))


def test_composite_2c():
    """DPw should not change when combining distributions with same DPw."""
    DPw = 98
    f = Flory((DPw + 1)/2, 50)
    p = LogNormal(DPw/2, 2, 67)
    cases = [f + p, 1*f + p*2 + f, f*2.0 + 3.0*p + 2*p]
    for s in cases:
        assert (np.isclose(s.DPw, DPw, rtol=rtol))


def test_composite_2d():
    """Mw should not change when combining distributions with same Mw."""
    DPw = 98
    f = Flory((DPw + 1)/2, 100)
    p = LogNormal(DPw/2, 4, 50)
    cases = [f + p, 1*f + p*2 + f, f*2.0 + 3.0*p + 2*p]
    for s in cases:
        print(f.Mw, p.Mw, s.Mw, DPw*100)
        assert (np.isclose(s.Mw, DPw*100.0, rtol=rtol))
