from polykin.distributions import Flory, Poisson, LogNormal, SchulzZimm

import numpy as np
import scipy.integrate as integrate
from copy import copy
import pytest

# Default tolerance for "exact" comparisons
rtol = float(10*np.finfo(np.float64).eps)

# Default distributions for tests


@pytest.fixture
def dist1():
    return [Flory(90, 120, name='Flory'), Poisson(25, 42, name='Poisson')]


@pytest.fixture
def dist2():
    return [LogNormal(80, 3.0, 50, name='LogNormal'),
            SchulzZimm(90, 2.5, 69, name='Schulz-Zimm')]


def test_properties(dist1, dist2):
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


def test_inputs():
    with pytest.raises(ValueError):
        d = Poisson(1)
    with pytest.raises(ValueError):
        d = LogNormal(10, 1)
    with pytest.raises(ValueError):
        d = LogNormal(10, 2, M0=-10)
    with pytest.raises(TypeError):
        d = LogNormal(10, 2, name=123)

    d = Flory(100)
    with pytest.raises(ValueError):
        d.pdf(1, type='notvalid')
    with pytest.raises(TypeError):
        d.pdf(1, sizeasmass='notvalid')
    with pytest.raises(ValueError):
        d.cdf(1, type='notvalid')
    with pytest.raises(TypeError):
        d.cdf(1, sizeas='notvalid')


def test_pdf_discrete_sum(dist1):
    """For all discrete distributions, we should get sum(pdf)=1
    """
    distributions = dist1
    DPn = 69
    x = [i for i in range(1, 100*DPn)]
    for d in distributions:
        d.DPn = DPn
        for type in ["number", "mass", "gpc"]:
            pdf = d.pdf(x, type=type)
            assert (np.isclose(sum(pdf), 1.0, rtol=rtol))


def test_pdf_continuous_integral(dist2):
    """For all continuous distributions, we should get integral(pdf)=1
    """
    distributions = dist2
    for d in distributions:
        for type in ["number", "mass", "gpc"]:
            pdf_integral, atol = integrate.quad(
                lambda x: d.pdf(x, type=type),
                0, np.Inf)
            # print(d.name, dist, pdf_integral, atol)
            assert (np.isclose(pdf_integral, 1.0, atol=atol))


def test_pdf_discrete_moment(dist1):
    """For all discrete distributions, the moment obtined by summing the pdf
    should match the analytical value.
    """
    distributions = dist1
    DPn = 69
    x = np.asarray([i for i in range(1, 100*DPn)])
    for d in distributions:
        d.DPn = DPn
        pdf = d.pdf(x, type='number')
        for order in range(0, 4):
            mom_analytical = d._moment_length(order)
            mom_sum = np.sum(pdf*x**order)
            assert (np.isclose(mom_sum, mom_analytical, rtol=1e-4))


def test_pdf_continuous_moment(dist2):
    """For all continuous distributions, the moment obtined by integrating the
    pdf should match the analytical value.
    """
    distributions = dist2
    for d in distributions:
        for order in range(0, 4):
            mom_analytical = d._moment_length(order)
            mom_integral, atol = integrate.quad(
                lambda x: x**order *
                d.pdf(x, type='number'),
                0, np.Inf)
            # print(d.name, mom_integral, atol)
            assert (np.isclose(mom_integral, mom_analytical, atol=atol))


def test_pdf_length_mass(dist2):
    """For all distributions, integrating between two chain lengths or between
    the corresponding molar masses should be identical.
    """
    print("")
    distributions = dist2
    for d in distributions:
        length_range = np.asarray([int(d.DPn/2), int(d.DPw*2)])
        for type in ['number', 'mass']:
            integral = {}
            xlimits = length_range
            for sizeasmass in [False, True]:
                if sizeasmass:
                    xlimits = xlimits*d.M0
                integral[sizeasmass], atol = integrate.quad(
                    lambda x: d.pdf(x, type=type, sizeasmass=sizeasmass),
                    xlimits[0], xlimits[1])
            # print(d.name, integral)
            assert (np.isclose(integral[False], integral[True], atol=atol))


def test_pfd_cdf_discrete(dist1):
    """For all discrete distributions, we should get sum(pdf(1:k))=cdf(k)
    """
    distributions = dist1
    DPn = 51
    x = [i for i in range(1, DPn+1)]
    for d in distributions:
        d.DPn = DPn
        for type in ["number", "mass"]:
            cdf = d.cdf(DPn, type=type)
            pdf = d.pdf(x, type=type)
            sum_pdf = sum(pdf)
            assert (np.isclose(sum_pdf, cdf, rtol=1e-8))


def test_pfd_cdf_continuous(dist2):
    """For all continuous distributions, we should get integral(pdf(0:s)=cdf(s)
    """
    distributions = dist2
    DPn = 51
    PDI = 2.9
    s = 1*DPn
    for d in distributions:
        d.DPn = DPn
        d.PDI = PDI
        for type in ["number", "mass"]:
            cdf = d.cdf(s, type=type)
            integral_pdf, atol = integrate.quad(
                lambda x: d.pdf(x, type=type),
                0, s)
            # print(cdf, integral_pdf)
            assert (np.isclose(integral_pdf, cdf, atol=atol))


def test_random(dist1, dist2):
    """The moments of the random samples should match the analytical moments of
    the corresponding distributions.
    """
    distributions = dist1 + dist2
    DPn = 49
    PDI = 1.8
    num_samples = 10**6
    rtol = [None, 5e-3, 2e-2, 5e-2]
    for d in distributions:
        d.DPn = DPn
        try:
            d.PDI = PDI
        except AttributeError:
            pass
        x = d.random(num_samples)
        for order in range(1, 4):
            mom = np.sum(x**order)/num_samples
            # print(type(d), order, mom, d.moment(order))
            assert (np.isclose(mom, d._moment_length(order), rtol=rtol[order]))


def test_composite_1(dist1, dist2):
    """Properties should remain unchanged when a distribution is combined with
    itself.
    """
    distributions = dist1 + dist2
    for d in distributions:
        cases = [1*d, d*1, 2.0*d, d*3.0, d + d + d, d + 2*d + 3.0*d]
        for s in cases:
            assert (np.isclose(s.DPn, d.DPn, rtol=rtol))
            assert (np.isclose(s.DPw, d.DPw, rtol=rtol))
            assert (np.isclose(s.DPz, d.DPz, rtol=rtol))
            assert (np.isclose(s.M0, d.M0, rtol=rtol))
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
    """Mw should not change when combining distributions with same Mw."""
    DPw = 98
    f = Flory((DPw + 1)/2, 100)
    p = LogNormal(DPw/2, 4, 50)
    cases = [f + p, 1*f + p*2 + f, f*2.0 + 3.0*p + 2*p]
    for s in cases:
        # print(f.Mw, p.Mw, s.Mw, DPw*100)
        assert (np.isclose(s.Mw, DPw*100.0, rtol=rtol))


def test_composite_pdf_integral():
    """The mnumber/mass weights of the components should match the integrals of
    the corresponding peaks."""
    DPn = 100
    f = 3
    w1 = 0.4
    d1 = Poisson(DPn, M0=1.5, name='1')
    d2 = Poisson(DPn*f, M0=1, name='2')
    d = w1*d1 + (1-w1)*d2
    integral = {}
    xn = d._calc_molefracs()
    for type in ['number', 'mass']:
        integral[type], atol = integrate.quad(
            lambda x: d.pdf(x, type=type), 0, DPn*(1+f)/2)
    # print(xn, integral)
    assert (np.isclose(integral['number'], xn[0], rtol=1e-2))
    assert (np.isclose(integral['mass'], w1, rtol=1e-5))


def test_fit_itself(dist1, dist2):
    """The fit function should provide an exact fit of itself."""
    distributions = dist1 + dist2
    rng = np.random.default_rng()
    rnoise = 5e-2
    for d in distributions:
        x = np.linspace(1, 2*d.DPw, 100)
        for type in ["number", "mass", "gpc"]:
            y = d.pdf(x, type=type)
            dy = rng.uniform(1-rnoise, 1+rnoise, y.size)
            y *= dy
            d2 = copy(d)
            d2.DPn = 10
            try:
                d.PDI = 10
            except AttributeError:
                pass
            d2.fit(x, y, type=type, sizeasmass=False)
            # print(d2.DPn, d.DPn)
        assert (np.isclose(d2.DPn, d.DPn, rtol=1e-1))
        assert (np.isclose(d2.PDI, d.PDI, rtol=1e-1))
