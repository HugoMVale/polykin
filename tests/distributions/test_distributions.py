# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
import pytest
import scipy.integrate as integrate
from numpy import isclose

from polykin.distributions import (DataDistribution, Flory, LogNormal, Poisson,
                                   SchulzZimm, WeibullNycanderGold_pdf,
                                   convolve_moments, convolve_moments_self,
                                   plotdists)


@pytest.fixture
def dist1():
    return [Flory(90, 120, name='Flory'), Poisson(25, 42, name='Poisson')]


@pytest.fixture
def dist2():
    return [LogNormal(80, 3.0, 50, name='LogNormal'),
            SchulzZimm(90, 2.5, 69, name='Schulz-Zimm')]


# Default tolerance for "exact" comparisons
rtol = float(10*np.finfo(np.float64).eps)


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
        assert (isclose(d.M0, M0, rtol=rtol))
        assert (isclose(d.Mn*d.Mw*d.Mz, M0**3*d.DPn*d.DPw*d.DPz, rtol=rtol))


def test_inputs():
    with pytest.raises(ValueError):
        d = Poisson(0.99)
    with pytest.raises(ValueError):
        d = LogNormal(10, 1)
    with pytest.raises(ValueError):
        d = LogNormal(10, 2, M0=-10)

    d = Flory(100)
    with pytest.raises(ValueError):
        d.pdf(1, kind='notvalid')
    with pytest.raises(TypeError):
        d.pdf(1, kind=['mass'])
    with pytest.raises(TypeError):
        d.pdf(1, sizeasmass='notvalid')
    with pytest.raises(ValueError):
        d.cdf(1, kind='notvalid')
    with pytest.raises(TypeError):
        d.cdf(1, kind=['number'])
    with pytest.raises(TypeError):
        d.cdf(1, sizeas='notvalid')


def test_sort(dist1, dist2):
    distributions = dist1 + dist2
    d1 = distributions[0]
    for d2 in distributions[1:]:
        assert (d1 > d2) == (d1.Mw > d2.Mw)


def test_pdf_discrete_sum(dist1):
    """For all discrete distributions, we should get sum(pdf)=1
    """
    distributions = dist1
    DPn = 69
    x = np.arange(1, 100*DPn)
    for d in distributions:
        d.DPn = DPn
        for kind in ["number", "mass", "gpc"]:
            pdf = d.pdf(x, kind=kind)
            assert (isclose(sum(pdf), 1.0, rtol=rtol))


def test_pdf_continuous_integral(dist2):
    """For all continuous distributions, we should get integral(pdf)=1
    """
    distributions = dist2
    for d in distributions:
        for kind in ["number", "mass", "gpc"]:
            pdf_integral, atol = integrate.quad(
                lambda x: d.pdf(x, kind=kind),
                0, np.inf)
            # print(d.name, kind, pdf_integral, atol)
            assert (isclose(pdf_integral, 1.0, atol=atol))


def test_pdf_moment(dist1, dist2):
    """For all distributions, the moment obtined by integrating the pdf should
    match the analytical value.
    """
    distributions = dist1 + dist2
    for d in distributions:
        for order in range(0, 4):
            mom_analytical = d._moment_length(order)
            mom_numeric = super(type(d), d)._moment_length(order)
            # print(d.name, mom_integral, atol)
            assert (isclose(mom_numeric, mom_analytical, rtol=1e-4))


def test_pdf_length_mass(dist2):
    """For all continuous distributions, integrating between two chain lengths
    or between the corresponding molar masses should be identical.
    """
    distributions = dist2
    for d in distributions:
        length_range = np.asarray([int(d.DPn/2), int(d.DPw*2)])
        for kind in ['number', 'mass']:
            integral = {}
            xlimits = length_range
            for sizeasmass in [False, True]:
                if sizeasmass:
                    xlimits = xlimits*d.M0
                integral[sizeasmass], atol = integrate.quad(
                    lambda x: d.pdf(x, kind=kind, sizeasmass=sizeasmass),
                    xlimits[0], xlimits[1])
            # print(d.name, integral)
            assert (isclose(integral[False], integral[True], atol=atol))


def test_pfd_cdf(dist1, dist2):
    """For all distributions, we should get cdf(s)=sum(pdf(1:s)) or
    cdf(s)=integral(pdf(0:s).
    """
    distributions = dist1 + dist2
    for d in distributions:
        DPn = d.DPn
        for kind in ["number", "mass"]:
            order = d.kind_order[kind]
            cdf_analytical = d.cdf(DPn, kind=kind)
            cdf_generic = super(type(d), d)._cdf_length(DPn, order)
            # print(type(d), kind, cdf_analytical, cdf_generic)
            assert (isclose(cdf_analytical, cdf_generic, rtol=1e-6))


def test_random(dist1, dist2):
    """The moments of the random samples should match the analytical moments of
    the corresponding distributions.
    """
    distributions = dist1 + dist2
    DPn = 49
    PDI = 1.8
    num_samples = 10**6
    rtol = [None, 5e-3, 2e-2, 15e-2]
    for d in distributions:
        d.DPn = DPn
        if isinstance(d, (LogNormal, SchulzZimm)):
            d.PDI = PDI
        x = d.random(num_samples)
        for order in range(1, 4):
            mom = np.sum(x**order)/num_samples
            # print(type(d), order, mom, d.moment(order))
            assert (isclose(mom, d._moment_length(order), rtol=rtol[order]))


def test_composite_1(dist1, dist2):
    """Properties should remain unchanged when a distribution is combined with
    itself.
    """
    distributions = dist1 + dist2
    for d in distributions:
        cases = [1*d, d*1, 2.0*d, d*3.0, d + d + d, d + 2*d + 3.0*d]
        for s in cases:
            assert (isclose(s.DPn, d.DPn, rtol=rtol))
            assert (isclose(s.DPw, d.DPw, rtol=rtol))
            assert (isclose(s.DPz, d.DPz, rtol=rtol))
            assert (isclose(s.M0, d.M0, rtol=rtol))
            assert (isclose(s.Mn, d.Mn, rtol=rtol))
            assert (isclose(s.Mw, d.Mw, rtol=rtol))
            assert (isclose(s.Mz, d.Mz, rtol=rtol))


def test_composite_2a():
    """DPn should not change when combining distributions with same DPn."""
    DPn = 34
    f = Flory(DPn, 50)
    p = Poisson(DPn, 70)
    cases = [f + p + f, 1*f + p*2, f*2.0 + 3.0*p + 0.5*f]
    for s in cases:
        assert (isclose(s.DPn, DPn, rtol=rtol))


def test_composite_2b():
    """Mn should not change when combining distributions with same Mn."""
    Mn = 15000
    f = Flory(Mn/50, 50)
    p = Poisson(Mn/70, 70)
    cases = [f + p + f, 1*f + p*2, f*2.0 + 3.0*p + 0.5*f]
    for s in cases:
        assert (isclose(s.Mn, Mn, rtol=rtol))


def test_composite_2c():
    """Mw should not change when combining distributions with same Mw."""
    DPw = 98
    f = Flory((DPw + 1)/2, 100)
    p = LogNormal(DPw/2, 4, 50)
    cases = [f + p, 1*f + p*2 + f, f*2.0 + 3.0*p + 2*p]
    for s in cases:
        # print(f.Mw, p.Mw, s.Mw, DPw*100)
        assert (isclose(s.Mw, DPw*100.0, rtol=rtol))


def test_composite_pdf_integral():
    """The number/mass weights of the components should match the integrals of
    the corresponding peaks."""
    DPn = 100
    f = 3
    w1 = 0.4
    d1 = Poisson(DPn, M0=1.5, name='1')
    d2 = Poisson(DPn*f, M0=1, name='2')
    d = w1*d1 + (1-w1)*d2
    integral = {}
    xn = d._molefrac
    for kind in ['number', 'mass']:
        integral[kind], atol = integrate.quad(
            lambda x: d.pdf(x, kind=kind), 0, DPn*(1+f)/2)
    # print(xn, integral)
    assert (isclose(integral['number'], xn[0], rtol=1e-2))
    assert (isclose(integral['mass'], w1, rtol=1e-5))


def test_data_distribution():
    """The properties of a data distribution generated from a given analytical
    distribution should match those of the original distribution."""
    d1 = SchulzZimm(200, 1.5, M0=145, name='original')
    length_data = np.linspace(1, 5*d1.DPz, 1000)
    kind = 'mass'
    pdf_data = d1.pdf(length_data, kind=kind)
    d2 = DataDistribution(length_data, pdf_data,
                          kind=kind, M0=d1.M0, name='data')
    rtol = 1e-4
    for attr in ['DPn', 'DPw', 'DPz', 'Mn', 'Mw', 'Mz', 'PDI', 'M0']:
        y1 = getattr(d1, attr)
        y2 = getattr(d2, attr)
        # print(attr, y1, y2)
        assert (isclose(y2, y1, rtol=rtol))
    assert (isclose(d2.pdf(d2.DPw), d1.pdf(d1.DPw), rtol=rtol))
    assert (isclose(d2.cdf(d2.DPw), d1.cdf(d1.DPw), rtol=rtol))


def test_fit_itself(dist1, dist2):
    """The fit function should provide an (almost) exact fit of itself."""
    distributions = dist1 + dist2
    kind = 'mass'
    rng = np.random.default_rng()
    rnoise = 5e-2
    rtol = 1e-1
    for d in distributions:
        length_data = np.linspace(1, 5*d.DPz, 1000)
        pdf_data = d.pdf(length_data, kind=kind)
        pdf_data *= rng.uniform(1-rnoise, 1+rnoise, pdf_data.size)
        d2 = DataDistribution(length_data, pdf_data,
                              kind=kind, M0=d.M0, name='data'+d.name)
        dfit = d2.fit(type(d), 1)
        for attr in ['DPn', 'DPw', 'DPz', 'Mn', 'Mw', 'Mz', 'PDI', 'M0']:
            assert (isclose(getattr(dfit, attr), getattr(d, attr),
                            rtol=rtol))


def test_plot_method(dist1):
    out = dist1[0].plot()
    for kind in ['number', 'mass', 'gpc']:
        out = dist1[0].plot(kind=kind)
    for cdf in [0, 1, 2]:
        out = dist1[0].plot(cdf=cdf)


def test_plotdists(dist1):
    plotdists(dist1, kind='gpc')


def test_WeibullNycanderGold_pdf():
    v = 10.
    c = 2.
    s = np.arange(4*int(v))
    Ps = WeibullNycanderGold_pdf(s, v, c)
    assert s.shape == Ps.shape
    assert isclose(Ps.sum(), 1.0)
    # examples from Gold (1957)
    r = 2.
    p0 = 0.3679
    res = _moments(p0, r)
    assert np.all(isclose(res, (2.164, 2.849), rtol=1e-3))
    r = 10.
    p0 = 0.4594
    res = _moments(p0, r)
    assert np.all(isclose(res, (5.388, 7.110), rtol=1e-3))


def _moments(p0, r):
    v = - r*np.log(p0) - (r - 1.)*(1. - p0)
    s = np.arange(1, 10*int(v))
    Ps = WeibullNycanderGold_pdf(s, v, r)
    m0 = Ps.sum()
    m1 = np.dot(s, Ps)
    m2 = np.dot(s**2, Ps)
    xn = m1/m0
    xw = m2/m1
    return (xn, xw)


def test_convolve_moments():

    # Define inputs
    q0 = 1.0
    q1 = q0 * 100.0
    q2 = q1 * 200.0

    p0, p1, p2 = convolve_moments(q0, q1, q2, q0, q1, q2)

    assert isclose(p0*p2/p1**2, 1.5)


def test_convolve_moments_self():

    def convolve_moments_self_iter(q0, q1, q2, order):
        r0 = q0
        r1 = q1
        r2 = q2
        for _ in range(order):
            r0, r1, r2 = convolve_moments(q0, q1, q2, r0, r1, r2)
        return r0, r1, r2

    for order in range(1, 10):
        q0 = 1.0
        q1 = q0 * 100.0
        q2 = q1 * 200.0
        s0, s1, s2 = convolve_moments_self_iter(q0, q1, q2, order)
        p0, p1, p2 = convolve_moments_self(q0, q1, q2, order)
        assert np.all(isclose([s0, s1, s2], [p0, p1, p2]))
