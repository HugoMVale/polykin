from numpy import isclose

from polykin.flow.prv import (
    area_relief_2phase,
    area_relief_2phase_subcooled,
    area_relief_gas,
    area_relief_liquid,
)


def test_area_relief_gas():
    # API 520, Example 1, p. 75
    W = 24270  # kg/h
    M = 51  # g/mol
    k = 1.11
    T = 348  # K
    P2 = 1.01325  # bara
    P1 = 6.70  # bara
    Z = 0.90
    res = area_relief_gas(W, P1, P2, T, k, M, Z=Z)
    assert isclose(res.Pcf, 3.92, rtol=1e-2)
    assert isclose(res.A, 3698, rtol=1e-2)

    # API 520, Example 2, p. 80
    P2 = 5.32  # bara
    res = area_relief_gas(W, P1, P2, T, k, M, Z=Z)
    assert isclose(res.A, 4226, rtol=1e-2)

    # API 520, Example 4, p. 91
    W = 69615  # kg/h
    M = 18.02  # g/mol
    k = 1.33
    T = 593  # K
    P1 = 122.36  # bara
    P2 = 1.01325  # bara
    res = area_relief_gas(W, P1, P2, T, k, M, steam=True)
    assert isclose(res.A, 1100, rtol=1e-2)
    # Compare with gas=water
    res2 = area_relief_gas(W, P1, P2, T, k, M, steam=False)
    assert isclose(res2.A, res.A, rtol=0.15)


def test_area_relief_liquid():
    # API 520, Example 5, p. 94
    Q = 6814  # L/min
    Gl = 0.9
    mu = 440 * Gl
    P1 = 18.96  # barg
    P2 = 3.45  # barg
    Kw = 0.97
    A = area_relief_liquid(Q, P1, P2, mu, Gl, Kw=Kw)
    assert isclose(A, 3180, rtol=1e-2)


def test_area_relief_2phase():
    # API 520, Example C.2.2.2, p. 136
    W = 216560  # kg/h
    v1 = 0.01945  # m³/kg
    v9 = 0.02265  # m³/kg
    Kv = 1.0
    Kb = 1.0
    P1 = 5.564  # bara
    P2 = 2.045  # bara
    res = area_relief_2phase(W, P1, P2, v1, v9, Kv=Kv, Kb=Kb)
    assert isclose(res.Pcf, 3.6721, rtol=1e-2)
    assert isclose(res.A, 24400, rtol=1e-2)


def test_area_relief_2phase_subcooled():
    # API 520, Example C.2.3, p. 143
    Q = 378.5  # L/min
    P1 = 20.733  # bara
    P2 = 1.703  # bara
    Ps = 7.419  # bara
    rho1 = 511.3  # kg/m³
    rho9 = 262.7  # kg/m³
    Kv = 1.0
    Kb = 1.0
    res = area_relief_2phase_subcooled(Q, P1, P2, Ps, rho1, rho9, Kb=Kb, Kv=Kv)
    assert res.critical_flow
    assert isclose(res.A, 134.5, rtol=1e-2)


def test_compare_gas():
    # For non-flashing flow, area_relief_gas =~ area_relief_2phase
    # Sonic flow
    W = 1000  # kg/h
    P1 = 5.0  # bara
    P2 = 1.0  # bara
    T = 300  # K
    k = 1.11
    M = 32  # g/mol
    Z = 1.0
    Kd = 0.85
    v1 = (8.314 * T) / (P1 * 1e5 * M * 1e-3)
    v9 = (8.314 * T) / (0.9 * P1 * 1e5 * M * 1e-3)
    res1 = area_relief_gas(W, P1, P2, T, k, M, Z=Z, Kd=Kd)
    res2 = area_relief_2phase(W, P1, P2, v1=v1, v9=v9, Kd=Kd)
    assert res1.critical_flow and res2.critical_flow
    assert isclose(res1.A, res2.A, rtol=0.05)
    # Subsonic flow
    P1 = 1.5
    v1 = (8.314 * T) / (P1 * 1e5 * M * 1e-3)
    v9 = (8.314 * T) / (0.9 * P1 * 1e5 * M * 1e-3)
    res1 = area_relief_gas(W, P1, P2, T, k, M, Z=Z, Kd=Kd)
    res2 = area_relief_2phase(W, P1, P2, v1=v1, v9=v9, Kd=Kd)
    assert not res1.critical_flow and not res2.critical_flow
    assert isclose(res1.A, res2.A, rtol=0.05)


def test_compare_subcooled_liquid():
    # For high subcooling, area_relief_liquid =~ area_relief_2phase_subcooled
    Q = 1e3  # L/min
    P1 = 10.0  # bara
    P2 = 1.0  # bara
    Ps = 0.1234  # bara (T1=50°C)
    mu = 1.0  # cP
    xV = 0.0037  # vapor fraction at P=0.9*Ps
    v1 = 1.012  # cm³/g
    v9 = (1 - xV) * v1 + xV * 12050
    rho1 = 1e3 / v1  # kg/m³
    rho9 = 1e3 / v9  # kg/m³
    Gl = rho1 / 1e3
    res1 = area_relief_liquid(Q, P1, P2, mu, Gl)
    res2 = area_relief_2phase_subcooled(Q, P1, P2, Ps, rho1, rho9)
    assert isclose(res1, res2.A, rtol=0.05)


def test_compare_saturated_liquid():
    # For saturated liquid, area_relief_2phase =~ area_relief_2phase_subcooled
    Q = 1e3  # L/min
    T1 = 150.0  # °C
    Ps = 4.76  # bara (Ps at T1)
    P1 = Ps  # bara
    P2 = 1.0  # bara
    v1 = 1.091e-3  # m³/kg
    xV = 4.2 * 4 / 2126
    v9 = (1 - xV) * 1.086e-3 + xV * 434.6e-3
    rho1 = 1 / v1
    rho9 = 1 / v9
    W = Q * rho1 * 1e-3 * 60
    Kd = 0.85
    res1 = area_relief_2phase(W, P1, P2, v1, v9, Kd=Kd)
    res2 = area_relief_2phase_subcooled(Q, P1, P2, Ps, rho1, rho9, Kd=Kd)
    assert isclose(res1.A, res2.A, rtol=0.05)
