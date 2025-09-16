from numpy import isclose

from polykin.flow.prv import area_relief_gas, area_relief_liquid


def test_area_relief_gas():
    # API 520, Example 1, p. 60
    W = 24270    # kg/h
    M = 51       # g/mol
    k = 1.11
    T = 348      # K
    P2 = 1.01325  # bara
    P1 = 6.70    # bara
    Z = 0.90
    res = area_relief_gas(W, P1, P2, T, k, M, Z)
    print(res)
    assert isclose(res['Pcf'], 392, rtol=1e-2)
    assert isclose(res['A'], 3698, rtol=1e-2)

    # API 520, Example 2, p. 64
    P2 = 5.32  # bara
    res = area_relief_gas(W, P1, P2, T, k, M, Z)
    assert isclose(res['A'], 4226, rtol=1e-2)

    # API 520, Example 4, p. 69
    W = 69615    # kg/h
    M = 18       # g/mol
    k = 1.33
    T = 593      # K
    P1 = 122.36  # bara
    P2 = 1.01325  # bara
    res = area_relief_gas(W, P1, P2, T, k, M, steam=True)
    assert isclose(res['A'], 1100, rtol=1e-2)


def test_area_relief_liquid():
    # API 520, Example 5, p. 73
    Q = 6814  # L/min
    Gl = 0.9
    mu = 440*Gl
    P1 = 18.96  # barg
    P2 = 3.45   # barg
    Kw = 0.97
    A = area_relief_liquid(Q, P1, P2, mu, Gl, Kw=Kw)
    assert isclose(A, 3180, rtol=1e-2)
