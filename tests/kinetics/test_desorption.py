from numpy import isclose

from polykin.kinetics.emulsion.desorption import K0_Nomura, kdesorption_Asua


def test_kdesorption_Asua():
    kdes = kdesorption_Asua(kfmp=1e-3, kpp=1e2, Mp=10, K0=1e3, beta=0.20)
    assert isclose(kdes, 8.33e-3, rtol=1e-2)


def test_K0_Nomura():
    K0 = K0_Nomura(Dw=1e-9, Dp=1e-10, q=30, dp=200e-9)
    assert isclose(K0, 9375.0, rtol=1e-3)
