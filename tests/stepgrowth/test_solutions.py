# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from numpy import all, isclose

from polykin.stepgrowth import (Case_1, Case_3, Case_5, Case_6, Case_7, Case_8,
                                Case_9, Case_10, Case_11, Stockmayer)


def test_Case_1():
    Mn, Mw = Case_1(0.9, 1., 42., 42.)
    assert isclose(Mn, 420.)
    assert isclose(Mw/Mn, 2, rtol=0.05)
    Mn, Mw = Case_1(0.9999, 1., 39., 54.)
    assert isclose(Mw/Mn, 2, rtol=1e-3)
    Mn, _ = Case_1(0.5, 1., 42., 0.)
    assert isclose(Mn, 42.)
    Mn, _ = Case_1(0.5, 1., 0., 33.)
    assert isclose(Mn, 33.)
    Mn, _ = Case_1(0.5, 1., 20., 30.)
    assert isclose(Mn, 50.)
    # AA-BB-AA
    Mn, _ = Case_1(1., 0.5, 20., 30.)
    assert isclose(Mn, 20+30+20)


def test_Case_3():
    p = 0.97
    r = 1.2
    MAA = 42
    MBB = 56
    sol1 = Case_1(p, r, MAA, MBB)
    sol3 = Case_3(p, p, r, MAA, MBB)
    assert all(isclose(sol1, sol3))  # type:ignore
    Mn, _ = Case_3(1., 0, 2., MAA, MBB)
    assert isclose(Mn, MAA + 2*MBB)
    Mn, _ = Case_3(0., 1, 2., MAA, MBB)
    assert isclose(Mn, MAA + 2*MBB)


def test_Case_5():
    MAA = 123.
    MBB = 45.
    MC = 38.
    # ... AA-BB-AA-BB-AA-BB ...
    for pB in [0.5, 0.9, 0.999]:
        for rBA in [0.5, 1.]:
            Mn0, _ = Case_1(pB, rBA, MAA, MBB)
            Mn = Case_5(pB=pB, pC=0., r_BC_A=rBA, r_C_B=0,
                        MAA=MAA, MBB=MBB, MC=MC)
            assert isclose(Mn0, Mn)
    # C-AA-C
    Mn = Case_5(pB=0, pC=1., r_BC_A=1., r_C_B=1e10, MAA=MAA, MBB=MBB, MC=MC)
    assert isclose(Mn, MAA+2*MC)
    # AA-BB-AA
    Mn = Case_5(pB=1., pC=0., r_BC_A=0.5, r_C_B=0., MAA=MAA, MBB=MBB, MC=MC)
    assert isclose(Mn, MBB+2*MAA)
    # C-AA-BB-AA-C
    # Mn = Case_5(pB=1., pC=1., r_BC_A=1., r_C_B=1., MAA=MAA, MBB=MBB, MC=WC)
    # assert isclose(Mn, MBB+2*MAA+2*WC)


def test_Case_6():
    MAA = 85.
    MBC = 45.
    # ...BC-BC-BC-BC...
    Mn = Case_6(pC=0.99, r_BC_AA=1e10, MAA=MAA, MBC=MBC)
    assert isclose(Mn, 100*MBC)


def test_Case_7():
    MAB = 85.
    MCD = 45.
    # ...BC-BC-BC-BC...
    Mn = Case_7(pA=0.99, pC=0, r_CD_AB=0, MAB=MAB, MCD=MCD)
    assert isclose(Mn, 100*MAB)
    Mn = Case_7(pA=0, pC=0.99, r_CD_AB=1e10, MAB=MAB, MCD=MCD)
    assert isclose(Mn, 100*MCD)
    Mn = Case_7(pA=0.99, pC=0.99, r_CD_AB=1, MAB=MAB, MCD=MCD)
    assert isclose(Mn, 100*(MAB + MCD)/2)


def test_Case_8():
    MAA = 123.
    MBB = 45.
    MCC = 83.
    # ... AA-BB-AA-BB-AA-BB ...
    for pB in [0.5, 0.9, 0.999]:
        for rBA in [0.5, 1.]:
            # AA + BB
            Mn0, _ = Case_1(pB, rBA, MAA, MBB)
            Mn = Case_8(pB=pB, pC=0., r_BC_A=rBA, r_CC_BB=0,
                        MAA=MAA, MBB=MBB, MCC=MCC)
            assert isclose(Mn0, Mn)
            # AA + CC
            Mn0, _ = Case_1(pB, rBA, MAA, MCC)
            Mn = Case_8(pB=0, pC=pB, r_BC_A=rBA, r_CC_BB=1e10,
                        MAA=MAA, MBB=MBB, MCC=MCC)
            assert isclose(Mn0, Mn)
            # AA + 0.5(BB + CC)
            Mn0, _ = Case_1(pB, rBA, MAA, (MBB + MCC)/2)
            Mn = Case_8(pB=pB, pC=pB, r_BC_A=rBA, r_CC_BB=1,
                        MAA=MAA, MBB=MBB, MCC=MCC)
            assert isclose(Mn0, Mn)


def test_Case_9():
    MAA = 123.
    MBB = 45.
    MCC = 83.
    MDD = 65.
    for pB in [0.5, 0.9, 0.999]:
        for pC in [0.4, 0.8, 0.99]:
            for rCB in [0.5, 1., 2.]:
                for rCA in [0.8, 1.]:
                    # AA + (CC + DD)
                    Mn0 = Case_8(pB=pB, pC=pC, r_BC_A=rCA, r_CC_BB=rCB,
                                 MAA=MAA, MBB=MCC, MCC=MDD)
                    Mn = Case_9(pB=0, pC=pB, pD=pC, r_BB_AA=0, r_DD_CC=rCB,
                                r_CD_AB=rCA, MAA=MAA, MBB=MBB, MCC=MCC,
                                MDD=MDD)
                    assert isclose(Mn0, Mn)


def test_Case_10():
    MAA = 123.
    MBC = 83.
    MDD = 65.
    # AA + DD
    pD = 0.998
    r_BCD_A = 0.95
    Mn0, _ = Case_1(pD, r_BCD_A, MAA, MDD)
    Mn = Case_10(pB=0, pC=0., pD=pD, r_BC_DD=0, r_BCD_A=r_BCD_A,
                 MAA=MAA, MBC=MBC, MDD=MDD)
    assert isclose(Mn0, Mn)
    # AA + BC
    pBC = 0.998
    r_BCD_A = 0.93
    Mn0, _ = Case_1(pBC, r_BCD_A, MAA, MBC)
    Mn = Case_10(pB=pBC, pC=pBC, pD=0, r_BC_DD=1e10, r_BCD_A=r_BCD_A,
                 MAA=MAA, MBC=MBC, MDD=MDD)
    assert isclose(Mn0, Mn)
    # AA + BC + DD, with C=B
    pB = 0.99
    pD = 0.95
    r_BCD_A = 0.967
    r_BC_DD = 0.8
    Mn0 = Case_8(pB, pD, r_BCD_A, 1/r_BC_DD, MAA, MBC, MDD)
    Mn = Case_10(pB=pB, pC=pB, pD=pD, r_BC_DD=r_BC_DD, r_BCD_A=r_BCD_A,
                 MAA=MAA, MBC=MBC, MDD=MDD)
    assert isclose(Mn0, Mn)


def test_Case_11():
    MAA = 123.
    MBC = 83.
    MDD = 65.
    # AA + DD
    pD = 0.998
    r_DD_AA = 0.95
    Mn0, _ = Case_1(pD, r_DD_AA, MAA, MDD)
    Mn = Case_11(pB=0, pC=0., pD=pD, r_BC_AA=0, r_DD_AA=r_DD_AA,
                 MAA=MAA, MBC=MBC, MDD=MDD)
    assert isclose(Mn0, Mn)


def test_StockMayer():
    # AA + BB
    MAA = 123.
    MBB = 83.
    pB = 0.998
    r_BB_AA = 0.98
    A = [1.]
    f = [2]
    MA = [MAA]
    B = [r_BB_AA]
    g = [2]
    MB = [MBB]
    Mn0, Mw0 = Case_1(pB, r_BB_AA, MAA, MBB)
    Mn, Mw = Stockmayer(A, B, f, g, MA, MB, pB)
    assert isclose(Mn0, Mn)
    assert isclose(Mw0, Mw)
    # AA + B
    MAA = 83.
    MBC = 36.
    pB = 0.978
    r_BC_AA = 0.95
    A = [1.]
    f = [2]
    MA = [MAA]
    B = [r_BC_AA]
    g = [1]
    MB = [MBC]
    Mn0, Mw0 = Case_3(pB, 0, r_BC_AA, MAA, MBC)
    Mn, Mw = Stockmayer(A, B, f, g, MA, MB, pB)
    assert isclose(Mn0, Mn)
    assert isclose(Mw0, Mw)
