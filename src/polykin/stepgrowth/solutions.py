# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024


from typing import Union

import numpy as np
from numpy import dot, sqrt

from polykin.utils.types import (FloatArray, FloatArrayLike, FloatVectorLike,
                                 IntVectorLike)

__all__ = ['Case_1',
           'Case_3',
           'Case_5',
           'Case_6',
           'Case_7',
           'Case_8',
           'Case_9',
           'Case_10',
           'Case_11',
           'Flory_Af',
           'Stockmayer',
           'Miller_1',
           'Miller_2']


def Case_1(pB: Union[float, FloatArrayLike],
           r_BB_AA: float,
           MAA: float,
           MBB: float
           ) -> tuple[Union[float, FloatArray],
                      Union[float, FloatArray]]:
    r"""Case's analytical solution for AA reacting with BB (A₂ + B₂).

    The expressions for the number- and mass-average molar masses are:

    \begin{aligned}
    M_n &= \frac{\beta M_{AA} + \alpha M_{BB}}{\alpha + \beta -2\alpha\beta} \\
    M_w &= \frac{{1 + \alpha\beta}}{{1 - \alpha\beta}}
      \frac{{\beta M_{AA}^2 + \alpha M_{BB}^2}}{{\beta M_{AA} + \alpha M_{BB}}}
      + \frac{{4\alpha\beta M_{AA} M_{BB}}}
      {{(1 - \alpha\beta) (\beta M_{AA} + \alpha M_{BB})}}
    \end{aligned}

    where $\alpha$ and $\beta$ denote, respectively, the conversions of A and B
    groups.

    **References**

    *   Case, L.C. (1958), Molecular distributions in polycondensations
        involving unlike reactants. II. Linear distributions. J. Polym. Sci.,
        29: 455-495. https://doi.org/10.1002/pol.1958.1202912013

    Parameters
    ----------
    pB : float | FloatArrayLike
        Conversion of B groups.
    r_BB_AA : float
        Initial molar ratio of BB/AA molecules (or, equivalently, B/A groups).
        Unit = mol/mol.
    MAA : float
        Molar mass of a reacted AA unit.
    MBB : float
        Molar mass of a reacted BB unit.

    Returns
    -------
    tuple[float | FloatArray, float | FloatArray]
        Tuple of molar mass averages, ($M_n$, $M_w$).

    Examples
    --------
    Calculate Mn and Mw for a mixture with 1.01 mol of A₂ (100 g/mol) and
    1.0 mol of B₂ (90 g/mol) at 99% conversion of B groups.
    >>> from polykin.stepgrowth.solutions import Case_1
    >>> Mn, Mw = Case_1(0.99, 1.0/1.01, 100., 90.)
    >>> print(f"Mn={Mn:.0f}; Mw={Mw:.0f}")
    Mn=6367; Mw=12645

    """

    pB = np.asarray(pB)

    b = pB
    a = b*r_BB_AA

    Mn = (b*MAA + a*MBB)/(a + b - 2*a*b)

    Mw = (1 + a*b)/(1 - a*b)*(b*MAA**2 + a*MBB**2) / \
        (b*MAA + a*MBB) + (4*a*b*MAA*MBB)/((1 - a*b)*(b*MAA + a*MBB))

    return (Mn, Mw)


def Case_3(pB: Union[float, FloatArrayLike],
           pC: Union[float, FloatArrayLike],
           r_BC_AA: float,
           MAA: float,
           MBC: float
           ) -> tuple[Union[float, FloatArray],
                      Union[float, FloatArray]]:
    r"""Case's analytical solution for AA reacting with BC, where BC is
    an unsymmetric species (A₂ + BB').

    Alternative notation:

    \begin{matrix}
    AA&            +&   BC\\
    \updownarrow&   &   \updownarrow\\
    A_2&           +&   BB'
    \end{matrix}

    The expressions for the number- and mass-average molar masses are:

    \begin{aligned}
    M_n &= \frac{{(\beta + \gamma) M_{AA} + 2\alpha M_{BC}}}
                {{2\alpha + \beta + \gamma - 2\alpha(\beta + \gamma)}} \\
    M_w &= \frac{\left(1+\frac{2\alpha\beta\gamma}{\beta+\gamma}\right)M_{AA}^2
        + 4\alpha M_{AA} M_{BC}
        + \left(\frac{2\alpha}{(\beta + \gamma)^2}
        (\beta + \gamma + \alpha\beta^2 + \alpha\gamma^2)\right)M_{BC}^2}
        {\left(1 - \frac{2\alpha\beta\gamma}{\beta + \gamma}\right)
        \left(M_{AA} + \frac{2\alpha}{\beta + \gamma}M_{BC}\right)}
    \end{aligned}

    where $\alpha$, $\beta$ and $\gamma$ denote, respectively, the conversions
    of A, B and C groups.

    **References**

    *   Case, L.C. (1958), Molecular distributions in polycondensations
        involving unlike reactants. II. Linear distributions. J. Polym. Sci.,
        29: 455-495. https://doi.org/10.1002/pol.1958.1202912013

    Parameters
    ----------
    pB : float | FloatArrayLike
        Conversion of B groups.
    pC : float | FloatArrayLike
        Conversion of C groups.
    r_BC_AA : float
        Initial molar ratio of BC/AA molecules.
        Unit = mol/mol.
    MAA : float
        Molar mass of a reacted AA unit.
    MBC : float
        Molar mass of a reacted BC unit.

    Returns
    -------
    tuple[float | FloatArray, float | FloatArray]
        Tuple of molar mass averages, ($M_n$, $M_w$).

    Examples
    --------
    Calculate Mn and Mw for a mixture with 1.01 mol of A₂ (100 g/mol) and
    1.0 mol of BC (90 g/mol) at 99% conversion of B groups and 95% conversion
    of C groups.
    >>> from polykin.stepgrowth.solutions import Case_3
    >>> Mn, Mw = Case_3(0.99, 0.95, 1.0/1.01, 100., 90.)
    >>> print(f"Mn={Mn:.0f}; Mw={Mw:.0f}")
    Mn=2729; Mw=5332

    """

    pB = np.asarray(pB)
    pC = np.asarray(pC)

    b = pB
    c = pC
    a = r_BC_AA*(b + c)/2

    Mn = ((b + c)*MAA + 2*a*MBC)/(2*a + b + c - 2*a*(b + c))

    Mw = ((1 + 2*a*b*c/(b + c))*MAA**2 + 4*a*MAA*MBC +
          (2*a*(b + c + a*b**2 + a*c**2)/(b + c)**2)*MBC**2) \
        / ((1 - 2*a*b*c/(b + c))*(MAA + 2*a/(b + c)*MBC))

    return (Mn, Mw)


def Case_5(pB: Union[float, FloatArrayLike],
           pC: Union[float, FloatArrayLike],
           r_BC_A: float,
           r_C_B: float,
           MAA: float,
           MBB: float,
           MC: float
           ) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AA reacting with BB and C, where
    B does not react with C (A₂ + B₂ + B').

    Alternative notation:

    \begin{matrix}
    AA&  +&  BB  &  +& C\\
    \updownarrow&   & \updownarrow& & \updownarrow\\
    A_2& +&  B_2 &  +& B'
    \end{matrix}

    The expression for the number-average molar mass is:

    $$ M_n = \frac{M_{AA} + \frac{2\alpha}{2\beta + \nu\gamma} M_{BB}
            + \frac{2\alpha\nu}{2\beta + \nu\gamma} M_{C}}
            {1 - 2\alpha + \frac{2\alpha(1 + \nu)}{2\beta + \nu\gamma}} $$

    where $\nu$ represents the ratio of C to B groups, and $\alpha$, $\beta$
    and $\gamma$ denote, respectively, the conversions of A, B and C groups.

    **References**

    *   Case, L.C. (1958), Molecular distributions in polycondensations
        involving unlike reactants. II. Linear distributions. J. Polym. Sci.,
        29: 455-495. https://doi.org/10.1002/pol.1958.1202912013

    Parameters
    ----------
    pB : float | FloatArrayLike
        Conversion of B groups.
    pC : float | FloatArrayLike
        Conversion of C groups.
    r_BC_A : float
        Initial molar ratio of (B + C)/A groups.
        Unit = mol/mol.
    r_C_B : float
        Initial molar ratio of C/B groups.
        Unit = mol/mol.
    MAA : float
        Molar mass of a reacted AA unit.
    MBB : float
        Molar mass of a reacted BB unit.
    MC : float
        Molar mass of a reacted C unit.

    Returns
    -------
    float | FloatArray
        Number-average molar mass, $M_n$.

    Examples
    --------
    Calculate Mn for a mixture with 1.01 mol of A₂ (100 g/mol), 1.0 mol of B₂
    (90 g/mol) and 0.01 mol of C (40 g/mol) at 99% conversion of B groups and
    95% conversion of C groups.
    >>> from polykin.stepgrowth.solutions import Case_5
    >>> Mn = Case_5(0.99, 0.95, (2*1.0 + 0.01)/(2*1.01), 0.01/(2*1), 100., 90., 40.)
    >>> print(f"Mn={Mn:.0f}")
    Mn=6860

    """

    pB = np.asarray(pB)
    pC = np.asarray(pC)

    b = pB
    c = pC
    v = r_C_B
    a = r_BC_A*(b + v*c)/(1 + v)

    Mn = (MAA + 2*a/(2*b + v*c)*MBB + 2*a*v/(2*b + v*c)*MC) / \
         (1 - 2*a + 2*a*(1 + v)/(2*b + v*c))

    return Mn


def Case_6(pC: Union[float, FloatArrayLike],
           r_BC_AA: float,
           MAA: float,
           MBC: float
           ) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AA reacting with BC, where A and
    B react with C (A₂ + A'B).

    Alternative notation:

    \begin{matrix}
    AA&            +&   BC\\
    \updownarrow&   &   \updownarrow\\
    A_2&           +&   A'B
    \end{matrix}

    The expression for the number-average molar mass is:

    $$ M_n = \frac{M_{AA} + \nu M_{BC}}{1 + \nu(1 - \gamma)} $$

    where $\nu$ represents the ratio of BC to AA units, and $\gamma$ denotes
    the conversion of C groups.

    !!! note

        For this case, a formula for $M_w$ was also reported, but it provides
        inconsistent results in the limiting case of BC-only systems.

    **References**

    *   Case, L.C. (1958), Molecular distributions in polycondensations
        involving unlike reactants. II. Linear distributions. J. Polym. Sci.,
        29: 455-495. https://doi.org/10.1002/pol.1958.1202912013

    Parameters
    ----------
    pC : float | FloatArrayLike
        Conversion of C groups.
    r_BC_AA : float
        Initial molar ratio of BC/AA molecules.
        Unit = mol/mol.
    MAA : float
        Molar mass of a reacted AA unit.
    MBC : float
        Molar mass of a reacted BC unit.

    Returns
    -------
    float | FloatArray
        Number-average molar mass, $M_n$.

    Examples
    --------
    Calculate Mn for a mixture with 0.01 mol of AA (100 g/mol) and 2.0 mol of
    A'B (80 g/mol) at 99% conversion of B groups.
    >>> from polykin.stepgrowth.solutions import Case_6
    >>> Mn = Case_6(0.99, 2.0/0.01, 100., 80.)
    >>> print(f"Mn={Mn:.0f}")
    Mn=5367

    """

    pC = np.asarray(pC)

    # a = pA
    # b = pB
    # c = 2*a/v + b
    c = pC
    v = r_BC_AA

    Mn = (MAA + v*MBC)/(1 + v*(1 - c))

    # Mw formula is inconsistent for v->inf
    # Mw = (MAA**2 + 4*a*MAA*MBC/(1 - b) +
    #       (2*a*(a + 2*b)/(1 - b)**2 + v)*MBC**2)/(MAA + v*MBC)

    return Mn


def Case_7(pA: Union[float, FloatArrayLike],
           pC: Union[float, FloatArrayLike],
           r_CD_AB: float,
           MAB: float,
           MCD: float
           ) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AB reacting with CD (AB + A'B').

    Alternative notation:

    \begin{matrix}
    AB&            +&   CD\\
    \updownarrow&   &   \updownarrow\\
    AB&            +&   A'B'
    \end{matrix}

    The expression for the number-average molar mass is:

    $$ M_n = \frac{M_{AB} + \nu M_{CD}}{{1 + \nu - (\alpha + \nu\gamma)}} $$

    where $\nu$ represents the ratio of CD to AB units, while $\alpha$ and
    $\gamma$ denote, respectively, the conversions of A and C groups.

    **References**

    *   Case, L.C. (1958), Molecular distributions in polycondensations
        involving unlike reactants. II. Linear distributions. J. Polym. Sci.,
        29: 455-495. https://doi.org/10.1002/pol.1958.1202912013

    Parameters
    ----------
    pA : float | FloatArrayLike
        Conversion of A groups.
    pC : float | FloatArrayLike
        Conversion of C groups.
    r_CD_AB : float
        Initial molar ratio of CD/AB molecules.
        Unit = mol/mol.
    MAB : float
        Molar mass of a reacted AB unit.
    MCD : float
        Molar mass of a reacted CD unit.

    Returns
    -------
    float | FloatArray
        Number-average molar mass, $M_n$.

    Examples
    --------
    Calculate Mn for a mixture with 1.0 mol of AB (100 g/mol) and 2.0 mol of
    A'B' (80 g/mol) at 99% conversion of A groups and 98% conversion of A'
    groups.
    >>> from polykin.stepgrowth.solutions import Case_7
    >>> Mn = Case_7(0.99, 0.98, 2.0/1.0, 100., 80.)
    >>> print(f"Mn={Mn:.0f}")
    Mn=5200

    """

    pA = np.asarray(pA)
    pC = np.asarray(pC)

    a = pA
    c = pC
    v = r_CD_AB

    Mn = (MAB + v*MCD)/(1 + v - (a + v*c))

    return Mn


def Case_8(pB: Union[float, FloatArrayLike],
           pC: Union[float, FloatArrayLike],
           r_BC_A: float,
           r_CC_BB: float,
           MAA: float,
           MBB: float,
           MCC: float
           ) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AA reacting with BB and CC, where B does
    not react with C (A₂ + B₂ + B₂').

    Alternative notation:

    \begin{matrix}
    AA&   +&  BB  &  +& CC\\
    \updownarrow&   & \updownarrow& & \updownarrow\\
    A_2&  +&  B_2 &  +& B_2'
    \end{matrix}

    The expression for the number-average molar mass is:

    $$ M_n = \frac{M_{AA} + \frac{\alpha (M_{BB} + \nu M_{CC})}
            {\beta + \nu\gamma}}
            {1 - 2\alpha + \frac{\alpha (1 + \nu)}{\beta + \nu\gamma}} $$

    where $\nu$ represents the ratio of CC to BB units, while $\alpha$, $\beta$
    and $\gamma$ denote, respectively, the conversions of A, B and C groups.

    **References**

    *   Case, L.C. (1958), Molecular distributions in polycondensations
        involving unlike reactants. II. Linear distributions. J. Polym. Sci.,
        29: 455-495. https://doi.org/10.1002/pol.1958.1202912013

    Parameters
    ----------
    pB : float | FloatArrayLike
        Conversion of B groups.
    pC : float | FloatArrayLike
        Conversion of C groups.
    r_BC_A : float
        Initial molar ratio of (B + C)/A groups.
        Unit = mol/mol.
    r_CC_BB : float
        Initial molar ratio of CC/BB molecules (or, equivalently, C/B groups).
        Unit = mol/mol.
    MAA : float
        Molar mass of a reacted AA unit.
    MBB : float
        Molar mass of a reacted BB unit.
    MCC : float
        Molar mass of a reacted CC unit.

    Returns
    -------
    float | FloatArray
        Number-average molar mass, $M_n$.

    Examples
    --------
    Calculate Mn for a mixture with 1.01 mol of A₂ (100 g/mol), 0.6 mol of B₂
    (90 g/mol) and 0.4 mol of B₂' (50 g/mol) at 99% conversion of B groups and
    95% conversion of B' groups.
    >>> from polykin.stepgrowth.solutions import Case_8
    >>> Mn = Case_8(0.99, 0.95, (0.6 + 0.4)/1.01, 0.4/0.6, 100., 90., 50.)
    >>> print(f"Mn={Mn:.0f}")
    Mn=2823

    """

    pB = np.asarray(pB)
    pC = np.asarray(pC)

    b = pB
    c = pC
    v = r_CC_BB
    a = r_BC_A*(b + v*c)/(1 + v)

    Mn = (MAA + a*(MBB + v*MCC)/(b + v*c))/(1 - 2*a + a*(1 + v)/(b + v*c))

    return Mn


def Case_9(pB: Union[float, FloatArrayLike],
           pC: Union[float, FloatArrayLike],
           pD: Union[float, FloatArrayLike],
           r_CD_AB: float,
           r_BB_AA: float,
           r_DD_CC: float,
           MAA: float,
           MBB: float,
           MCC: float,
           MDD: float
           ) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AA and BB reacting with CC and DD, where
    A and B react only with C and D (A₂ + A₂' + B₂ + B₂').

    Alternative notation:

    \begin{matrix}
    AA &   + &  BB &  + &  CC  & + & DD \\
    \updownarrow&   & \updownarrow& & \updownarrow& & \updownarrow\\
    A_2 &  + &  A_2' &  + & B_2 & + & B_2'
    \end{matrix}

    The expression for the number-average molar mass is:

    $$ M_n = \frac{M_{AA} + \nu M_{BB} + \frac{\alpha + \nu \beta}
        {\gamma + \rho \delta} (M_{CC} + \rho M_{DD})}
        {1 - \alpha + \nu (1 - \beta) + \frac{\alpha + \nu \beta}
        {\gamma + \rho \delta} \left(1 - \gamma + \rho (1 - \delta)\right)} $$

    where $\nu$ is the ratio of B to A groups, $\rho$ is the ratio of D to C
    groups, and $\alpha$, $\beta$, $\gamma$ and $\delta$ denote, respectively,
    the conversions of A, B, C and D groups.

    **References**

    *   Case, L.C. (1958), Molecular distributions in polycondensations
        involving unlike reactants. II. Linear distributions. J. Polym. Sci.,
        29: 455-495. https://doi.org/10.1002/pol.1958.1202912013

    Parameters
    ----------
    pB : float | FloatArrayLike
        Conversion of B groups.
    pC : float | FloatArrayLike
        Conversion of C groups.
    pD : float | FloatArrayLike
        Conversion of D groups.
    r_CD_AB : float
        Initial molar ratio of (C+D)/(A+B) groups.
        Unit = mol/mol.
    r_BB_AA : float
        Initial molar ratio of BB/AA molecules (or, equivalently, B/A groups).
        Unit = mol/mol.
    r_DD_CC : float
        Initial molar ratio of DD/CC molecules (or, equivalently, D/C groups).
        Unit = mol/mol.
    MAA : float
        Molar mass of a reacted AA unit.
    MBB : float
        Molar mass of a reacted BB unit.
    MCC : float
        Molar mass of a reacted CC unit.
    MDD : float
        Molar mass of a reacted DD unit.

    Returns
    -------
    float | FloatArray
        Number-average molar mass, $M_n$.

    Examples
    --------
    Calculate Mn for a mixture with 1.0 mol of A₂ (100 g/mol), 0.5 mol of A₂'
    (80 g/mol), 0.6 mol of B₂ (90 g/mol) and 0.9 mol of B'₂ (70 g/mol) at 99%
    conversion of A' groups, 98% conversion of B groups, and 99.9% conversion
    of B' groups.
    >>> from polykin.stepgrowth.solutions import Case_9
    >>> Mn = Case_9(0.99, 0.98, 0.999, (0.6 + 0.9)/(1.0 + 0.5), 0.5/1.0,
    ...             0.9/0.6, 100., 80., 90., 70.)
    >>> print(f"Mn={Mn:.0f}")
    Mn=9961

    """

    pB = np.asarray(pB)
    pC = np.asarray(pC)
    pD = np.asarray(pD)

    b = pB
    c = pC
    d = pD
    v = r_BB_AA
    r = r_DD_CC
    a = r_CD_AB*(1 + v)/(1 + r)*(c + r*d) - v*b

    Mn = (MAA + v*MBB + (a + v*b)/(c + r*d)*(MCC + r*MDD)) / \
        (1 - a + v*(1 - b) + (a + v*b)/(c + r*d)*(1 - c + r*(1 - d)))

    return Mn


def Case_10(pB: Union[float, FloatArrayLike],
            pC: Union[float, FloatArrayLike],
            pD: Union[float, FloatArrayLike],
            r_BCD_A: float,
            r_BC_DD: float,
            MAA: float,
            MBC: float,
            MDD: float
            ) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AA reacting with BC and DD, where A
    reacts only with B, C or D (A₂ + BB' + B₂'').

    Alternative notation:

    \begin{matrix}
    AA&  +&  BC &  +& DD\\
    \updownarrow&   & \updownarrow& & \updownarrow\\
    A_2&  +&  BB' &  +& B_2''
    \end{matrix}

    The expression for the number-average molar mass is:

    $$ M_n = \frac{M_{AA} + \frac{\alpha( M_{DD} + 2 \nu M_{BC})}
                                 {\nu (\beta + \gamma) + \delta}}
        {1 - \alpha + \frac{\alpha}{\nu (\beta + \gamma) + \delta}
              \left(1 - \delta + \nu (2 - \beta - \gamma) \right)} $$

    where $2\nu$ represents the ratio of BC to DD units, while $\alpha$,
    $\beta$, $\gamma$ and $\delta$ denote, respectively, the conversions of
    A, B, C and D groups.

    **References**

    *   Case, L.C. (1958), Molecular distributions in polycondensations
        involving unlike reactants. II. Linear distributions. J. Polym. Sci.,
        29: 455-495. https://doi.org/10.1002/pol.1958.1202912013

    Parameters
    ----------
    pB : float | FloatArrayLike
        Conversion of B groups.
    pC : float | FloatArrayLike
        Conversion of C groups.
    pD : float | FloatArrayLike
        Conversion of D groups.
    r_BCD_A : float
        Initial molar ratio of (B + C + D)/A groups.
        Unit = mol/mol.
    r_BC_DD : float
        Initial molar ratio of BC/DD molecules.
        Unit = mol/mol.
    MAA : float
        Molar mass of a reacted AA unit.
    MBC : float
        Molar mass of a reacted BC unit.
    MDD : float
        Molar mass of a reacted DD unit.

    Returns
    -------
    float | FloatArray
        Number-average molar mass, $M_n$.

    Examples
    --------
    Calculate Mn for a mixture with 1.01 mol of A₂ (100 g/mol), 0.6 mol of BB'
    (90 g/mol) and 0.4 mol of B₂'' (50 g/mol) at 99% conversion of B groups,
    98% conversion of B' groups and 99.9% conversion of B'' groups.
    >>> from polykin.stepgrowth.solutions import Case_10
    >>> Mn = Case_10(0.99, 0.98, 0.999, (0.6 + 0.4)/1.01, 0.6/0.4, 100., 90., 50.)
    >>> print(f"Mn={Mn:.0f}")
    Mn=6076

    """

    pB = np.asarray(pB)
    pC = np.asarray(pC)
    pD = np.asarray(pD)

    b = pB
    c = pC
    d = pD
    v = r_BC_DD/2
    a = r_BCD_A*(v*(b + c) + d)/(2*v + 1)

    Mn = (MAA + a*(MDD + 2*v*MBC)/(v*(b + c) + d)) / \
        (1 - a + a/(v*(b + c) + d)*((1 - d) + v*(2 - b - c)))

    return Mn


def Case_11(pB: Union[float, FloatArrayLike],
            pC: Union[float, FloatArrayLike],
            pD: Union[float, FloatArrayLike],
            r_BC_AA: float,
            r_DD_AA: float,
            MAA: float,
            MBC: float,
            MDD: float
            ) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AA and DD reacting with BC, where A
    and B react only with C and D (A₂ + A'B' + B₂).

    Alternative notation:

    \begin{matrix}
    AA&  +&  BC &  +& DD\\
    \updownarrow&   & \updownarrow& & \updownarrow\\
    A_2&  +&  A'B' &  +& B_2
    \end{matrix}

    The expression for the number-average molar mass is:

    $$ M_n =
    \frac{M_{AA}+2\nu M_{BC}+\frac{\alpha+\nu (\beta-\gamma)}{\delta} M_{DD}}
    {1-2\alpha+2\nu (1-\beta) + \frac{\alpha+\nu (\beta-\gamma)}{\delta}} $$

    where $2\nu$ represents the ratio of BC to AA units, while $\alpha$,
    $\beta$, $\gamma$ and $\delta>0$ denote, respectively, the conversions of
    A, B, C and D groups.

    **References**

    *   Case, L.C. (1958), Molecular distributions in polycondensations
        involving unlike reactants. II. Linear distributions. J. Polym. Sci.,
        29: 455-495. https://doi.org/10.1002/pol.1958.1202912013

    Parameters
    ----------
    pB : float | FloatArrayLike
        Conversion of B groups.
    pC : float | FloatArrayLike
        Conversion of C groups.
    pD : float | FloatArrayLike
        Conversion of D groups.
    r_BC_AA : float
        Initial molar ratio of BC/AA molecules.
        Unit = mol/mol.
    r_DD_AA : float
        Initial molar ratio of DD/AA molecules.
        Unit = mol/mol.
    MAA : float
        Molar mass of a reacted AA unit.
    MBC : float
        Molar mass of a reacted BC unit.
    MDD : float
        Molar mass of a reacted DD unit.

    Returns
    -------
    float | FloatArray
        Number-average molar mass, $M_n$.

    Examples
    --------
    Calculate Mn for a mixture with 1.01 mol of A₂ (100 g/mol), 1.5 mol of A'B'
    (90 g/mol) and 1.0 mol of B₂ (50 g/mol) at 99% conversion of A' groups,
    98% conversion of B groups and 99.9% conversion of B' groups.
    >>> from polykin.stepgrowth.solutions import Case_11
    >>> Mn = Case_11(0.99, 0.999, 0.98, 1.5/1.01, 1.0/1.01, 100., 90., 50.)
    >>> print(f"Mn={Mn:.0f}")
    Mn=5553

    """

    pB = np.asarray(pB)
    pC = np.asarray(pC)
    pD = np.asarray(pD)

    b = pB
    c = pC
    d = pD
    v = r_BC_AA/2
    a = v*(c - b) + r_DD_AA*d

    Mn = (MAA + 2*v*MBC + (a + v*(b - c))/d*MDD) / \
        (1 - 2*a + 2*v*(1 - b) + (a + v*(b - c))/d)

    return Mn


def Stockmayer(nA: FloatVectorLike,
               nB: FloatVectorLike,
               f: IntVectorLike,
               g: IntVectorLike,
               MA: FloatVectorLike,
               MB: FloatVectorLike,
               pB: Union[float, FloatArrayLike]
               ) -> tuple[Union[float, FloatArray],
                          Union[float, FloatArray]]:
    r"""Stockmayer's analytical solution for an arbitrary mixture of A-type
    monomers reacting with an arbitry mixture of B-type monomers.

    The expressions for the number- and mass-average molar masses are:

    \begin{aligned}
    M_n &= \frac{\sum_i M_{A,i}A_i + \sum_j M_{B,j}B_j}{\sum_i A_i + \sum_j B_j -p_A\sum_i f_iA_i} \\
    M_w &= \frac{p_B\frac{\sum_i M_{A,i}^2A_i}{\sum_i f_i A_i}+p_A\frac{\sum_j M_{B,j}^2B_j}
    {\sum_j g_j B_j}+\frac{p_Ap_B[p_A(f_e-1)M_b^2+p_B(g_e-1)M_a^2+2M_aM_b)]}{1-p_A p_B(f_e-1)(g_e-1)}}
    {p_B\frac{\sum_i M_{A,i}A_i}{\sum_i f_i A_i}+p_A\frac{\sum_j M_{B,j}B_j}{\sum_j g_j B_j}}
    \end{aligned}

    with:

    \begin{aligned}
    f_e &= \frac{\sum_i f_i^2 A_i}{\sum_i f_i A_i}  \\
    g_e &= \frac{\sum_j g_j^2 B_j}{\sum_j g_j B_j} \\
    M_a &= \frac{\sum_i M_{A,i}f_i A_i}{\sum_i f_i A_i}  \\
    M_b &= \frac{\sum_j M_{B,j}g_j B_j}{\sum_j g_j B_j}
    \end{aligned}

    where $p_A$ and $p_B$ denote, respectively, the conversion of A and B
    groups, which must respect the limit imposed by the gelation condition
    $p_A p_B < [(f_e-1)(g_e - 1)]^{-1}$.

    **References**

    *   Stockmayer, W.H. (1952), Molecular distribution in condensation
        polymers. J. Polym. Sci., 9: 69-71. https://doi.org/10.1002/pol.1952.120090106

    Parameters
    ----------
    nA : FloatVectorLike
        Vector (N) with relative mole amounts of A monomers.
        Unit = mol or mol/mol.
    nB : FloatVectorLike
        Vector (M) with relative mole amounts of B monomers.
        Unit = mol or mol/mol.
    f : IntVectorLike
        Vector (N) with functionality of A monomers.
    g : IntVectorLike
        Vector (M) with functionality of B monomers.
    MA : FloatVectorLike
        Vector (N) with molar mass of reacted A monomers.
    MB : FloatVectorLike
        Vector (M) with molar mass of reacted B monomers.
    pB : float | FloatArrayLike
        Overall conversion of B groups.

    Returns
    -------
    tuple[float | FloatArray, float | FloatArray]
        Tuple of molar mass averages, ($M_n$, $M_w$).

    Examples
    --------
    Calculate Mn and Mw for a mixture with 1 mol of A₂ (100 g/mol), 0.01 mol of
    A₃ (150 g/mol), 1.01 mol of B₂ (80 g/mol) and 0.03 mol of B (40 g/mol) at
    99% conversion of B groups.
    >>> from polykin.stepgrowth.solutions import Stockmayer
    >>> Mn, Mw = Stockmayer(nA=[1., 0.01], nB=[1.01, 0.03], f=[2, 3], g=[2, 1],
    ...                     MA=[100., 150.], MB=[80., 40.], pB=0.99)
    >>> print(f"Mn={Mn:.0f}; Mw={Mw:.0f}")
    Mn=8951; Mw=34722

    """

    nA = np.asarray(nA)
    nB = np.asarray(nB)
    f = np.asarray(f)
    g = np.asarray(g)
    MA = np.asarray(MA)
    MB = np.asarray(MB)
    pB = np.asarray(pB)

    sMA = dot(MA, nA)
    sM2A = dot(MA**2, nA)
    sfA = dot(f, nA)
    sf2A = dot(f**2, nA)
    sMB = dot(MB, nB)
    sM2B = dot(MB**2, nB)
    sgB = dot(g, nB)
    sg2B = dot(g**2, nB)

    fe = sf2A/sfA
    ge = sg2B/sgB
    Ma = np.sum(MA*f*nA)/sfA
    Mb = np.sum(MB*g*nB)/sgB

    pA = pB*sgB/sfA

    pB_gel = sqrt(sfA/(sgB*(fe - 1)*(ge - 1)))
    pB = np.where(pB < pB_gel, pB, np.nan)

    Mn = (sMA + sMB)/(nA.sum() + nB.sum() - pA*sfA)

    Mw = (pB*(sM2A/sfA) + pA*(sM2B/sgB) + pA*pB *
          (pA*(fe - 1)*Mb**2 + pB*(ge - 1)*Ma**2 + 2*Ma*Mb) /
          (1 - pA*pB*(fe - 1)*(ge - 1)))/(pB*(sMA/sfA) + pA*(sMB/sgB))

    return (Mn, Mw)


def Flory_Af(f: int,
             MAf: float,
             p: Union[float, FloatArrayLike]
             ) -> tuple[Union[float, FloatArray],
                        Union[float, FloatArray],
                        Union[float, FloatArray]]:
    r"""Flory's analytical solution for the hopolymerization of an $A_f$
    monomer.

    The expressions for the number-, mass-, and z-average molar masses are:

    \begin{aligned}
    M_n &= M_{A_f}\frac{1}{1 - (f/2)p} \\
    M_w &= M_{A_f}\frac{1 + p}{1 - (f-1)p} \\
    M_z &= M_{A_f}\frac{(1 + p)^3 - f(3 + p)p^2}{(1 + p)[1 - (f-1)p]^2}
    \end{aligned}

    where $p$ denotes the conversion of A groups, which must respect the limit
    imposed by the gelation condition $p<p_{gel}=(f-1)^{-1}$.

    **References**

    *   P. J. Flory, Molecular Size Distribution in Three Dimensional
        Polymers. I. Gelation, J. Am. Chem. Soc. 1941, 63, 11, 3083.
    *   M. Gordon, Proc. R. Soc. London, Ser. A, 1962, 268, 240.

    Parameters
    ----------
    f : int
        Functionality.
    MAf : float
        Molar mass of reacted Af monomer.
    p : float | FloatArrayLike
        Conversion of A groups. 

    Returns
    -------
    tuple[float | FloatArray, ...]
        Tuple of molar mass averages, ($M_n$, $M_w$, $M_z$).

    Examples
    --------
    Calculate Mn, Mw and Mz for A₄ (100 g/mol) at 30% conversion of A groups.
    >>> from polykin.stepgrowth.solutions import Flory_Af
    >>> Mn, Mw, Mz = Flory_Af(4, 100., 0.30)
    >>> print(f"Mn={Mn:.0f}; Mw={Mw:.0f}; Mz={Mz:.0f}")
    Mn=250; Mw=1300; Mz=7762

    """

    p = np.asarray(p)

    p_gel = 1/(f - 1)
    p = np.where(p < p_gel, p, np.nan)

    Mn = MAf/(1 - (f/2)*p)
    Mw = MAf*(1 + p)/(1 - p*(f - 1))
    Mz = MAf*((1 + p)**3 - (3 + p)*f*p**2)/((1 + p)*(1 - p*(f - 1))**2)

    return (Mn, Mw, Mz)


def Miller_1(nAf: float,
             nA2: float,
             nB2: float,
             f: int,
             MAf: float,
             MA2: float,
             MB2: float,
             pB1: Union[float, FloatArrayLike],
             pB2: Union[float, FloatArrayLike]
             ) -> tuple[Union[float, FloatArray],
                        Union[float, FloatArray]]:
    r"""Miller and Macosko's analytical solution for Af and A₂ reacting with
    B₂, where the two B groups comprising the B₂ monomer react at different
    rates.

    The expressions for the number- and mass-average molar masses are:

    \begin{aligned}
    M_n &= \frac{M_{A_f}A_f + M_{A_2}A_2 + M_{B_2}B_2}{A_f + A_2 + B_2(1 - p_{B^1} - p_{B^2})} \\
    M_w &= \frac{M_{A_f}A_fE(W_{A_f}) + M_{A_2}A_2E(W_{A_2}) + M_{B_2}B_2E(W_{B_2})}
                {M_{A_f}A_f + M_{A_2}A_2 + M_{B_2}B_2}
    \end{aligned}

    where $E(W_X)$ is the expected weight attached to a unit of type $X$.

    **References**

    *   D. R. Miller and C. W. Macosko, Average Property Relations for
        Nonlinear Polymerization with Unequal Reactivity, Macromolecules 1978,
        11, 4, 656-662. https://doi.org/10.1021/ma60064a008

    Parameters
    ----------
    nAf : float
        Relative mole amount of Af monomer.
        Unit = mol or mol/mol.
    nA2 : float
        Relative mole amount of A₂ monomer.
        Unit = mol or mol/mol.
    nB2 : float
        Relative mole amount of B¹B² monomer.
        Unit = mol or mol/mol.
    f : int
        Functionality of Af.
    MAf : float
        Molar mass of reacted Af monomer.
    MA2 : float
        Molar mass of reacted A₂ monomer.
    MB2 : float
        Molar mass of reacted B¹B² monomer.
    pB1 : float | FloatArrayLike
        Conversion of B¹ groups.
    pB2 : float | FloatArrayLike
        Conversion of B² groups.

    Returns
    -------
    tuple[float | FloatArray, float | FloatArray]
        Tuple of molar mass averages, ($M_n$, $M_w$).

    Examples
    --------
    Calculate Mn and Mw for a mixture with 0.1 mol of A₃ (150 g/mol), 0.9 mol
    of A₂ (100 g/mol), and 1 mol of B₂ (80 g/mol) at 95% conversion of B¹
    groups and 90% conversion of B² groups.
    >>> from polykin.stepgrowth.solutions import Miller_1
    >>> Mn, Mw = Miller_1(0.1, 0.9, 1.0, 3, 150., 100., 80., 0.95, 0.90)
    >>> print(f"Mn={Mn:.0f}; Mw={Mw:.0f}")
    Mn=1233; Mw=5024

    """

    pB1 = np.asarray(pB1)
    pB2 = np.asarray(pB2)

    r = (f*nAf + 2*nA2)/(2*nB2)
    pA = 0.5*(pB1 + pB2)/r
    af = f*nAf/(f*nAf + 2*nA2)

    EAout = (pA*MB2 + pB1*pB2/r*((1 - af)*MA2 + af*MAf)) / \
        (1 - pB1*pB2/r*(1 + (f - 2)*af))
    EBout = (1 - af)*MA2 + af*MAf + (1 + (f - 2)*af)*EAout
    EB1out = pB1*EBout
    EB2out = pB2*EBout

    EAf = MAf + f*EAout
    EA2 = MA2 + 2*EAout
    EBB = MB2 + EB1out + EB2out

    Mn = (MAf*nAf + MA2*nA2 + MB2*nB2)/(nAf + nA2 + nB2 - nB2*(pB1 + pB2))
    Mw = (MAf*nAf*EAf + MA2*nA2*EA2 + MB2*nB2*EBB) / \
        (MAf*nAf + MA2*nA2 + MB2*nB2)

    return (Mn, Mw)


def Miller_2(nAf: float,
             nB2: float,
             f: int,
             MAf: float,
             MB2: float,
             p: FloatVectorLike,
             ) -> tuple[float, float]:
    r"""Miller and Macosko's analytical solution for Af reacting with B₂,
    with first-shell substitution effect on the A's and independent reaction
    for B's.

    The expressions for the number- and mass-average molar masses are:

    \begin{aligned}
    M_n &= \frac{M_{A_f}A_f + M_{B_2}B_2}{A_f(1 - f p_A) + B_2} \\
    M_w &= \frac{ 2\frac{r}{f}\left( 1 + r p_A \left( f p_A - \mu + 1 \right) \right) M_{A_f}^2 
    + \left( 1 + r p_A \left( \mu - 1 \right) \right) M_{B_2}^2 + 4 r p_A M_{A_f} M_{B_2} }
    {\left( 2\frac{r}{f} M_{A_f} + M_{B_2} \right) \left( 1 - r p_A \left(\mu - 1 \right) \right)}
    \end{aligned}

    with:

    $$ \mu = \frac{\sum_{i=1}^f i^2 p_i}{\sum_{i=1}^f i p_i} $$

    where $r$ is the molar ratio of A to B groups, $p_i$ is the fraction of
    $A_f$ units wich have exactly $i$ reacted sites, and 
    $p_A=\sum_{i=1}^f (i/f)p_i$ is the total conversion of A groups. The latter
    must respect the limit imposed by the gelation condition
    $p_A < [r(\mu-1)]^{-1}$.

    **References**

    *   D. R. Miller and C. W. Macosko, Substitution Effects in Property
        Relations for Stepwise Polyfunctional Polymerization, Macromolecules
        1980 13 (5), 1063-1069. https://doi.org/10.1021/ma60077a008

    Parameters
    ----------
    nAf : float
        Relative mole amount of Af monomer.
        Unit = mol or mol/mol.
    nB2 : float
        Relative mole amount of B₂ monomer.
        Unit = mol or mol/mol.
    f : int
        Functionality of Af.
    MAf : float
        Molar mass of reacted Af monomer.
    MB2 : float
        Molar mass of reacted B₂ monomer.
    p : FloatVectorLike
        Vector of reaction extents $(p_1, ..., p_f)$, where $p_i$ denotes
        the fraction of $A_f$ units wich have exactly $i$ reacted sites.

    Returns
    -------
    tuple[float | FloatArray, float | FloatArray]
        Tuple of molar mass averages, ($M_n$, $M_w$).

    Examples
    --------
    Calculate Mn and Mw for a mixture with 1 mol of A₃ (100 g/mol) and 1.5 mol
    of B₂ (80 g/mol) at a conversion of A₃ corresponding to p=[0.1, 0.2, 0.4].
    >>> from polykin.stepgrowth.solutions import Miller_2
    >>> Mn, Mw = Miller_2(1., 1.5, 3, 100., 80., [0.1, 0.2, 0.4])
    >>> print(f"Mn={Mn:.0f}; Mw={Mw:.0f}")
    Mn=275; Mw=3822

    """

    p = np.asarray(p)

    i = np.arange(1, f + 1, dtype=np.int32)
    sip = dot(i, p)
    m = dot(i**2, p) / sip

    pA = sip/f
    r = (f*nAf)/(2*nB2)

    if r*pA*(m - 1) < 1.:
        Mn = (MAf*nAf + MB2*nB2)/(nAf + nB2 - f*nAf*pA)

        Mw = ((2*r/f)*(1 + r*pA*(f*pA - m + 1))*MAf**2 + (1 + r*pA*(m - 1))
              * MB2**2 + 4*r*pA*MAf*MB2)/(((2*r/f)*MAf + MB2)*(1 - r*pA*(m - 1)))
    else:
        Mn = np.nan
        Mw = np.nan

    return (Mn, Mw)
