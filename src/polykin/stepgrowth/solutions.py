# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Union

from polykin.utils.types import FloatArray

__all__ = ['Case_1',
           'Case_3',
           'Case_5',
           'Case_6',
           'Case_7',
           'Case_8',
           'Case_9',
           'Case_10',
           'Case_11']


def Case_1(pB: Union[float, FloatArray],
           r_BB_AA: float,
           MAA: float,
           MBB: float
           ) -> tuple[Union[float, FloatArray],
                      Union[float, FloatArray]]:
    r"""Case's analytical solution for AA reacting with BB.

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
    pB : float | FloatArray
        Conversion of B groups.
    r_BB_AA : float
        Initial molar ratio of BB/AA molecules (or, equivalently, B/A groups).
    MAA : float
        Molar mass of a reacted AA unit.
    MBB : float
        Molar mass of a reacted BB unit.

    Returns
    -------
    tuple[float | FloatArray, float | FloatArray]
        Tuple of molar mass averages, ($M_n$, $M_w$).
    """

    b = pB
    a = b*r_BB_AA

    Mn = (b*MAA + a*MBB)/(a + b - 2*a*b)

    Mw = (1 + a*b)/(1 - a*b)*(b*MAA**2 + a*MBB**2) / \
        (b*MAA + a*MBB) + (4*a*b*MAA*MBB)/((1 - a*b)*(b*MAA + a*MBB))

    return (Mn, Mw)


def Case_3(pB: Union[float, FloatArray],
           pC: Union[float, FloatArray],
           r_BC_AA: float,
           MAA: float,
           MBC: float
           ) -> tuple[Union[float, FloatArray],
                      Union[float, FloatArray]]:
    r"""Case's analytical solution for AA reacting with BC, where BC is
    an unsymmetric species (AA + BB').

    Alternative notation:

    \begin{matrix}
    AA&            +&   BC\\
    \updownarrow&   &   \updownarrow\\
    AA&            +&   BB'
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
    pB : float | FloatArray
        Conversion of B groups.
    pC : float | FloatArray
        Conversion of C groups.
    r_BC_AA : float
        Initial molar ratio of BC/AA molecules.
    MAA : float
        Molar mass of a reacted AA unit.
    MBC : float
        Molar mass of a reacted BC unit.

    Returns
    -------
    tuple[float | FloatArray, float | FloatArray]
        Tuple of molar mass averages, ($M_n$, $M_w$).
    """

    b = pB
    c = pC
    a = r_BC_AA*(b + c)/2

    Mn = ((b + c)*MAA + 2*a*MBC)/(2*a + b + c - 2*a*(b + c))

    Mw = ((1 + 2*a*b*c/(b + c))*MAA**2 + 4*a*MAA*MBC +
          (2*a*(b + c + a*b**2 + a*c**2)/(b + c)**2)*MBC**2) \
        / ((1 - 2*a*b*c/(b + c))*(MAA + 2*a/(b + c)*MBC))

    return (Mn, Mw)


def Case_5(pB: Union[float, FloatArray],
           pC: Union[float, FloatArray],
           r_BC_A: float,
           r_C_B: float,
           MAA: float,
           MBB: float,
           MC: float
           ) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AA reacting with BB and C, where
    B does not react with C (AA + BB + B').

    Alternative notation:

    \begin{matrix}
    AA&  +&  BB &  +& C\\
    \updownarrow&   & \updownarrow& & \updownarrow\\
    AA&  +&  BB &  +& B'
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
    pB : float | FloatArray
        Conversion of B groups.
    pC : float | FloatArray
        Conversion of C groups.
    r_BC_A : float
        Initial molar ratio of (B + C)/A groups.
    r_C_B : float
        Initial molar ratio of C/B groups.
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
    """

    b = pB
    c = pC
    v = r_C_B
    a = r_BC_A*(b + v*c)/(1 + v)

    Mn = (MAA + 2*a/(2*b + v*c)*MBB + 2*a*v/(2*b + v*c)*MC) / \
         (1 - 2*a + 2*a*(1 + v)/(2*b + v*c))

    return Mn


def Case_6(pC: Union[float, FloatArray],
           r_BC_AA: float,
           MAA: float,
           MBC: float
           ) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AA reacting with BC, where A and
    B react with C (AA + A'B).

    Alternative notation:

    \begin{matrix}
    AA&            +&   BC\\
    \updownarrow&   &   \updownarrow\\
    AA&            +&   A'B
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
    pC : float | FloatArray
        Conversion of C groups.
    r_BC_AA : float
        Initial molar ratio of BC/AA molecules.
    MAA : float
        Molar mass of a reacted AA unit.
    MBC : float
        Molar mass of a reacted BC unit.

    Returns
    -------
    float | FloatArray
        Number-average molar mass, $M_n$.
    """

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


def Case_7(pA: Union[float, FloatArray],
           pC: Union[float, FloatArray],
           r_CD_AB: float,
           MAB: float,
           MCD: float) -> Union[float, FloatArray]:
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
    pA : float | FloatArray
        Conversion of A groups.
    pC : float | FloatArray
        Conversion of C groups.
    r_CD_AB : float
        Initial molar ratio of CD/AB molecules.
    MAB : float
        Molar mass of a reacted AB unit.
    MCD : float
        Molar mass of a reacted CD unit.

    Returns
    -------
    float | FloatArray
        Number-average molar mass, $M_n$.
    """

    a = pA
    c = pC
    v = r_CD_AB

    Mn = (MAB + v*MCD)/(1 + v - (a + v*c))

    return Mn


def Case_8(pB: Union[float, FloatArray],
           pC: Union[float, FloatArray],
           r_BC_A: float,
           r_CC_BB: float,
           MAA: float,
           MBB: float,
           MCC: float) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AA reacting with BB and CC, where B does
    not react with C (AA + BB + B'B').

    Alternative notation:

    \begin{matrix}
    AA&  +&  BB &  +& CC\\
    \updownarrow&   & \updownarrow& & \updownarrow\\
    AA&  +&  BB &  +& B'B'
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
    pB : float | FloatArray
        Conversion of B groups.
    pC : float | FloatArray
        Conversion of C groups.
    r_BC_A : float
        Initial molar ratio of (B + C)/A groups.
    r_CC_BB : float
        Initial molar ratio of CC/BB molecules (or, equivalently, C/B groups).
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
    """

    b = pB
    c = pC
    v = r_CC_BB
    a = r_BC_A*(b + v*c)/(1 + v)

    Mn = (MAA + a*(MBB + v*MCC)/(b + v*c))/(1 - 2*a + a*(1 + v)/(b + v*c))

    return Mn


def Case_9(pB: Union[float, FloatArray],
           pC: Union[float, FloatArray],
           pD: Union[float, FloatArray],
           r_CD_AB: float,
           r_BB_AA: float,
           r_DD_CC: float,
           MAA: float,
           MBB: float,
           MCC: float,
           MDD: float) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AA and BB reacting with CC and DD, where
    A and B react only with C and D (AA + A'A' + BB + B'B').

    Alternative notation:

    \begin{matrix}
    AA &  + &  BB &  + &  CC  & + & DD \\
    \updownarrow&   & \updownarrow& & \updownarrow& & \updownarrow\\
    AA &  + &  A'A' &  + & BB & + & B'B'
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
    pB : float | FloatArray
        Conversion of B groups.
    pC : float | FloatArray
        Conversion of C groups.
    pD : float | FloatArray
        Conversion of D groups.
    r_CD_AB : float
        Initial molar ratio of (C+D)/(A+B) groups.
    r_BB_AA : float
        Initial molar ratio of BB/AA molecules (or, equivalently, B/A groups).
    r_DD_CC : float
        Initial molar ratio of DD/CC molecules (or, equivalently, D/C groups).
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
    """

    b = pB
    c = pC
    d = pD
    v = r_BB_AA
    r = r_DD_CC
    a = r_CD_AB*(1 + v)/(1 + r)*(c + r*d) - v*b

    Mn = (MAA + v*MBB + (a + v*b)/(c + r*d)*(MCC + r*MDD)) / \
        (1 - a + v*(1 - b) + (a + v*b)/(c + r*d)*(1 - c + r*(1 - d)))

    return Mn


def Case_10(pB: Union[float, FloatArray],
            pC: Union[float, FloatArray],
            pD: Union[float, FloatArray],
            r_BCD_A: float,
            r_BC_DD: float,
            MAA: float,
            MBC: float,
            MDD: float) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AA reacting with BC and DD, where A
    reacts only with B, C or D (AA + BB' + B''B'').

    Alternative notation:

    \begin{matrix}
    AA&  +&  BC &  +& DD\\
    \updownarrow&   & \updownarrow& & \updownarrow\\
    AA&  +&  BB' &  +& B''B''
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
    pB : float | FloatArray
        Conversion of B groups.
    pC : float | FloatArray
        Conversion of C groups.
    pD : float | FloatArray
        Conversion of D groups.
    r_BCD_A : float
        Initial molar ratio of (B + C + D)/A groups.
    r_BC_DD : float
        Initial molar ratio of BC/DD molecules.
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
    """

    b = pB
    c = pC
    d = pD
    v = r_BC_DD/2
    a = r_BCD_A*(v*(b + c) + d)/(2*v + 1)

    Mn = (MAA + a*(MDD + 2*v*MBC)/(v*(b + c) + d)) / \
        (1 - a + a/(v*(b + c) + d)*((1 - d) + v*(2 - b - c)))

    return Mn


def Case_11(pB: Union[float, FloatArray],
            pC: Union[float, FloatArray],
            pD: Union[float, FloatArray],
            r_BC_AA: float,
            r_DD_AA: float,
            MAA: float,
            MBC: float,
            MDD: float) -> Union[float, FloatArray]:
    r"""Case's analytical solution for AA and DD reacting with BC, where A
    and B react only with C and D (AA + A'B' + BB).

    Alternative notation:

    \begin{matrix}
    AA&  +&  BC &  +& DD\\
    \updownarrow&   & \updownarrow& & \updownarrow\\
    AA&  +&  A'B' &  +& BB
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
    pB : float | FloatArray
        Conversion of B groups.
    pC : float | FloatArray
        Conversion of C groups.
    pD : float | FloatArray
        Conversion of D groups.
    r_BC_AA : float
        Initial molar ratio of BC/AA molecules.
    r_DD_AA : float
        Initial molar ratio of DD/AA molecules.
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
    """

    b = pB
    c = pC
    d = pD
    v = r_BC_AA/2
    a = v*(c - b) + r_DD_AA*d

    Mn = (MAA + 2*v*MBC + (a + v*(b - c))/d*MDD) / \
        (1 - 2*a + 2*v*(1 - b) + (a + v*(b - c))/d)

    return Mn
