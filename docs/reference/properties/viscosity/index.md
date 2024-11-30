# Viscosity (polykin.properties.viscosity)

This module implements methods to calculate the viscosity of pure gases, gas
mixtures, and liquid mixtures.

|                     |                     Gas             |        Liquid        |
|---------------------|:-----------------------------------:|:--------------------:|
| DIPPR equations     | [DIPPR102]                          |  [DIPPR101], [Yaws]  |
| Estimation methods  | [MUV_Lucas], [MUVMX_Lucas]          | —                    |
| Mixing rules        | [MUVMX2_Herning_Zipperer]           | [MULMX2_Perry]       |
| Pressure correction | [MUVPC_Jossi], [MUVMXPC_Dean_Stiel] | —                    |

[:simple-jupyter: Tutorial](../../../tutorials/viscosity){ .md-button }

[DIPPR101]: ../equations/index.md#polykin.properties.equations.dippr.DIPPR101
[DIPPR102]: ../equations/index.md#polykin.properties.equations.dippr.DIPPR102
[Yaws]: ../equations/index.md#polykin.properties.equations.viscosity.Yaws

[MUV_Lucas]: MUV_Lucas.md
[MUVMX_Lucas]: MUVMX_Lucas.md
[MUVMX2_Herning_Zipperer]: MUVMX2_Herning_Zipperer.md
[MULMX2_Perry]: MULMX2_Perry.md
[MUVPC_Jossi]: MUVPC_Jossi.md
[MUVMXPC_Dean_Stiel]: MUVMXPC_Dean_Stiel.md
