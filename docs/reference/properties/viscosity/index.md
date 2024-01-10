# Viscosity (polykin.properties.viscosity)

::: polykin.properties.viscosity

|                     |                     Gas                                                         |        Liquid                     |
|---------------------|:-------------------------------------------------------------------------------:|:---------------------------------:|
| DIPPR equations     | [`DIPPR102`](../equations/index.md#polykin.properties.equations.dippr.DIPPR102) |  [`DIPPR101`](../equations/index.md#polykin.properties.equations.dippr.DIPPR101), [`Yaws`](../equations/index.md#polykin.properties.equations.viscosity.Yaws)  |
| Estimation methods  | [`MUV_Lucas`](MUV_Lucas.md), [`MUVMX_Lucas`](MUVMX_Lucas.md)                    | —                                 |
| Mixing rules        | [`MUVMX2_Herning_Zipperer`](MUVMX2_Herning_Zipperer.md)                         | [`MULMX2_Perry`](MULMX2_Perry.md) |
| Pressure correction | [`MUVPC_Jossi`](MUVPC_Jossi.md), [`MUVMXPC_Dean_Stiel`](MUVMXPC_Dean_Stiel.md)  | —                                 |

!!! info

    For illustration examples, please refer to the associated
    [tutorial](../../../tutorials/viscosity).
