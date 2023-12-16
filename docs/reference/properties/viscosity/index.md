# Viscosity (polykin.properties.viscosity)

::: polykin.properties.viscosity

|                     |                     Gas                                                         |        Liquid                     |
|---------------------|:-------------------------------------------------------------------------------:|:---------------------------------:|
| DIPPR equations     | [`DIPPR102`](../equations/index.md#polykin.properties.equations.dippr.DIPPR102) |  [`DIPPR101`](../equations/index.md#polykin.properties.equations.dippr.DIPPR101), [`Yaws`](../equations/index.md#polykin.properties.equations.viscosity.Yaws)  |
| Estimation methods  | [`MUV_lucas`](MUV_lucas.md), [`MUVMX_lucas`](MUVMX_lucas.md)                    | —                                 |
| Mixing rules        | [`MUVMX2_herning_zipperer`](MUVMX2_herning_zipperer.md)                         | [`MULMX2_perry`](MULMX2_perry.md) |
| Pressure correction | [`MUVPC_jossi`](MUVPC_jossi.md), [`MUVMXPC_dean_stiel`](MUVMXPC_dean_stiel.md)  | —                                 |

For illustration examples, please refer to the associated
[tutorial](../../../tutorials/viscosity).