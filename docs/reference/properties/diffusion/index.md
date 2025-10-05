# Diffusion (polykin.properties.diffusion)

This module implements methods to calculate infinite-dilution, mutual and self-diffusion
coefficients in binary liquid and gas mixtures.

|                   |       Gas      |                Liquid                | Polymer solution    |
|-------------------|:--------------:|:------------------------------------:|---------------------|
| Infinite-dilution | [DV_Wilke_Lee] | [DL_Wilke_Chang], [DL_Hayduk_Minhas] | [VrentasDudaBinary] |
| Mutual-diffusion  | [DV_Wilke_Lee] |                   —                  | [VrentasDudaBinary] |
| Self-diffusion    | [DV_Wilke_Lee] |                   —                  | [VrentasDudaBinary] |

[:simple-jupyter: Tutorial](../../../tutorials/diffusion_coefficients){ .md-button }

[DV_Wilke_Lee]: DV_Wilke_Lee.md
[DL_Wilke_Chang]: DL_Wilke_Chang.md
[DL_Hayduk_Minhas]: DL_Hayduk_Minhas.md
[VrentasDudaBinary]: VrentasDudaBinary.md