# polykin.properties.diffusion

::: polykin.properties.diffusion
    options:
        members:
            - DL_hayduk_minhas

## Examples

Estimate the diffusion coefficient of vinyl chloride through liquid water.

```python exec="on" source="material-block"
from polykin.properties.diffusion import DL_hayduk_minhas

D = DL_hayduk_minhas(
    T=298.,           # temperature
    method='aqueous', # equation for aqueous solutions
    MA=62.5e-3,       # molar mass of vinyl chloride
    rhoA=910.,        # density of vinyl chloride at the normal boiling point
    viscB=0.89e-3     # viscosity of water at solution temperature
    )

print(f"{D:.2e} m²/s")
```
