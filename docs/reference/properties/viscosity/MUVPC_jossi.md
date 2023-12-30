# polykin.properties.viscosity

::: polykin.properties.viscosity
    options:
        members:
            - MUVPC_jossi

## Examples

Estimate the residual viscosity of ethylene at 350 K and 100 bar.

```python exec="on" source="material-block"
from polykin.properties.viscosity import MUVPC_jossi

vc = 130. # cm³/mol
v  = 184. # cm³/mol, with Peng-Robinson
rhor = vc/v

mu_residual = MUVPC_jossi(rhor=rhor, Tc=282.4, Pc=50.4e5, M=28.05e-3)

print(f"{mu_residual:.2e} Pa·s")
```
