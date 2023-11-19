# polykin.properties.viscosity

::: polykin.properties.viscosity
    options:
        members:
            - MUVPC_jossi

## Examples

Estimate the residual viscosity of ethylene at 350 K and 10 bar.

```python exec="on" source="console"
from polykin.properties.viscosity import MUVPC_jossi

Vc = 130.4 # cm3/mol
V = 2.8e3  # cm3/mol, @ 350K, 10 bar
rhor = Vc/V

mu_residual = MUVPC_jossi(rhor=rhor, Tc=282.4, Pc=50.4e5, M=28.05e-3)

print(mu_residual, "Pa.s")
```
