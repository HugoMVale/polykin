# polykin.properties.thermal_conductivity

::: polykin.properties.thermal_conductivity
    options:
        members:
            - KVPC_stiel_thodos

## Examples

Estimate the residual thermal conductivity of ethylene at 350 K and 10 bar.

```python exec="on" source="console"
from polykin.properties.thermal_conductivity import KVPC_stiel_thodos

Vc = 130.4 # cm3/mol
V = 2.8e3  # cm3/mol, @ 350K, 10 bar
rhor = Vc/V

k_residual = KVPC_stiel_thodos(rhor=rhor, M=28.05e-3, Tc=282.4, Pc=50.4e5, Zc=0.280)

print(k_residual, "W/(m·K)")
```
