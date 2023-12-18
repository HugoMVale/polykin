# polykin.properties.thermal_conductivity

::: polykin.properties.thermal_conductivity
    options:
        members:
            - KVPC_stiel_thodos

## Examples

Estimate the residual thermal conductivity of ethylene at 350 K and 100 bar.

```python exec="on" source="console"
from polykin.properties.thermal_conductivity import KVPC_stiel_thodos

v = 1.84e-4 # m³/mol, with Peng-Robinson

k_residual = KVPC_stiel_thodos(v=v, M=28.05e-3,
                               Tc=282.4, Pc=50.4e5, Zc=0.280)

print(f"{k_residual:.2e} W/(m·K)")
```
