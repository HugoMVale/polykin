# polykin.properties.thermal_conductivity

::: polykin.properties.thermal_conductivity
    options:
        members:
            - KVPC_stiel_thodos

## Examples

Estimate the residual thermal conductivity of ethylene at 350 K and 100 bar.

```python exec="on" source="console"
from polykin.properties.thermal_conductivity import KVPC_stiel_thodos

V = 1.8e-4  # m³/mol, @ 350K, 100 bar

k_residual = KVPC_stiel_thodos(V=V, M=28.05e-3,
                               Tc=282.4, Pc=50.4e5, Zc=0.280)

print(k_residual, "W/(m·K)")
```
