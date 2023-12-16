# polykin.properties.thermal_conductivity

::: polykin.properties.thermal_conductivity
    options:
        members:
            - KVMXPC_stiel_thodos

## Examples

Estimate the residual thermal conductivity of a 50 mol% ethylene/propylene mixture
at 350 K and 100 bar.

```python exec="on" source="console"
from polykin.properties.thermal_conductivity import KVMXPC_stiel_thodos
import numpy as np

V = 1.12e-4  # m³/mol, with Peng-Robinson

y = np.array([0.5, 0.5])
M = np.array([28.05e-3, 42.08e-3])  # kg/mol
Pc = np.array([50.4e5, 46.0e5])     # Pa
Tc = np.array([282.4, 364.9])       # K
Zc = np.array([0.280, 0.274])
w = np.array([0.089, 0.144]) 

k_residual = KVMXPC_stiel_thodos(V, y, M, Tc, Pc, Zc, w)

print(f"{k_residual:.2e} W/(m·K)")
```
