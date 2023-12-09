# polykin.properties.thermal_conductivity

::: polykin.properties.thermal_conductivity
    options:
        members:
            - KVMX2_wassilijewa

## Examples

Estimate the thermal conductivity of a 50 mol% styrene/ethyl-benzene gas mixture at 25°C and
0.1 bar.

```python exec="on" source="console"
from polykin.properties.thermal_conductivity import KVMX2_wassilijewa
import numpy as np

y = np.array([0.5, 0.5])
k = np.array([1.00e-2, 1.55e-2]) # W/(m·K), from literature
M = np.array([104.15, 106.17])   # g/mol

k_mix = KVMX2_wassilijewa(y, k, M)

print(k_mix, "W/(m·K)")
```
