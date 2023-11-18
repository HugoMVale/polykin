# polykin.properties.thermal_conductivity

::: polykin.properties.thermal_conductivity
    options:
        members:
            - KLMX2_li

## Examples

Estimate the viscosity of a 50 wt% styrene/isoprene mixture at 20°C.

```python exec="on" source="console"
from polykin.properties.thermal_conductivity import KLMX2_li
import numpy as np

w = np.array([0.5, 0.5])
k = np.array([0.172, 0.124])    # W/(m.K), from literature
rho = np.array([0.909, 0.681])  # kg/L

KL = KLMX2_li(w=w, k=k, rho=rho)

print(KL, "W/(m.K)")
```
