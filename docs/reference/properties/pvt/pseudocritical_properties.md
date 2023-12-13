# polykin.properties.pvt

::: polykin.properties.pvt
    options:
        members:
            - pseudocritical_properties

## Examples

Estimate the pseudocritical properties of 60 mol% ethylene/nitrogen gas mixture.

```python exec="on" source="console"
from polykin.properties.pvt import pseudocritical_properties
import numpy as np

y = np.array([0.6, 0.4])
Tc = np.array([282.4, 126.2])    # K
Pc = np.array([50.4e5, 33.9e5])  # Pa
Zc = np.array([0.280, 0.290])
w = np.array([0.089, 0.039])

out = pseudocritical_properties(y, Tc, Pc, Zc, w) 

print(out)
```
