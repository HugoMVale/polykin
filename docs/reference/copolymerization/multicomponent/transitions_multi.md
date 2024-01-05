# polykin.copolymerization

::: polykin.copolymerization
    options:
        members:
            - transitions_multi

## Examples

```python exec="on" source="material-block"
from polykin.copolymerization import transitions_multi
import numpy as np

r = np.ones((3, 3))
r[0, 1] = 0.2
r[1, 0] = 2.3
r[0, 2] = 3.0
r[2, 0] = 0.9
r[1, 2] = 0.4
r[2, 1] = 1.5

P = transitions_multi(np.array([0.5, 0.3, 0.2]), r)

print(f"P11 = {P[0]:.2f}; P22 = {P[1]:.2f}; P33 = {P[2]:.2f}")
```
