# polykin.copolymerization

::: polykin.copolymerization
    options:
        members:
            - inst_copolymer_multi

## Examples

```python exec="on" source="material-block"
from polykin.copolymerization import inst_copolymer_multi
import numpy as np

r = np.ones((3, 3))
r[0, 1] = 0.2
r[1, 0] = 2.3
r[0, 2] = 3.0
r[2, 0] = 0.9
r[1, 2] = 0.4
r[2, 1] = 1.5

F = inst_copolymer_multi(np.array([0.5, 0.3]), r)

print(f"F1 = {F[0]:.2f}; F2 = {F[1]:.2f}; F3 = {F[2]:.2f}")
```
