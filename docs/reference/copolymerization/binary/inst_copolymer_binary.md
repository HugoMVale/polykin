# polykin.copolymerization

::: polykin.copolymerization.binary
    options:
        members:
            - inst_copolymer_binary

## Examples

```python exec="on" source="material-block"
from polykin.copolymerization import inst_copolymer_binary
import numpy as np

F1 = inst_copolymer_binary(f1=np.array([0.1, 0.5, 0.9]), r1=0.16, r2=0.70)

print(f"F1 = {F1}")
```
