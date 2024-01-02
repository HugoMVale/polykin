# polykin.copolymerization

::: polykin.copolymerization
    options:
        members:
            - TerminalModel
<!--
## Examples

Analyze the behavior of the butyl acrylate and styrene system,
using parameters from the literature.

```python exec="on" source="console"
from polykin.copolymerization import TerminalModel

model = TerminalModel(r1=0.16, r2=0.70, 
        M1='BA', M2='ST', name='BA/ST, 50°C')

print("f1azeo =", model.azeotrope)
print("F1(f1=0.5) =", model.F1(0.5))
```
-->
