# polykin.copolymerization

::: polykin.copolymerization
    options:
        members:
            - convert_Qe_to_r

## Examples

Estimate  the reactivity ratio matrix for styrene (1) and methyl methacrylate (2), and vinyl
acetate(3) using Q-e values from the literature.

```python exec="on" source="material-block"
from polykin.copolymerization import convert_Qe_to_r

Qe1 = (1.0, -0.80)    # Sty
Qe2 = (0.78, 0.40)    # MMA
Qe3 = (0.026, -0.88)  # VAc
r = convert_Qe_to_r([Qe1, Qe2, Qe3])

print(r)
```
