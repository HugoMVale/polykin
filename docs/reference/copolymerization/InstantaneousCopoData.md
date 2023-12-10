# polykin.copolymerization

::: polykin.copolymerization.analysis
    options:
        members:
            - InstantaneousCopoData

## Examples

Define a dataset based on the copolymerization data reported by
[Kostanski & Hamielec (1992)](https://doi.org/10.1016/0032-3861(92)90659-K).

```python exec="on" source="console"
from polykin.copolymerization import InstantaneousCopoData

ds1 = InstantaneousCopoData(
    M1='BA',
    M2='ST',
    f1=[0.186, 0.299, 0.527, 0.600, 0.700, 0.798],
    F1=[0.196, 0.279, 0.415, 0.473, 0.542, 0.634],
    sigma_f=0.001,
    sigma_F=0.001,
    name="BA/ST, 110°C",
    reference="Kostanski & Hamielec (1992)") 
```
