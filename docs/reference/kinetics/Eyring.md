# polykin.kinetics

::: polykin.kinetics.thermal
    options:
        members:
            - Eyring

## Examples

Define and evaluate a rate coefficient from transition state properties.

```python exec="on" source="material-block"
from polykin.kinetics import Eyring

k = Eyring(
    DSa=20.,        # activation entropy
    DHa=5e4,        # activation entropy
    kappa=0.8,      # transmission factor
    Tmin=273.,
    Tmax=373.,
    symbol='k',
    name='A->B'
    )

print(f"k = {k(25.,'C'):.2e} " + k.unit)
```
