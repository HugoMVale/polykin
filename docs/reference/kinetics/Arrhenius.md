# polykin.kinetics

::: polykin.kinetics.thermal
    options:
        members:
            - Arrhenius

## Examples

Define and evaluate the propagation rate coefficient of styrene.

```python exec="on" source="material-block"
from polykin.kinetics import Arrhenius

kp = Arrhenius(
    10**7.63,               # pre-exponential factor
    32.5e3/8.314,           # Ea/R, K
    Tmin=261.,
    Tmax=366.,
    symbol='k_p',
    unit='L/mol/s',
    name='kp of styrene'
    )

print(f"kp = {kp(25.,'C'):.1f} " + kp.unit)
```
