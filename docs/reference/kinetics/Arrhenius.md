# polykin.kinetics

::: polykin.kinetics.thermal
    options:
        members:
            - Arrhenius

## Examples

Define and evaluate the propagation rate coefficient of styrene.

```python exec="on" source="console"
from polykin.kinetics import Arrhenius

kp = Arrhenius(
    10**7.63, 32.5e3/8.314, Tmin=261., Tmax=366., symbol='k_p',
    unit='L/mol/s', name='kp of styrene')

print(f"kp = {kp(25.,'C'):.1f} " + kp.unit)
```
