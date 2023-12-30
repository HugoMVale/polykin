# polykin.kinetics

::: polykin.kinetics.cld
    options:
        members:
            - TerminationCompositeModel

## Examples

```python exec="on" source="material-block"
from polykin.kinetics import TerminationCompositeModel, Arrhenius

kt11 = Arrhenius(1e9, 2e3, T0=298., symbol='k_t(T,1,1)', unit='L/mol/s',
                name='kt11 of Y')

ktij = TerminationCompositeModel(kt11, icrit=30, name='ktij of Y')

# kt between radicals with chain lengths 150 and 200 at 25°C
print(f"{ktij(T=25., i=150, j=200, Tunit='C'):.2e} " + kt11.unit)
```
