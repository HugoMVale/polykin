# polykin.properties.vaporization_enthalpy

::: polykin.properties.vaporization_enthalpy
    options:
        members:
            - DHVL_Kistiakowsky_Vetere

## Examples

Estimate the vaporization enthalpy of butadiene at the normal boiling temperature.

```python exec="on" source="material-block"
from polykin.properties.vaporization_enthalpy import DHVL_Kistiakowsky_Vetere

DHVL = DHVL_Kistiakowsky_Vetere(Tb=268.6, M=54.1e-3, kind='hydrocarbon')

print(f"{DHVL/1e3:.1f} kJ/mol")
```
