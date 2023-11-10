# polykin.properties.vaporization_enthalpy

::: polykin.properties.vaporization_enthalpy
    options:
        members:
            - DHVL_kistiakowsky_vetere

## Examples

Estimate the vaporization enthalpy of butdiene at the normal boiling temperature.

```python exec="on" source="console"
from polykin.properties.vaporization_enthalpy import DHVL_kistiakowsky_vetere

DHVL = DHVL_kistiakowsky_vetere(Tb=268.6, M=54.1e-3, kind='hydrocarbon')

print(DHVL/1e3, "kJ/mol")
```
