# polykin.properties.vaporization_enthalpy

::: polykin.properties.vaporization_enthalpy
    options:
        members:
            - hvapb_kistiakowsky_vetere

## Examples

Estimate the vaporization enthalpy of butdiene at the normal boiling temperature.

```python exec="on" source="console"
from polykin.properties.vaporization_enthalpy import hvapb_kistiakowsky_vetere

hvap = hvapb_kistiakowsky_vetere(Tb=268.6, M=54.1e-3, kind='hydrocarbon')

print(hvap/1e3, "kJ/mol")
```
