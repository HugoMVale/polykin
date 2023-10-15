# polykin.properties.diffusion

::: polykin.properties.diffusion.estimation_methods
    options:
        members:
            - wilke_lee

## Examples

Estimate the diffusion coefficient of vinyl chloride through water vapor.

```pycon exec="on" source="console"
>>> from polykin.properties.diffusion import wilke_lee
>>>
>>> D = wilke_lee(
...     T=298.,       # temperature
...     P=1e5,        # pressure
...     MA=62.5e-3,   # molar mass of vinyl chloride
...     MB=18.0e-3,   # molar mass of water
...     rhoA=910.,    # density of vinyl chloride at the normal boiling point
...     rhoB=959.,    # density of water at the normal boiling point
...     TA=260.,      # normal boiling point of vinyl chloride
...     TB=373.,      # normal boiling point of water
...     )
>>>
>>> print(D)
```

Estimate the diffusion coefficient of vinyl chloride through air.

```pycon exec="on" source="console"
>>> from polykin.properties.diffusion import wilke_lee
>>> 
>>> D = wilke_lee(
...     T=298.,       # temperature
...     P=1e5,        # pressure
...     MA=62.5e-3,   # molar mass of vinyl chloride
...     MB=18.0e-3,   # molar mass of water
...     rhoA=910.,    # density of vinyl chloride at the normal boiling point
...     rhoB=None,    # air
...     TA=260.,      # normal boiling point of vinyl chloride
...     TB=None,      # air
...     )
>>>
>>> print(D)
```
