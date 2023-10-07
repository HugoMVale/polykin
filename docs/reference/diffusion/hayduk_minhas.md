# polykin.diffusion.hayduk_minhas

::: polykin.diffusion.estimation_methods
    options:
        members:
            - hayduk_minhas

## Examples

Estimate the diffusion coefficient of vinyl chloride through liquid water.

```pycon exec="on" source="console"
>>> from polykin.diffusion import hayduk_minhas
>>>
>>> D = hayduk_minhas(
...     T=298.,           # temperature
...     method='aqueous', # equation for aqueous solutions
...     MA=62.5e-3,       # molar mass of vinyl chloride
...     rhoA=910.,        # density of vinyl chloride at the normal boiling point
...     viscB=0.89e-3     # viscosity of water at solution temperature
...     )
>>>
>>> print(D)
```
