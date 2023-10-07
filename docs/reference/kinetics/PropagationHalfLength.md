# polykin.kinetics

::: polykin.kinetics.cld
    options:
        members:
            - PropagationHalfLength

## Examples

```pycon exec="on" source="console"
>>> from polykin.kinetics import PropagationHalfLength, Arrhenius
>>>
>>> kp = Arrhenius(
...     10**7.63, 32.5e3/8.314, Tmin=261., Tmax=366., symbol='k_p',
...     unit='L/mol/s', name='kp of styrene')
>>>
>>> kpi = PropagationHalfLength(kp, C=10, ihalf=0.5, name='kp(T,i) of styrene')
>>>
>>> # kp of a trimeric radical at 50°C
>>> print(kpi(T=50., i=3, Tunit='C'))
```
