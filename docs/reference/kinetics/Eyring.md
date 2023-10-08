# polykin.kinetics

::: polykin.kinetics.thermal
    options:
        members:
            - Eyring
            - eyring

## Examples

Define and evaluate a rate coefficient from transition state properties.

```pycon exec="on" source="console"
>>> from polykin.kinetics import Eyring
>>>
>>> k = Eyring(
...     DSa=20., DHa=5e4, kappa=0.8, Tmin=273., Tmax=373., symbol='k',
...     name='A->B')
>>>
>>> print(k(25.,'C'))
```