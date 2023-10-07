# polykin.diffusion

::: polykin.diffusion.vrentasduda
    options:
        members:
            - VrentasDudaBinary

## Examples

Estimate the mutual and self-diffusion coefficient of toluene in polyvinylacetate.

```pycon exec="on" source="console"
>>> from polykin.diffusion import VrentasDudaBinary
>>>
>>> d = VrentasDudaBinary(
...     D0=4.82e-4, E=0., V1star=0.917, V2star=0.728, z=0.82,
...     K11=1.45e-3, K12=4.33e-4, K21=-86.32, K22=-258.2, X=0.5,
...     unit='cm²/s',
...     name='Tol(1)/PVAc(2)')
>>>
>>> # D at w1=0.2 and T=25°C.
>>> D = d(0.2, 25., Tunit='C')
>>> print("D= ", D)
>>> # D1 at w1=0.2 and T=50°C.
>>> D1 = d(0.2, 25., Tunit='C', selfd=True)
>>> print("D1=", D1)
```
