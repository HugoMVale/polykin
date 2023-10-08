# polykin.distributions

::: polykin.distributions.analyticaldistributions
    options:
        members:
            - Poisson

## Examples

Define a Poisson distribution and evaluate the corresponding probability density function and
cumulative distribution function for representative chain lengths.

```pycon exec="on" source="console"
>>> from polykin.distributions import Poisson
>>> 
>>> a = Poisson(100, M0=0.050, name='A')
>>> 
>>> print(a.Mz)
>>> print(a.pdf(a.DPn))
>>> print(a.cdf([a.DPn, a.DPw, a.DPz]))
```