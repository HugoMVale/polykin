# polykin.distributions

::: polykin.distributions.analyticaldistributions
    options:
        members:
            - LogNormal

## Examples

Define a LogNormal distribution and evaluate the corresponding probability density function and
cumulative distribution function for representative chain lengths.

```pycon exec="on" source="console"
>>> from polykin.distributions import LogNormal
>>> 
>>> a = LogNormal(100, PDI=3., M0=0.050, name='A')
>>> 
>>> print(a.Mz)
>>> print(a.pdf(a.DPn))
>>> print(a.cdf([a.DPn, a.DPw, a.DPz]))
```