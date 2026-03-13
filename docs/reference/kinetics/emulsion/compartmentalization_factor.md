# polykin.kinetics.emulsion

::: polykin.kinetics.emulsion.smithewart
    options:
        members:
            - compartmentalization_factor

### Graphical Illustration

Typical dependence of the compartmentalization factor on the dimensionless entry and desorption frequencies ($\alpha$ and $m$), and the average number of radicals per particle ($\bar{n}$).

```python exec="on" source="above" html="on"
import matplotlib.pyplot as plt
import numpy as np
from polykin.kinetics.emulsion import (
    compartmentalization_factor,
    nbar_Stockmayer_OToole,
)
from polykin.utils.docs import to_html

fig, ax = plt.subplots(2, 1)

alpha = np.logspace(-4, 4, 100)
for m in [0, 0.1, 1, 10]:
    Df = compartmentalization_factor(alpha, m)
    nbar = nbar_Stockmayer_OToole(alpha, m)
    ax[0].plot(alpha, Df,  label=rf"$m={m}$")
    ax[1].plot(nbar, Df)

ax[0].set_xlabel(r"$\alpha$")
ax[0].set_ylabel(r"$D_f$")
ax[0].set_xscale("log")
ax[0].set_xlim(1e-4, 1e4)
ax[0].grid(True)
ax[0].legend(loc="best")

ax[1].set_xlabel(r"$\bar{n}$")
ax[1].set_ylabel(r"$D_f$")
ax[1].set_xscale("log")
ax[1].set_xlim(1e-2, 1e2)
ax[1].grid(True)

fig.align_ylabels()
fig.tight_layout()

print(to_html(fig))
```
