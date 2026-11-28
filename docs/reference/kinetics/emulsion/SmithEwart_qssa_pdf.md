# polykin.kinetics.emulsion

::: polykin.kinetics.emulsion.smithewart
    options:
        members:
            - SmithEwart_qssa_pdf

### Graphical Illustration

Illustration of the probability distribution of the number of radicals per particle according to the Smith-Ewart for given values of $\alpha$ and $m$.

```python exec="on" source="above" html="on"
import matplotlib.pyplot as plt
import numpy as np
from polykin.kinetics.emulsion import SmithEwart_qssa_pdf
from polykin.utils.docs import to_html

fig, ax = plt.subplots()
n = np.arange(0, 6)
alpha = 1.0
m = 1e-2
pdf = SmithEwart_qssa_pdf(n, alpha, m)
ax.bar(n, pdf, label=fr"$\alpha={alpha}, m={m}$")
ax.grid(True)
ax.legend(loc="upper right")
ax.set_xlabel("Number of radicals per particle")
ax.set_ylabel("Probability")

print(to_html(fig))
```
