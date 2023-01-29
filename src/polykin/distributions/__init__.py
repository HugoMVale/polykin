"""
PolyKin `distributions` provides classes to create, visualize, fit, combine,
integrate, etc. theoretical and experimental chain-length distributions.

"""

from polykin.distributions.analytical import \
    Flory, Poisson, LogNormal, SchulzZimm
from polykin.distributions.experimental import DatalDistribution
from polykin.distributions.baseclasses import plotdists
