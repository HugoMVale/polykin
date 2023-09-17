"""
PolyKin `coefficients` provides classes to create and visualize the types of
kinetic coefficients and physical property correlations most often found in
polymer reactor models.

"""

from polykin.coefficients.kinetics import Arrhenius, Eyring
from polykin.coefficients.dippr import DIPPR100, DIPPR101, DIPPR105
from polykin.coefficients.cld import TerminationCompositeModel, \
    PropagationHalfLength
from polykin.coefficients.baseclasses import plotcoeffs
