
# pySubDisc package API: expose main interface functions and version
from .version import __version__

# Import main factory functions for subgroup discovery tasks
from .interface import singleNominalTarget
from .interface import singleNumericTarget
from .interface import doubleRegressionTarget
from .interface import doubleBinaryTarget
from .interface import doubleCorrelationTarget
from .interface import multiNumericTarget
from .interface import loadDataFrame

# Define what is exported when using 'from pysubdisc import *'
__all__ = [ singleNominalTarget,
            singleNumericTarget,
            doubleRegressionTarget,
            doubleBinaryTarget,
            doubleCorrelationTarget,
            multiNumericTarget,
            loadDataFrame ]
