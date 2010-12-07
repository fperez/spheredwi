"""Test utilities for sphquad library.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function

# Stdlib imports

# Third-party imports
import nose.tools as nt
import numpy as np
import numpy.testing as npt

# Our own imports
import sphquad

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

def test_legp():
    p = sphquad.legp(0, 0)
    npt.assert_equal(p[0], 1)
