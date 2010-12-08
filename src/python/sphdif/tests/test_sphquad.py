"""Test utilities for sphquad library.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function

# Stdlib imports
import os

# Third-party imports
import nose.tools as nt
import numpy as np
import numpy.testing as npt

# Our own imports
from .. import sphquad

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

def test_legp():
    p = sphquad.legp(0, 0)
    npt.assert_equal(p[0], 1)
    #path = os.path.join(
    my_dir = os.path.split(__file__)[0]
    legp_true = np.loadtxt(os.path.join(my_dir, 'legendrep.txt'))
    max_order = legp_true.shape[0] - 1
    x_grid = legp_true[1]
    legp = sphquad.legp(x_grid, max_order)
    npt.assert_almost_equal(legp, legp_true, 13)
