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


def test_spherical_distances():
    pi = np.pi
    # lay out 7 points on a radius 2 sphere
    r = 2.0
    c = np.sqrt(2.0)/2  # cos(45deg)
    pts = np.array([ [r, 0, 0],
                     [-r, 0, 0],
                     [0, r, 0],
                     [0, 0, r],
                     [0, -r, 0],
                     [0, 0, -r],
                     [0, -c*r, c*r]])

    # A few statically computed geodesic distances
    dist = {}
    dist[0,1] = r*pi
    for i in range(2, 7):
        dist[0, i] = r*pi/2
    dist[2, 3] = r*pi/2
    dist[2, 4] = r*pi
    dist[4, 6] = r*pi/4

    # Compute all geodesic distances
    gdist = sphquad.spherical_distances(pts, pts)
    for (i, j), d in dist.items():
        npt.assert_almost_equal(gdist[i,j], d, 13)
    
