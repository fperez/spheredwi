import numpy as np
from numpy.testing import assert_almost_equal

from sphdif.coord import sph2car, car2sph

def test_sph2car():
    x, y, z = sph2car([1], [np.pi / 2], [0])
    assert_almost_equal(x, 1, decimal=5)
    assert_almost_equal(y, 0, decimal=5)
    assert_almost_equal(z, 0, decimal=5)

def test_car2sph():
    r, theta, phi = car2sph(1, 0, 0)
    assert_almost_equal(r, 1, decimal=5)
    assert_almost_equal(theta, np.pi / 2, decimal=5)
    assert_almost_equal(phi, 0, decimal=5)

    r, theta, phi = car2sph(0, 1, 0)
    assert_almost_equal(r, 1, decimal=5)
    assert_almost_equal(theta, np.pi / 2, decimal=5)
    assert_almost_equal(phi, np.pi / 2, decimal=5)
