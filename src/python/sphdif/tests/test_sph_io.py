import numpy as np
from numpy.testing import assert_equal

from sphdif import sph_io

def test_round_trip():
    sph_io.savez('__test.npz', y=1)
    data = sph_io.load('__test.npz')
    assert_equal(data['y'], 1)
    sph_io.remove('__test.npz')


