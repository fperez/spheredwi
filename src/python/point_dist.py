"""Compute different distributions of points on the unit sphere.

The initial distribution of points is calculated using the Golden
Section Spiral as provided by Patrick Boucher on his blog [1]_.

.. note::

   TODO: Try the methods in [2]_, [3]_ and [4]_.

.. [1] http://www.softimageblog.com/?author=1
.. [2] http://www.math.niu.edu/~rusin/known-math/95/equispace.elect
.. [3] http://sitemason.vanderbilt.edu/page/hmbADS
.. [4] http://cgafaq.info/wiki/Evenly_distributed_points_on_sphere

"""

from __future__ import division

import numpy as np
 
def golden_points(N):
    """Golden Section Spiral.

    Parameters
    ----------
    N : int
        Number of points to return.

    Returns
    -------
    x, y, z : (N,) ndarrays
        Cartesian coordinates of points distributed on the unit sphere.

    """
    inc = np.pi * (3 - np.sqrt(5))
    off = 2 / N
    k = np.arange(N)
    phi = k * inc
    y = k * off - 1 + (off / 2)
    r = np.sqrt(1 - y**2)
    x = np.cos(phi) * r
    z = np.sin(phi) * r

    return x, y, z

def charged_particles(N):
    """Simulate charged particles floating on the sphere.

    """
    pass


if __name__ == "__main__":
    N = 200

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def plot_point_dists(dists, subplot=(2, 2)):
        rows, cols = subplot
        for i, (name, f) in enumerate(dists):
            x, y, z = f(N)
            c = np.linspace(0, 1, N)
            
            ax = plt.subplot(rows, cols, i + 1, projection='3d')
            ax.set_axis_off()
            ax.scatter(x, y, z, 'o', color=plt.cm.jet(c))
            ax.set_aspect('equal')
            ax.set_title(name)


    plot_point_dists([('Golden Section Spiral', golden_points),
                      ('Golden Section Spiral', golden_points)])
    plt.show()
