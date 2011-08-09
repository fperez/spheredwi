"""Compute different distributions of points on the unit sphere.

The initial distribution of points is calculated using the Golden
Section Spiral as provided by Patrick Boucher on his blog [1]_.

.. note::

   TODO: Try the methods in [2]_, [3]_ and [4]_.
         Convert the MATLAB toolbox in [5]_ to Python.

See also [6]_ for a discussion on point distributions on the sphere.


.. [1] http://www.softimageblog.com/?author=1
.. [2] http://www.math.niu.edu/~rusin/known-math/95/equispace.elect
.. [3] http://sitemason.vanderbilt.edu/page/hmbADS
.. [4] http://cgafaq.info/wiki/Evenly_distributed_points_on_sphere
.. [5] http://eqsp.sourceforge.net/#pres
.. [6] http://www.math.niu.edu/~rusin/known-math/95/sphere.faq

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

def quadrature_points(N):
    """N is ignored for these pre-calculated optimal quadrature points.

    """
    pts = np.loadtxt('qsph1-37-492DP.dat')
    pts = pts[:, :3]

    return pts.T

def uniform_random(N):
    """Generate points inside the sphere randomly, discard those lying
    outside the sphere, and project the remaining points to the sphere surface.

    """
    # Try to generate enough points so that, after rejection, there will
    # be at least N left

    square_volume = 2**3
    sphere_volume = 4 / 3 * np.pi
    p = 1 / (1 - sphere_volume / square_volume) * 1.1 * N

    x = np.random.uniform(low=-1, high=1, size=p)
    y = np.random.uniform(low=-1, high=1, size=p)
    z = np.random.uniform(low=-1, high=1, size=p)

    r = np.sqrt(x**2 + y**2 + z**2)
    mask = ~((r == 0) | (r > 1))

    r = r[mask][:N]
    x = x[mask][:N]
    y = y[mask][:N]
    z = z[mask][:N]

    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    r = np.ones_like(theta)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z
    

def charged_particles(N):
    """Simulate charged particles floating on the sphere.

    """
    pass


if __name__ == "__main__":
    N = 492

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def plot_point_dists(dists):
        rows = 2
        cols = len(dists)

        for i, (name, f) in enumerate(dists):
            x, y, z = f(N)

            c = np.linspace(0, 1, len(x))
            
            ax = plt.subplot(rows, cols, i + 1, projection='3d')
            ax.set_axis_off()
            ax.scatter(x, y, z, 'o', color=plt.cm.jet(c))
            ax.set_aspect('equal')
            ax.set_title(name)


    plot_point_dists([('Golden Section Spiral', golden_points),
                      ('Optimal Quadrature', quadrature_points),
                      ('Uniform Random', uniform_random)])
    plt.show()
