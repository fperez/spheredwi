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

def sph2car(r, theta, phi):
    """Convert spherical coordinates to Cartesian coordinates."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z

def car2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    return r, theta, phi


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

    return sph2car(r, theta, phi)

def saff_kuijlaars(N):
    """

    References
    ----------
    'Distributing many points on a sphere' by E.B. Saff and A.B.J. Kuijlaars,
    Mathematical Intelligencer, 19.1 (1997), pp. 5--11

    """
    k = np.arange(N)
    h = -1 + 2 * k / (N - 1)
    theta = np.arccos(h)
    phi = np.zeros_like(h)
    for i in range(1, N - 1):
        phi[i] = (phi[i - 1] + 3.6 / np.sqrt(N * (1 - h[i]**2))) % (2 * np.pi)

    return sph2car(np.ones_like(theta), theta, phi)


def charged_particles(N, init_func=golden_points):
    """Simulate charged particles floating on the sphere.

    Parameters
    ----------
    N : int
        Number of particles to place on sphere.
    init_func : f(N), callable
        Function called to obtain the initial points,
        before the simulation is started.

    """
    x, y, z = init_func(N)
    r, theta, phi = car2sph(x, y, z)
    r.fill(1)

    # We can derive the formulas for the cost function and derivative,
    # but for now we'll do it numerically

    # Notes:
    #
    # It is quite possible for the simulation to carry on
    # indefinitely, with points simply rotating around the sphere.  It
    # may therefore be necessary to rotate the points so that at least
    # two remain in the same position with every iteration.
    #
    # Direction of repulsive force between two vectors, a and b:
    #
    # t = a x (a x b); t = t / |t|

    def forces(theta, phi):
        """
        Determine the electrostatic force on each particle.
        
        Parameters
        ----------
        theta, phi : ndarray
            Angles of points on the unit sphere.

        Notes
        -----
        There are several ways of computing arc lengths on the sphere.  Some of
        these suffer from numeric instability, mainly due to `arccos` being so
        sensitive around 1 and -1.

        For example, in Cartesian coordinates, the distance can be
        computed using the dot product.  Since,

        ::

          a.dot(b) = a b cos(rho)

        we have that the included angle, rho, is given by
        ``arccos(a.dot(b) / ab)``.  The arc length is then ``R * rho``
        where R is the sphere radius.

        Similarly, in spherical coordinates, the included angle
        is computed as::

          arccos(cos(theta1) * cos(theta2)
                 + sin(theta1) * sin(theta2) * cos(phi2 - phi1))

        In order to address numerical issues, the Haversine
        formula may be used, but that formule, in turn, also
        suffers from round-off errors when dealing with anti-podal vectors.

        To avoid these issues, it is recommended using the Vincenty
        formula for distances on ellipsoids, specialised for the
        sphere.

        .. note::

           Many of the formulas given on this topic use latitude
           and longitude, whereas in physics the elevation is
           measured downwards from the z axis.

           To convert formulas, substitude cos for sin for
           single angles, and leave trigonometric identities
           in place when operating on angle differences.  Or,
           alternatively, simply subtract pi/2 from the elevation
           before using them.

        References
        ----------
        http://en.wikipedia.org/wiki/Great-circle_distance#Formulae

        """
        theta = np.atleast_2d(theta)
        phi = np.atleast_2d(phi)

        dp = phi.T - phi
        cdp = np.cos(dp)

        # Compute arc lengths between nodes
        D = np.arctan2(np.sqrt((np.sin(theta) * np.sin(dp))**2 + \
                               (np.sin(theta.T) * np.cos(theta) - \
                                np.cos(theta.T) * np.sin(theta) * cdp)**2),
                       np.cos(theta.T) * np.cos(theta) + \
                       np.sin(theta.T) * np.sin(theta) * cdp)

        # Inverse distance squared
        D[np.diag_indices_from(D)] = 1
        Di = 1 / D**2
        Di[np.diag_indices_from(D)] = 0
        
        E = np.sum(Di)

    forces(theta, phi)
    


if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-p', '--points', dest='points', type=int,
                      default=492, help='Number of points')
    parser.add_option('-n', '--no-coverage', dest='no_coverage',
                      action="store_true",
                      help="Disable coverage plots")
    (options, args) = parser.parse_args()

    N = options.points

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import dwicoverage

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

            if not options.no_coverage:
                ax = plt.subplot(rows, cols, cols + i + 1)
                bv_good, sphere, field, bv_miss = \
                         dwicoverage.build_coverage(np.vstack((x, y, z)),
                                                    None, None)
                dwicoverage.show_coverage_2d(bv_good, sphere, field, bv_miss,
                                             vmin=0, ax=ax, markersize=10)


    plot_point_dists([('Golden Section Spiral', golden_points),
                      ('Optimal Quadrature', quadrature_points),
                      ('Uniform Random', uniform_random),
                      ('Saff/Kuijlaars', saff_kuijlaars),
                      ('Charged Particles', charged_particles)])
    plt.show()


############# Tests #############

from numpy.testing import *

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
