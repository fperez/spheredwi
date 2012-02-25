import numpy as np
import os

import coord

def arc_length(theta1, phi1, theta2, phi2):
    """
    Parameters
    ----------
    theta1, phi1 : float, ndarray
        First angle coordinates. Theta is the polar angle (z-axis = 0), phi the
        azimuthal angle.
    theta2, phi2 : float, ndarray
        Second angle.  Because of broadcasting, angles can also be given as
        ndarrays.

    Notes
    -----
    There are several ways of computing arc lengths on the sphere.  Some of
    these suffer from numeric instability, mainly due to `arccos` being so
    sensitive around 1 and -1.

    For example, in Cartesian coordinates, the distance can be computed using
    the dot product.  Since,

    ::

        a.dot(b) = a b cos(rho)

    we have that the included angle, rho, is given by ``arccos(a.dot(b) /
    ab)``.  The arc length is then ``R * rho`` where R is the sphere radius.

    Similarly, in spherical coordinates, the included angle is computed as::

      arccos(cos(theta1) * cos(theta2)
             + sin(theta1) * sin(theta2) * cos(phi2 - phi1))

    In order to address numerical issues, the Haversine formula may be used,
    but that formule, in turn, also suffers from round-off errors when dealing
    with anti-podal vectors.

    To avoid these issues, it is recommended using the Vincenty formula for
    distances on ellipsoids, specialised for the sphere.

    .. note::

       Many of the formulas given on this topic use latitude and longitude,
       whereas in physics the elevation is measured downwards from the z axis.

       To convert formulas, substitude cos for sin for single angles, and leave
       trigonometric identities in place when operating on angle differences.
       Or, alternatively, simply subtract pi/2 from the elevation before using
       them.

    References
    ----------
    http://en.wikipedia.org/wiki/Great-circle_distance#Formulae

    """
    dp = phi2 - phi1
    cdp = np.cos(dp)

    return np.arctan2(np.sqrt((np.sin(theta1) * np.sin(dp))**2 + \
                              (np.sin(theta2) * np.cos(theta1) - \
                               np.cos(theta2) * np.sin(theta1) * cdp)**2),
                      np.cos(theta2) * np.cos(theta1) + \
                      np.sin(theta2) * np.sin(theta1) * cdp)


def quadrature_points(N=72):
    """Load quadrature points on the sphere.

    Parameters
    ----------
    N : int, {72, 132, 492}
        A quadrature set with N points is loaded.

    Returns
    -------
    theta, phi : (N,) ndarray
        Quadrature point coordinates (inclination and azimuth).
    w : (N,) ndarray
        Quadrature weights.

    """
    basedir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           '../data')
    quad_file = {72: 'qsph1-14-72DP.dat',
                 132: 'qsph1-16-132DP.dat',
                 492: 'qsph1-37-492DP.dat'}

    q_pts = np.loadtxt(os.path.join(basedir, quad_file[N]))
    q_theta, q_phi, _ = coord.car2sph(*q_pts[:, :3].T)
    q_w = q_pts[:, 3]

    return q_theta, q_phi, q_w


def cos_inc_angle(theta1, phi1, theta2, phi2):
    """
    Compute the cosine of the included angle between two vectors on the
    unit sphere, specified in spherical coordinates.

    Parameters
    ----------
    theta1, phi1 : float or ndarray
        First vector.
    theta2, phi2 : float or ndarray
        Second vector.

    Notes
    -----
    This is derived by taking the dot-product of the two vectors in Cartesian
    coordinates, then somewhat simplifying using trigonometric identities.

    More than one angle can be computed simultaneously, as long as theta1 and
    theta2, as well as phi1 and phi2 broadcast.

    """
    return np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2) \
           + np.cos(theta1) * np.cos(theta2)

def mesh(npts=(101, 101), closed=False):
    """Generate a meshgrid on the unit sphere.

    Uniformly sample the polar angle, theta, and the azimuthal angle, phi.

    Parameters
    ----------
    npts : int or tuple
        Number of angle points sampled.
    closed : bool
        Whether to generate an open mesh (like `np.ogrid`)
        or a closed mesh like `np.mgrid` or `np.meshgrid`.
        By default, an open grid is generated.

    Returns
    -------
    theta, phi : (N,) or (N,M) ndarray
        Sampling of the polar angle.  Shape depends on ``open``.

    """
    if np.isscalar(npts):
        npts = (npts, npts)

    theta = np.linspace(0, np.pi, npts[0])[:, None]
    phi = np.linspace(0, 2 * np.pi, npts[1])

    if open:
        return theta, phi
    else:
        mg_phi, mg_theta = np.meshgrid(phi_grid, theta_grid)
        return mg_theta, mg_phi
