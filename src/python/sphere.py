import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import os

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

def car2sph(x, y, z):
    """Convert Cartesian to spherical coordinates.

    Parameters
    ----------
    x, y, z : float or array_like
        Cartesian coordinates.

    Notes
    -----
    If you have a 3-column array with coordinate values, call
    ``car2sph(*X.T)``.

    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    return r, theta, phi

def sph2latlon(theta, phi):
    """Convert spherical coordinates to latitude and longitude.

    Returns
    -------
    lat, lon : ndarray
        Latitude and longitude.

    """
    return np.rad2deg(theta - np.pi/2), np.rad2deg(phi - np.pi)

def latlon2sph(lat, lon):
    """Convert latitude / longitude to spherical coordinates.

    Returns
    -------
    theta, phi : ndarray
        Inclination and azimuth angles.

    """
    return np.deg2rad(lat) + np.pi/2, np.deg2rad(lon) + np.pi


def surf_grid(r, theta, phi, ax=None, vmin=None, vmax=None, **basemap_args):
    """Draw a function r = f(theta, phi), evaluated on a grid, on the sphere.

    Parameters
    ----------
    r : (M, N) ndarray
        Function values.
    theta : (M,) ndarray
        Inclination / polar angles of function values.
    phi : (N,) ndarray
        Azimuth angles of function values.
    ax : mpl axis, optional
        If specified, draw onto this existing axis instead.
    basemap_args : dict
        Parameters used to initialise the basemap, e.g. ``projection='ortho'``.

    Returns
    -------
    m : basemap
        The newly created matplotlib basemap.

    """
    basemap_args.setdefault('projection', 'ortho')
    basemap_args.setdefault('lat_0', 0)
    basemap_args.setdefault('lon_0', 0)
    basemap_args.setdefault('resolution', 'c')
    basemap_args.setdefault('ax', ax)

    m = Basemap(**basemap_args)
    m.drawmapboundary()
    lat, lon = sph2latlon(theta, phi)
    x, y = m(*np.meshgrid(lon, lat))
    m.pcolor(x, y, r, vmin=vmin, vmax=vmax);

    return m

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
                           './data')
    quad_file = {72: 'qsph1-14-72DP.dat',
                 132: 'qsph1-16-132DP.dat',
                 492: 'qsph1-37-492DP.dat'}

    q_pts = np.loadtxt(os.path.join(basedir, quad_file[N]))
    _, q_theta, q_phi = car2sph(*q_pts[:, :3].T)
    q_w = q_pts[:, 3]

    return q_theta, q_phi, q_w

def cos_inc_angle(theta1, phi1, theta2, phi2):
    """
    Compute the cosine of the included angle between two vectors specified in
    spherical coordinates.

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

def scatter(theta, phi, basemap=None, **scatter_args):
    if basemap is None:
        z = np.array([0, 0])
        basemap = surf_grid(z, z, z, projection='moll')

    lat, lon = sph2latlon(theta, phi)
    x, y = basemap(lon, lat)
    basemap.scatter(x, y, **scatter_args)
