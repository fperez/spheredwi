import numpy as np


def car2sph(x, y, z):
    """Convert Cartesian to spherical coordinates.

    Parameters
    ----------
    x, y, z : float or array_like
        Cartesian coordinates.

    Returns
    -------
    r, theta, phi : float or array_like
        Radius, inclination and azimuth angles.

    Notes
    -----
    If you have a 3-column array with coordinate values, call
    ``car2sph(*X.T)``.

    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    return r, theta, phi


def sph2car(r, theta, phi):
    """Convert Cartesian to spherical coordinates.

    Parameters
    ----------
    r, theta, phi : float or array_like
        Radius, inclination and azimuth angles.

    """
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return x, y, z


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

