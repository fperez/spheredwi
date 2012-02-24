import numpy as np
import coord

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

    from mpl_toolkits.basemap import Basemap

    m = Basemap(**basemap_args)
    m.drawmapboundary()
    lat, lon = coord.sph2latlon(theta, phi)
    x, y = m(*np.meshgrid(lon, lat))
    m.pcolor(x, y, r, vmin=vmin, vmax=vmax);

    return m

def surf_grid_3D(r, theta, phi, scale_radius=False):
    """Draw a function r = f(theta, phi), evaluated on a grid, on the sphere.

    Parameters
    ----------
    r : (M, N) ndarray
        Function values.
    theta : (M,) ndarray
        Inclination / polar angles of function values.
    phi : (N,) ndarray
        Azimuth angles of function values.
    scale_radius : bool
        Whether to scale the radius with the function value (changes the
        surface height to reflect function values).

    """
    try:
        from enthought.mayavi import mlab
    except ImportError:
        from mayavi import mlab

    phi, theta = np.meshgrid(phi, theta)
    if scale_radius:
        x, y, z = coord.sph2car(r, theta, phi)
    else:
        x, y, z = coord.sph2car(np.ones_like(theta), theta, phi)

    mlab.mesh(x, y, z, scalars=r)

def scatter(theta, phi, basemap=None, **scatter_args):
    if basemap is None:
        z = np.array([0, 0])
        basemap = surf_grid(z, z, z, projection='moll')

    lat, lon = coord.sph2latlon(theta, phi)
    x, y = basemap(lon, lat)
    basemap.scatter(x, y, **scatter_args)

def scatter_3D(theta, phi, scalar=None, **points3d_args):
    points3d_args.setdefault('scale_factor', 0.1)
    points3d_args.setdefault('color', (1, 0, 0))

    try:
        from enthought.mayavi import mlab
    except ImportError:
        from mayavi import mlab

    if scalar is None:
        scalar = np.ones_like(theta)

    x, y, z = coord.sph2car(np.ones_like(theta), theta, phi)
    mlab.points3d(x, y, z, scalar, **points3d_args)

def show():
    try:
        from enthought.mayavi import mlab
    except ImportError:
        from mayavi import mlab

    mlab.show()
