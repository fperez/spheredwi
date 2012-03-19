"""Plot DWI coverage on the unit sphere.

XXX
- From the system cmd line, is it possible to open both the mpl and the mv
windows for interactive use?
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib
import warnings
from cStringIO import StringIO

# Third-party
import numpy as np
from numpy import pi, sin, cos, arccos,  arctan, exp, log, sqrt
import os

from sphdif import coord, sphere, plot

data_path = os.path.join(os.path.dirname(__file__), './data')

#-----------------------------------------------------------------------------
# Functions and classes
#-----------------------------------------------------------------------------

def vec_norms(vecs):
    """Return the L2 norms of an array of vectors.

    The input 2-d array is interpreted as a list of column vectors.
    """
    return (vecs**2).sum(0)


def symmetrize(v1,v2):
    """Symmetrize two lists of input vectors in R^3.
    """
    return np.hstack([v1,-v1]), np.hstack([v2,-v2])


def dwi_coverage(rp, rs, ndirs=None):
    """Compute coverage from points at rp on spherical mesh at rs.
    """
    # If the number of directions isn't given, we assume the total would be
    # sufficient for good coverage.
    if ndirs is None:
        ndirs = rp.shape[1]
    # Compute sigma**2 assuming that ndirs would provide 'even' coverage of the
    # full 4pi solid angle.  Here, we interpret 'even coverage' by treating
    # each gaussian as providing good coverage for the disk within its FWHM, so
    # we simply take 4pi/(ndirs*area_fwhm). With fwhm = 2*sqrt(2 ln2)*sigma, we
    # then get:
    sigma2 = 2.0 / (ndirs * log(2))
    # Precompute the actual factor that goes into the exponent of the gaussians
    # just once.
    sfac = -1.0 / (2 * sigma2)

    # Unpack the mesh coordinates to construct the field at these points
    xs, ys, zs = np.rollaxis(rs, -1)
    field = np.zeros_like(xs)
    # Actual field computation
    for xp, yp, zp in rp.T:
        arc = arccos(xs * xp + ys * yp + zs * zp)
        field += exp(sfac*arc**2)

    # Since the units are completely arbitrary, return a normalized value
    return field / field.max()


def show_coverage_2d(rg, field, rm=None, projection='moll',
                     vmin=None, markersize=30):
    """Show field coverage by projecting the sphere onto a flat 2d surface.

    Parameters
    ----------
    rg : (M, 3) ndarray
        Gradient directions in Cartesian coordinates.
    field : (M, N) ndarray
        Field values on a grid.  The grid is on ``theta = [0, pi]`` and
        ``phi = [0, 2 * pi]``.
    rm : (M, 3) ndarray
        Missing / dropped gradient directions.
    projection : 'moll', 'ortho', etc.
        Basemap projection.

    """
    import matplotlib.pyplot as plt

    npts_lat, npts_lon = field.shape
    theta = np.linspace(0, np.pi, npts_lat)
    phi = np.linspace(0, 2 * np.pi, npts_lon)

    m = plot.surf_grid(field, theta, phi,
                       projection=projection, vmin=vmin)
    plt.colorbar(orientation='horizontal', format='%.2g')

    x, y, z = rg
    theta, phi, rho = coord.car2sph(x, y, z)
    plot.scatter(theta, phi, basemap=m, color='g', s=markersize)

    if rm is not None and rm.shape[1] > 0:
        x, y, z = rm
        theta, phi, rho = coord.car2sph(x, y, z)
        plot.scatter(theta, phi, basemap=m, color='r', s=markersize)

    plt.show()


def show_coverage_3d(rg, field, rm=None, vmin=None, sphere_colormap='jet'):
    """
    """
    # Resolution with which the small spheres are drawn
    pt_resolution = 20
    # Size factor for small spheres
    scale_factor = 0.1

    # missing directions in red, if any
    if rm is not None:
        xm, ym, zm = rm
        theta, phi, rho = coord.car2sph(xm, ym, zm)
        plot.scatter_3D(theta, phi, resolution=pt_resolution,
                        scale_factor=scale_factor, name='Bad direction')

    # 'good' directions in green
    xg, yg, zg = rg
    theta, phi, rho = coord.car2sph(xg, yg, zg)

    # As simple spheres
    plot.scatter_3D(theta, phi, color=(0, 1, 0), scale_factor=scale_factor,
                    resolution=pt_resolution, name='Good directions')

    npts_lat, npts_lon = field.shape
    theta = np.linspace(0, np.pi, npts_lat)
    phi = np.linspace(0, 2 * np.pi, npts_lon)
    s = plot.surf_grid_3D(field, theta, phi, vmin=vmin, name='Coverage')

    mlab = plot.get_mlab()
    mlab.colorbar(s, orientation='horizontal')
    mlab.show()


def load_bvecs(fname='bvecs'):
    """Load a set of bvecs, making a few sanity checks along the way.

    This loads a text file containing unit vectors in R^3, with (by default) 3
    rows and N columns for N vectors.  A warning is emitted if any of the input
    vectors has a norm below 0.99.

    Parameters
    ----------
    fname : string or file
        Name of file in data directory.

    Notes
    -----
    The input file should contain a 3xN array.  If not, the array is
    transposed and it's accepted if the result is then 3xN, otherwise a
    ValueError is raised.  This means that if the input is 3x3, the rows are
    interpreted as (x,y,z) and the columns as (p1,p2,p3) for the three
    points.

    Returns
    -------
    pts : (3, N) ndarray
        Data points.

    """
    # If vectors are found with norms less than this, sound the alarm
    norm_tolerance = 0.99
    # Load the file declaring the point set
    pts = np.loadtxt(os.path.join(data_path, fname))
    # Sanity check
    if pts.ndim != 2:
        raise ValueError("Input file must be a 2-d array, %s-dims given" %
                         pts.ndim)
    if pts.shape[0] != 3:
        # Try to transpose it in case it was given as nx3 or nx4 (we'll try to
        # throw away the remaining columns
        pts = pts.T[:3]
        if pts.shape[0] != 3:
            raise ValueError("Input vectors must be 3-d, %s-dims given" %
                             pts.shape[0] )
        else:
            pts = pts.copy()  # ensure contiguity

    norms = vec_norms(pts)

    # Warn users if vectors aren't well normalized, except for possibly the B0
    # component which, if in the file, is always 0
    small = np.where(norms < norm_tolerance)
    if len(small)>1:
        e = """\
The following directions have norm below %.2g:
Indices: %s
Norms  : %s""" % (norm_tolerance, small, norms[small])
        warnings.warn(e)

    return pts


def build_coverage(points, fname_miss=None, symm=True):
    """Compute surface coverage for a set of directions.

    Parameters
    ----------
    points : string or ndarray
        File name of file containing a 3xN array of unit vectors,
        or ndarray with vectors.
    """

    # Norm threshold below which we simply drop vectors assuming they were
    # noise or zeros (such as B0 term in DWI data)
    drop_norm = 1e-8

    # Load vectors from disk

    if hasattr(points, '__array_interface__'):
        bvecs = points
    else:
        bvecs = load_bvecs(points)

    all_idx = np.arange(bvecs.shape[1])
    if fname_miss is None:
        # If no missing directions are given, make an empty 3x0 array so that
        # later we don't need to constantly special-case code for None
        bv_good = bvecs
        bv_miss = np.empty((3,0))
    else:
        # Preprocess the input to read it as a 1-d list while skipping comment
        # lines.
        tmp = open(os.path.join(data_path, fname_miss)).read()
        bad_raw = StringIO(' '.join(l for l in tmp.splitlines()
                                    if not l.strip().startswith('#')))
        bad_idx = np.loadtxt(bad_raw,int)
        if bad_idx.ndim==0:
            # Corner case: if the input file contains a single number, loadtxt
            # returns an array scalar instead of a 1-element 1-d array.  We
            # need to fix that so code further down doesn't explode
            bad_idx.shape = (1,)
        # split the full list between good/bad directions
        good_idx = np.setdiff1d(all_idx,bad_idx)
        bv_good = bvecs[:,good_idx]
        bv_miss  = bvecs[:,bad_idx]

    # The first vector is typically 0, so we drop it if present
    if vec_norms(bv_good)[0] < drop_norm:
        bv_good = bv_good[:,1:]

    # The full bvecs lists contains every vector and its opposite direction
    if symm:
        bv_good, bv_miss = symmetrize(bv_good, bv_miss)
    # Find the total number of bvecs.  We do this now rather than taking
    # len(bvecs) to properly account for the possibly dropped B0 term.
    nvecs = bv_good.shape[1] + bv_miss.shape[1]

    # Now compute and draw the coverage field generated by the 'good'
    # directions

    # compute scalar field from point distribution bv_good on the
    # sphere mesh
    theta, phi = sphere.mesh(closed=True)
    xyz = np.dstack(coord.sph2car(theta, phi))
    field = dwi_coverage(bv_good, xyz, nvecs)

    return bv_good,sphere,field,bv_miss


# Script-like entry point
def main_coverage(fpoints, fpoints_missing=None, symm=False,
                  vmin=None, show_3d=False):
    bv_good,sphere,field,bv_miss = build_coverage(fpoints, fpoints_missing,
                                                  symm=symm)
    show_coverage_2d(bv_good, field, bv_miss, vmin=vmin)
    if show_3d:
        show_coverage_3d(bv_good, field, bv_miss, vmin=vmin)


if __name__ == '__main__':
    main_coverage('bvecs', fpoints_missing='bvecs_missing',
                  symm=True, vmin=None, show_3d=True)
