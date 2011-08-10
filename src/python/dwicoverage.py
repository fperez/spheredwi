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

#-----------------------------------------------------------------------------
# Functions and classes 
#-----------------------------------------------------------------------------

def where2(cond):
    """Return both where cond is true and where it's false.

    Note: the return value is a single index array, NOT a 1-element tuple (in
    contrast to np.where), so this function isn't useful for multidimensional
    inputs.

    Parameters
    ----------
    cond : 1-d array
    """
    return cond.nonzero()[0], (~cond).nonzero()[0]


def cart2spherical(rcart,normalized=False):
    """Convert array of *unit* 3-vectors in cartesian coordinates to spherical.

    This computes the spherical coordinates (r, theta, phi) for the input
    array, where theta is the polar angle and phi is the azimuthal one.

    Parameters
    ----------
    rcart : 3xn or nx3 array
      Array of points in R^3 to be converted.

    normalized : boolean, optional (False)
      If true, assume the input vectors are all normalized.  This avoids
      computing the norms internally.  In this case, the r vector (see below)
      will NOT be returned.

    Returns
    -------
    r, theta, phi : 1-d arrays
      The spherical coordinates of the input arrays.
    """
    # Tolerance: points with x-coordinate below this are considered to lie
    # precisely on the y-z plane, to avoid issues of numerical error when
    # taking arctangents.
    eps = 1e-6

    # Validate dimensionality of input
    rs = rcart.shape
    if rcart.ndim != 2 or not (rs[0]==3 or rs[1]==3):
        raise ValueError("input must be shaped 3xN or Nx3, shape given: %s" % rs)
    if rs[1] == 3:
        rcart = rcart.T

    # Compute r and extract normalized x/y/z components (used for inversion
    # calculations)
    if normalized:
        x,y,z = rcart
    else:
        r = sqrt((rcart**2).sum(0))
        x,y,z = rcart/r
    
    # Theta, the latitude angle, is easy
    theta = arccos(z)

    # Getting phi (longitude angle) right is tricky...
    yz,xyz    = where2(abs(x)<eps)
    xpos,xneg = where2(x>0)
    ypos,yneg = where2(y>0)

    phi = arctan(y/x)
    phi[xneg] += pi
    phi[np.intersect1d(xpos,yneg)] += 2*pi
    phi[np.intersect1d(yz,ypos)] = pi/2
    phi[np.intersect1d(yz,yneg)] = 3*pi/2

    if normalized:
        return theta, phi
    else:
        return r, theta, phi


def cart2lonlat(rcart, normalized=False):
    """Convert array of 3-vectors in cartesian coordinates to longitude/latitude.

    Parameters
    ----------
    rcart : 3xn array
      Array of points in R^3 to be converted.

    normalized : boolean, optional (False)
      If true, assume the input vectors are all normalized.  See cart2spherical
      for details.

    Returns
    -------
    lon : 1d array
      Longitude in degrees,  in the [-180,180] range.
    lat : 1d array
      Latitude in degrees, in the [-90,90] range.
    """
    # We don't need the radial coordinates here, only the angles
    theta, phi = cart2spherical(rcart,normalized)[1:]
    
    # Convert phi/theta to longitude/latitude
    lon = (phi-pi)*(180/pi)
    lat = (theta-pi/2)*(180/pi)
    return lon, lat


def sphere_mesh(r=1.0, npts=(101,101)):
    """Create mesh for a sphere.

    Uniformly sample the spherical angles theta, phi (in physics convention,
    where theta is the polar angle and phi is the azimuthal one).

    Parameters
    ----------

    r : float
      radius of sphere

    npts : pair of floats
      The number of points for the theta and phi grids."""
    
    np_theta = npts[0]*1j
    np_phi = npts[1]*1j
    theta, phi = np.mgrid[0:pi:np_theta,0:2*pi:np_phi]
    x = r*sin(theta)*cos(phi)
    y = r*sin(theta)*sin(phi)
    z = r*cos(theta)
    return x,y,z


def vec_norms(vecs):
    """Return the L2 norms of an array of vectors.

    The input 2-d array is interpreted as a list of column vectors.
    """
    return (vecs**2).sum(0)


def symmetrize(v1,v2):
    """Symmetrize two lists of input vectors in R^3.
    """
    return np.hstack([v1,-v1]), np.hstack([v2,-v2])        


def dwi_coverage(rp,rs,ndirs=None):
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
    sigma2 = 2.0/(ndirs*log(2))
    # Precompute the actual factor that goes into the exponent of the gaussians
    # just once.
    sfac = -1.0/(2*sigma2)

    # Unpack the mesh coordinates to construct the field at these points
    xs,ys,zs = rs
    field = np.zeros_like(xs)
    # Actual field computation
    for xp,yp,zp in rp.T:
        arc = arccos(xs*xp+ys*yp+zs*zp)
        field += exp(sfac*arc**2)
    # Since the units are completely arbitrary, return a normalized value
    return field/field.max()


def sphere_field(rp,ndirs,npts=(100,100),**kw):
    """Compute coverage field on a sphere."""
    sphere = sphere_mesh(1.0,npts)
    # compute scalar field from point distribution xp,yp,zp at xs,ys,zs on the
    # sphere mesh
    field = dwi_coverage(rp,sphere,ndirs)
    #print 'fshape:',field.shape # dbg
    return sphere, field


def mscatter(bmap,ax,pts,color, **kwargs):
    """Scatter plot on a basemap.

    This is mostly just a utility wrapper around bmap.scatter.

    Parameters
    ----------

    bmap : basemap instance
    ax : matplotlib axes instance
      This should be the axes associated with map drawing from bmap, since
      ultimately ax.scatter() will be called.
    pts :   

    """
    rlon, rlat = cart2lonlat(pts)
    rx, ry = bmap(rlon, rlat)
    ax.scatter(rx, ry, c=color, edgecolors='none', **kwargs)
    

def show_coverage_2d(rg, sphere, field, rm=None,
                     sphere_colormap='jet', vmin=None,
                     ax=None, markersize=30):
    """Show field coverage by projecting the sphere onto a flat 2d surface.

    Parameters
    ----------
    rg :

    ax : matplotlib axis object, None
    
    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap

    if ax is None:
        plt.figure()

    projection='moll'
    
    nlevels = 56
    npts_lat,npts_lon = field.shape
    lats = np.linspace(-90,90,npts_lat)
    lons = np.linspace(-180,180,npts_lon)
    m = Basemap(projection=projection,lat_0=0,lon_0=0,resolution='c',ax=ax)
    x, y = m(*np.meshgrid(lons, lats))
    if vmin is None:
        # Use the minimum value in the data for the color scale
        c = m.contourf(x,y,field,nlevels,cmap=plt.cm.get_cmap(sphere_colormap))
        if ax is None:
            plt.colorbar(orientation='horizontal',format='%.2g')
    else:
        clevels = np.linspace(vmin,1,nlevels)
        cticks = ["%2g" % l for l in clevels]
        c = m.contourf(x,y,field,clevels,cmap=plt.cm.get_cmap(sphere_colormap))
        if ax is None:
            plt.colorbar(orientation='horizontal',ticks=[0,0.25,0.5,0.75,1])
    # Now, compute the lat/lon for the points
    mscatter(m,c.ax,rg,'g', s=markersize)
    if rm is not None and rm.shape[1] > 0:
        mscatter(m,c.ax,rm,'r', s=markersize)

    if ax is None:
        plt.show()


def show_coverage_3d(rg,sphere,field,rm=None,vmin=None,sphere_colormap='jet'):
    """
    """
    from enthought.mayavi import mlab

    # Resolution with which the small spheres are drawn
    pt_resolution = 20
    # Size factor for small spheres
    scale_factor = 0.1
    
    # missing directions in red, if any
    if rm is not None:
        xm,ym,zm = rm
        mlab.points3d(xm,ym,zm,color=(1,0,0),scale_factor=scale_factor,
                      resolution=pt_resolution,
                      name='Bad directions')
    
    # 'good' directions in green
    xg,yg,zg = rg

    # As simple spheres
    mlab.points3d(xg,yg,zg,color=(0,1,0),scale_factor=scale_factor,
                  resolution=pt_resolution,
                  name='Good directions')
    
    xs,ys,zs = sphere  # unpack mesh coordinates for mlab.mesh call
    s=mlab.mesh(xs, ys, zs, scalars=field,vmin=vmin,
              colormap=sphere_colormap,name='Coverage')
    mlab.colorbar(s,orientation='horizontal')
    mlab.show()


def load_bvecs(fname='bvecs'):
    """Load a set of bvecs, making a few sanity checks along the way.

    This loads a text file containing unit vectors in R^3, with (by default) 3
    rows and N columns for N vectors.  A warning is emitted if any of the input
    vectors has a norm below 0.99.

    Parameters
    ----------
      fname : string or file

      The input file should contain a 3xn array.  If not, the array is
      transposed and it's accepted if the result is then 3xn, otherwise a
      ValueError is raised.  This means that if the input is 3x3, the rows are
      interpreted as (x,y,z) and the columns as (p1,p2,p3) for the three
      points.

    Returns
    -------
      pts : a 2-d array of dimensions (3,N)
      """

    # If vectors are found with norms less than this, sound the alarm
    norm_tolerance = 0.99
    # Load the file declaring the point set
    pts = np.loadtxt(fname)
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
        File name of file containing a 3xn array of unit vectors,
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
        tmp = open(fname_miss).read()
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
        bv_good,bv_miss = symmetrize(bv_good,bv_miss)
    # Find the total number of bvecs.  We do this now rather than taking
    # len(bvecs) to properly account for the possibly dropped B0 term.  
    nvecs = bv_good.shape[1] + bv_miss.shape[1]
        
    # Now compute and draw the coverage field generated by the 'good'
    # directions
    sphere, field = sphere_field(bv_good,nvecs)

    return bv_good,sphere,field,bv_miss


# Script-like entry point
def main_coverage(fpoints, fpoints_missing=None, symm=False,
                  vmin=None, show_3d=False):
    bv_good,sphere,field,bv_miss = build_coverage(fpoints, fpoints_missing,
                                                  symm=symm)
    show_coverage_2d(bv_good, sphere, field, bv_miss, vmin=vmin)
    if show_3d:
        show_coverage_3d(bv_good, sphere, field, bv_miss, vmin=vmin)


if __name__ == '__main__':
    main_coverage('bvecs', 'bvecs_missing', True)
