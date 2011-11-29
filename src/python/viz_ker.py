"""Visualize inverse Funk-Radon transform elements with mayavi.
"""
from enthought.mayavi import mlab
import numpy as np

from sphdif import sphquad

# Create a spherical mesh
npts = 101
r    = 1.0
pi   = np.pi
cos  = np.cos
sin  = np.sin
theta, phi = np.mgrid[0:pi:npts*1j, 0:2*pi:npts*1j]

x = r*sin(theta)*cos(phi)
y = r*sin(theta)*sin(phi)
z = r*cos(theta)

mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 400))

# Sample the inverse kernel over one meridian and replicate across longitudes
# to get a spherical mesh.
nmax = 14  #order of kernel
mu = cos(theta[:,0])  # just one meridian

#Old kernel
#k = np.array([sphquad.inv_funk_radon_kernel(mm, nmax) for mm in mu])

#New kernel
#k = np.array([sphquad.inv_funk_radon_even_kernel(mm,nmax) for mm in mu])

#Even reproducing kernel
k = np.array([sphquad.even_kernel(mm,nmax) for mm in mu])

#Reproducing kernel for whole subspace
#k = np.array([sphquad.kernel(mm,nmax) for mm in mu])


k = k[:, None]        # fix dimensions for mesh broadcasting
s = np.tile(k, npts)  # mayavi expects the scalars value to really have the
                      # same shape as the mesh, so we have to tile
                      
mlab.mesh(k*x, k*y, k*z, scalars=s, colormap='jet')

mlab.show()
