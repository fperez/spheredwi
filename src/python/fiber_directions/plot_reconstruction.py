import sys
sys.path.insert(0, '..')

import numpy as np
import glob
import os

from sphdif import sph_io
from sphdif import plot
from sphdif.kernel import (kernel_matrix, std_kernel, even_kernel,
                           inv_funk_radon_even_kernel, kernel_reconstruct)
from sphdif.coord import car2sph


def plot_reconstruction(beta, phi, theta, kernel, kernel_order,
                        grid_density=100):
    theta_grid = np.linspace(0, np.pi, grid_density)
    phi_grid = np.linspace(0, 2 * np.pi, grid_density)

    ODF = -kernel_reconstruct(theta, phi, beta,
                              theta_grid, phi_grid,
                              kernel=kernel, N=kernel_order)


coords = sph_io.load('sphere_pts.npz')

#from dipy.data import get_sphere
#verts, faces = get_sphere('symmetric724')

from dipy.core.triangle_subdivide import create_unit_sphere
verts, edges, sides = create_unit_sphere(6)
faces = edges[sides, 0]

theta, phi, r = car2sph(*verts.T)

ODFs = []
for k, fn in enumerate(sorted(glob.glob(os.path.join(sph_io.data_path,
                                                     'odf_coeffs_*.npz')))):
    data = sph_io.load(os.path.basename(fn))
    odf = kernel_reconstruct(coords['theta'], coords['phi'], data['beta'],
                             theta, phi,
                             kernel=even_kernel, N=data['kernel_N'])
    odf[odf < 0] = 0
    ODFs.append(odf)

    #    ODF[ODF < 0] = 0
    #plot.surf_grid_3D(-ODF, theta_grid, phi_grid, scale_radius=True)


## mlab = plot.get_mlab()
## plot_reconstruction(beta, phi, theta, even_kernel, kernel_N)
## mlab.show()

ODFs = np.array([[ODFs]]).reshape(len(ODFs), 1, 1, -1)

from dipy.viz import show_odfs
show_odfs(ODFs, (verts, faces), scale=2, radial_scale=True)

