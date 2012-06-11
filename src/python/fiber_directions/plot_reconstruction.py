import sys
sys.path.insert(0, '..')

import numpy as np

from sphdif import sph_io
from sphdif import plot
from sphdif.kernel import (kernel_matrix, std_kernel, even_kernel,
                           inv_funk_radon_even_kernel, kernel_reconstruct)

data = sph_io.load('odf_coeffs.npz')

# unpack beta, theta, phi, separation
#        beta_dense, theta_dense, phi_dense
this_module = sys.modules[__name__]
for k, v in data.items():
    setattr(this_module, k, v)

def plot_reconstruction(beta, phi, theta, kernel, kernel_order,
                        grid_density=100):
    theta_grid = np.linspace(0, np.pi, grid_density)
    phi_grid = np.linspace(0, 2 * np.pi, grid_density)

    ODF = kernel_reconstruct(theta, phi, beta,
                             theta_grid, phi_grid,
                             kernel=kernel, N=kernel_order)

#    ODF[ODF < 0] = 0

    plot.surf_grid_3D(ODF, theta_grid, phi_grid, scale_radius=True)


mlab = plot.get_mlab()

plot_reconstruction(beta, phi, theta, even_kernel, 9)
mlab.show()

plot_reconstruction(beta_dense, phi_dense, theta_dense,
                    even_kernel, 12)
mlab.show()

