from __future__ import division
import sys
sys.path.insert(0, '..')

import numpy as np
eps = np.finfo(float).eps

from sphdif import sph_io, sphere, plot

data = sph_io.load('odf_coeffs.npz')

# unpack beta, theta, phi, separation
this_module = sys.modules[__name__]
for k, v in data.items():
    setattr(this_module, k, v)

# Initialize centroids
# TODO: Random intialization
N = 4
c_theta = np.pi * np.array([-1/3, 1/3, 2/3, -2/3])
#c_phi = 2 * np.pi * np.array([0, 1/4, 2/4, 3/4])
c_phi = np.ones(N)

def weighted_kmeans(c_theta, c_phi, p_theta, p_phi, v):
    D = 1 / (eps + sphere.arc_length(c_theta, c_phi,
                                     p_theta[:, None], p_phi[:, None]))**2
#    D = np.exp(-sphere.arc_length(c_theta, c_phi, p_theta[:, None], p_phi[:, None])**2)
    D *= v[:, None]
    D /= D.sum(axis=0)

    P = np.column_stack((p_theta, p_phi))

    C = (D[:, :, None] * P[:, None, :]).sum(axis=0)

    return C[:, 0], C[:, 1]

for i in range(50):
    c_theta, c_phi = weighted_kmeans(c_theta, c_phi, theta, phi, beta)

angles = np.rad2deg(np.arccos(sphere.cos_inc_angle(
    c_theta, c_phi, c_theta[:, None], c_phi[:, None])))
angles[angles < 1e-3] = 0

print "True separation:", separation
print "Calculated:", angles

mask = (beta != 0)
plot.scatter_3D(theta, phi, color=(0, 0, 1))
plot.scatter_3D(theta[mask], phi[mask], 1 + beta[mask]/beta.max(),
                transparent=True, color=(1, 0, 0), scale_mode='scalar',
                scale_factor=0.1, opacity=0.7)
plot.scatter_3D(c_theta, c_phi, color=(1, 1, 0))
plot.show()
