from __future__ import division
import sys
sys.path.insert(0, '..')

import numpy as np
import scipy as sp
import scipy.special

import matplotlib.pyplot as plt

from sphdif import sphere, coord, plot
from sphdif.kernel import (kernel_matrix, std_kernel, even_kernel,
                           inv_funk_radon_even_kernel, kernel_reconstruct)
from sphdif.linalg import rotation_around_axis
from sphdif.signal_sim import single_tensor, single_tensor_ODF
from sphdif import sph_io


visualize_odf = False
visualize_signal = True


def plot_ODF(grid_density=100):
    theta_grid = np.linspace(0, np.pi, grid_density)
    phi_grid = np.linspace(0, 2 * np.pi, grid_density)

    phi_vec, theta_vec = np.meshgrid(phi_grid, theta_grid)
    phi_vec, theta_vec = phi_vec.ravel(), theta_vec.ravel()

    xyz = np.column_stack(coord.sph2car(theta_vec, phi_vec))

    ODF = w[0] * single_tensor_ODF(xyz, rotation=R0)
    ODF += w[1] * single_tensor_ODF(xyz, rotation=R1)

    ODF = ODF.reshape((grid_density, grid_density))

    plot.surf_grid_3D(ODF, theta_grid, phi_grid, scale_radius=True)


def L(E, d1=0.001, d2=0.001):
    """Apply flexible threshold to measurement data
    in order to stay away from dangerous zones 0 and 1 as
    shown in Aganj.

    Then return ln(-ln(E)).

    """
    m1 = (E >= 0) & (E < d1)
    m2 = (E > 1 - d2) & (E <= 1)

    E[m1] = d1/2 + E[m1]**2 / (2 * d1)
    E[m2] = 1 - d2/2 - (1 - E[m2])**2 / (2 * d2)

    E[E < 0] = d1/2
    E[E > 1] = 1 - d2/2

    return np.log(-np.log(E))


# ======================
# Load quadrature points
# ======================

theta72, phi72, w72 = sphere.quadrature_points(N=72)
theta132, phi132, w132 = sphere.quadrature_points(N=132)
#theta492, phi492, w492 = sphere.quadrature_points(N=492)


theta, phi = theta72, phi72

from dipy.data import get_data
img, bvals, bvecs = get_data('small_64D')

b = np.load(bvals)
bvecs = np.load(bvecs)
bvecs = bvecs[b > 0]
b = b[b > 0]

theta, phi, _ = coord.car2sph(*bvecs.T)
theta_odf, phi_odf = theta132, phi132

kernel = inv_funk_radon_even_kernel
kernel_N = 8
X = kernel_matrix(theta, phi, theta_odf, phi_odf, kernel=kernel, N=kernel_N)
D = 150 # Grid density for plots (higher => more dense)

b = 3000 + np.random.normal(scale=4, size=len(theta)) # Make up somewhat realistic b-values

xyz = np.column_stack(coord.sph2car(theta, phi))

sph_io.savez('sphere_pts', gradient_theta=theta, gradient_phi=phi,
                           odf_theta=theta_odf, odf_phi=phi_odf, b=b)

# Fiber weights
w = [0.5, 0.5]

angles = np.deg2rad(np.arange(40, 60, 5))
angles = np.insert(angles, 0, 0)

SNR = 30

for k, gamma in enumerate(angles):
    print "Angle:", np.rad2deg(gamma)

    # Gamma is the angle separating fibres

    # ================
    # Set up test data
    # ================

    # b is tau * |q|^2 in s/mm^2
    # If b is too low, the signal does not attenuate enough to measure.
    # Too high, the signal to noise ratio increases.

    angles = np.array([gamma / 2., -gamma / 2])
    R0 = rotation_around_axis([0, 1, 0], 0)
    R1 = rotation_around_axis([0, 1, 0], angles[0] - angles[1])

    # ================
    # Q-Space Sampling
    # ================

    # Note: We sample our signal in Q-space on the low-order quadrature points
    # here, but wwe could just as well have used other, random points on the
    # sphere.

    E = w[0] * single_tensor(gradients=xyz, bvals=b, S0=1, rotation=R0, SNR=SNR)
    E += w[1] * single_tensor(gradients=xyz, bvals=b, S0=1, rotation=R1, SNR=SNR)

    print "Signal mean:", E.mean()

    if visualize_signal:
        from dipy.core.triangle_subdivide import create_unit_sphere
        verts, edges, sides = create_unit_sphere(6)
        faces = edges[sides, 0]
        bb = np.ones(len(verts)) * b.mean()

        R_ = rotation_around_axis([0, 1, 0], gamma)

        E_ = w[0] * single_tensor(gradients=verts, bvals=bb, S0=1, rotation=R0, SNR=SNR)
        E_ += w[1] * single_tensor(gradients=verts, bvals=bb, S0=1, rotation=R_, SNR=SNR)

        from dipy.viz import show_odfs
        show_odfs([[[E_, -L(E_)]]], (verts, faces))
        


    if visualize_odf:
        plot_ODF(grid_density=D)
        mlab = plot.get_mlab()
        mlab.show()

    # ===========================-=====
    # ODF-domain: Sparse reconstruction
    # =================================

    # Minimising the L1 penalized system
    #
    # ||Xb - y||_2^2 + lambda ||x||_1 subject to x_i >= 0.
    #
    # Here, X is reproducing Q-space kernel, y is the Q-space signal vector
    # and b are the coefficents.
    #
    # Note that the L1 penalization of b forces it to be sparse.  This also implies
    # that the ODF-domain signal is sparse, since the kernels in A (the reproducing
    # kernel matrix in ODF-space) are localised, thus the product Ab is sparse.
    # The product Xb is *not* sparse--the kernels in F are donut shaped, and not
    # localized.

    # See the the low-rank approximation wiki for more detail on these types of
    # problems: http://ugcs.caltech.edu/~srbecker/wiki/Main_Page

    y = -L(E)

    from sklearn import linear_model

    ##alpha = 0.0001
    ##L = linear_model.Lasso(alpha=alpha, copy_X=True)

    ## #L = linear_model.OrthogonalMatchingPursuit(copy_X=True, n_nonzero_coefs=5)

    aa = 0.0001 # L1 weight
    bb = 0.00001 # L2 weight
    alpha = aa + bb
    rho = aa / (aa + bb)
    lm = linear_model.ElasticNet(alpha=alpha, rho=rho, fit_intercept=True, copy_X=True)

    ## # # Penalise measurements with low absolute value
    ## # P = np.diag(1 + np.sqrt(np.abs(s / s.max())))
    ## # X = P.dot(X)
    ## # y = P.dot(y)

    beta = lm.fit(X, y).coef_

    sph_io.savez('odf_coeffs_%03d' % k, beta=beta, kernel_N=kernel_N,
                                        separation=gamma, weights=w,
                                        mevecs=[R0, R1],
                                        signal=E)

    nnz = np.sum(beta != 0)
    print 'Compression: %.2f%%' % ((len(beta) - nnz) / len(beta) * 100)
    print 'Non-zero coefficients: %d/%d' % (nnz, len(beta))
    print 'Error (Q-space):', np.linalg.norm(X.dot(beta) - y)

    ## beta[beta < 1e-5] = 0


    # ==========================
    # ODF-domain: Peak detection
    # ==========================

    if visualize_odf:
        mask = (beta != 0)
        plot.scatter_3D(theta_odf, phi_odf, color=(0, 0, 1))
        plot.scatter_3D(theta_odf[mask], phi_odf[mask], 1 + beta[mask]/beta.max(),
                          transparent=True, color=(1, 0, 0), scale_mode='scalar',
                          scale_factor=0.1, opacity=0.7)

        f1_theta, f1_phi, f1_r = coord.car2sph(*R0.dot([1, 0, 0]))
        f2_theta, f2_phi, f2_r = coord.car2sph(*R1.dot([1, 0, 0]))
        plot.scatter_3D(f1_theta, f1_phi, color=(0, 1, 0), scale_factor=0.15)
        plot.scatter_3D(f2_theta, f2_phi, color=(0, 1, 0), scale_factor=0.15)

        plot.show()
