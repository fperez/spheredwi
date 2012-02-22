from __future__ import division

import numpy as np
import scipy as sp
import scipy.special

import matplotlib.pyplot as plt

import sphere
from kernel import (kernel_matrix, std_kernel, even_kernel,
                    inv_funk_radon_even_kernel, kernel_reconstruct)
from linalg import rotation_around_axis

from signal_sim import single_tensor, single_tensor_ODF

# ========================
# Experiment configuration
# ========================

Q = 72  # Number of quadrature points
gamma = np.deg2rad(10)  # Angle separating fibers
D = 150 # Grid density for plots (higher => more dense)

# ====================================
# Load quadrature and set up test data
# ====================================

theta, phi, w = sphere.quadrature_points(N=Q)

# b is tau * |q|^2 in s/mm^2
# If b is too low, the signal does not attenuate enough to measure.
# Too high, the signal to noise ratio increases.

# Make up somewhat realistic b-values
b = 995 + np.random.normal(scale=4, size=Q)

xyz = np.column_stack(sphere.sph2car(np.ones_like(theta), theta, phi))

R0 = rotation_around_axis([0, 1, 0], gamma / 2.)
R1 = rotation_around_axis([0, 1, 0], -gamma / 2.)

S = single_tensor(gradients=xyz, bvals=b, S0=1, rotation=R0, SNR=None)
S += single_tensor(gradients=xyz, bvals=b, S0=1, rotation=R1, SNR=None)

# Visualise ODF
lat = np.linspace(-90, 90, D)
lon = np.linspace(-180, 180, D)
theta_grid = np.linspace(0, np.pi, D)
phi_grid = np.linspace(0, 2 * np.pi, D)


mg_phi, mg_theta = np.meshgrid(phi_grid, theta_grid)
xyz = np.dstack(sphere.sph2car(np.ones(mg_theta.shape), mg_theta, mg_phi))

## ODF
#ODF = single_tensor_ODF(xyz.reshape((-1, 3)), rotation=None)

# Q-space signal
ODF = single_tensor(gradients=xyz.reshape((-1, 3)), bvals=1000*np.ones(D*D),
                    S0=1, rotation=R0, SNR=None)

ODF = ODF.reshape(mg_theta.shape)

xyz = np.dstack(sphere.sph2car(ODF, mg_theta, mg_phi))

def basemap_ODF():
    m = sphere.surf_grid(ODF,
                         theta_grid, phi_grid,
                         vmin=0, vmax=1, projection='moll')
    plt.show()

def mayavi_ODF():
    from mayavi import mlab
    x, y, z = np.rollaxis(xyz, -1)
    mlab.mesh(x, y, z, scalars=ODF)
    #mlab.plot3d(x.ravel(), y.ravel(), z.ravel(), representation='points')
    mlab.show()

mayavi_ODF()

## def peak(theta, phi, sigma, theta_all, phi_all):
##     """Generate a peak of width sigma at (theta, phi)
##     with theta the polar angle, phi the azimuthal angle.
##     """

##     p = np.exp(-sphere.arc_length(theta, phi, theta_all, phi_all)**2 / float(sigma**2))
##     p /= p.max()

##     return p



## # Load quadrature points
## q_theta_72, q_phi_72, q_w_72 = sphere.quadrature_points(72)
## q_theta_132, q_phi_132, q_w_132 = sphere.quadrature_points(132)
## q_theta_492, q_phi_492, q_w_492 = sphere.quadrature_points(492)

# PDF parameters
## pdf_theta = [1.8 * np.pi / 3, np.pi / 3]
## pdf_phi = [2.5 * np.pi / 3, np.pi]
## pdf_sigma = [0.5, 0.5]

## def PDF(theta, phi):
##     """Evaluate the simulated PDF on the given points.

##     """
##     M, N = len(theta), len(phi)
##     pdf = np.zeros((M, N))

##     for (p_theta, p_phi, p_sigma) in zip(pdf_theta, pdf_phi, pdf_sigma):
##         pdf += peak(p_theta, p_phi, p_sigma, theta, phi)

##     return pdf / pdf.max()


## # <markdowncell>

## # # Test signal construction



## # Sample signal at several points
## # We choose the lower order quadrature points purely for convenience
## samples = np.zeros((72,))
## for (theta, phi, sigma) in zip(pdf_theta, pdf_phi, pdf_sigma):
##     samples += peak(theta, phi, sigma, q_theta_72, q_phi_72)

## # <markdowncell>

## # # Kernels




## # <markdowncell>

## # # Fitting



## def compare_reconstruction(s_theta, s_phi, q_theta, q_phi,
##                            kernel_weights, kernel, kernel_N,
##                            density):

##     mask = (kernel_weights != 0)
##     masked_theta = q_theta[mask]
##     masked_phi = q_phi[mask]
##     masked_weights = kernel_weights[mask]

##     f, (ax0, ax1) = plt.subplots(1, 2, figsize=[10, 5])

##     lat = np.linspace(-90, 90, density)
##     lon = np.linspace(-180, 180, density)
##     theta_grid, phi_grid = sphere.latlon2sph(lat, lon)

##     m = sphere.surf_grid(PDF(theta_grid[:, None], phi_grid),
##                          theta_grid, phi_grid,
##                          vmin=0, vmax=1, ax=ax0)

##     sphere.scatter(s_theta, s_phi, basemap=m)

##     PDF_recon = kernel_reconstruct(masked_theta, masked_phi, masked_weights,
##                                    theta_grid, phi_grid,
##                                    kernel=kernel, N=kernel_N)

##     m = sphere.surf_grid(PDF_recon, theta_grid, phi_grid, vmin=0, ax=ax1)

##     sphere.scatter(q_theta, q_phi,
##                    c=kernel_weights / kernel_weights.max(), cmap=plt.cm.gray,
##                    basemap=m)



## # Use coefficients from quadrature to do a direct fit

## q_theta, q_phi, q_w = q_theta_72, q_phi_72, q_w_72
## s_theta, s_phi = q_theta, q_phi

## kernel = std_kernel
## kernel_N = 16
## kernel_weights = samples * q_w
## kernel_weights[kernel_weights < 5e-2] = 0

## compare_reconstruction(s_theta, s_phi, q_theta, q_phi,
##                        kernel_weights, kernel, kernel_N, density=D)

## print 'Coefficients left after thresholding:', len(np.nonzero(kernel_weights)[0])

## # <markdowncell>

## # Minimising the L1 penalized system
## #
## # $$\Vert A\mathbf{x} - \mathbf{b} \Vert_2^2 + \lambda \vert \mathbf{x} \vert_1\quad
## # \mathrm{subject\ to}\quad x_i >= 0.$$
## #
## # Here, $A$ is reproducing kernel in sparse matrix form, $\mathbf{b}$ is the measured signal
## # and $\mathbf{x}$ are the coefficents in the sparse represenation.
## #
## # See the <a href="http://ugcs.caltech.edu/~srbecker/wiki/Main_Page">sparse and low-rank approximation wiki</a>
## # for more detail.






## # Use the low-order quadrature points as sampling points
## # in this experiment, since they're fairly equally spaced
## # across the sphere (unlike in the previous example,
## # we don't need them to be on the quadrature points).
## #
## # We use a finer representation of the function
## # to best establish the fiber directions.
## #
## s_theta, s_phi = q_theta_72, q_phi_72
## q_theta, q_phi, q_w = q_theta_492, q_phi_492, q_w_492

## kernel = std_kernel
## kernel_N = 16
## X = kernel_matrix(s_theta, s_phi, q_theta, q_phi,
##                   kernel=kernel, N=kernel_N)
## y = samples

## # # Penalise measurements with low absolute value
## # E = np.diag(1 + np.sqrt(np.abs(s / s.max())))
## # A = E.dot(A)
## # b = E.dot(b)

## from sklearn import linear_model

## alpha = 0.01
## L = linear_model.Lasso(alpha=alpha, copy_X=True)

## #L = linear_model.OrthogonalMatchingPursuit(copy_X=True, n_nonzero_coefs=5)

## #a = 0.1 # L1 weight
## #b = 0.8 # L2 weight
## #alpha = a + b
## #rho = a / (a + b)
## #L = linear_model.ElasticNet(alpha=alpha, rho=rho, fit_intercept=False, copy_X=True)

## beta = L.fit(X, y).coef_

## beta[beta < 1e-5] = 0

## compare_reconstruction(s_theta, s_phi, q_theta, q_phi,
##                        beta, kernel, kernel_N, density=D)

## plt.show()

## print "Number of coefficients: %d" % np.sum(beta != 0)

## # <markdowncell>

## # <div
## # style="border-top: solid 5px;"><br/></span>
## #
## # # Experimental



## # Use Mayavi to surf kernel surface
