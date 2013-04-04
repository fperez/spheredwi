import numpy as np
import scipy as sp
import scipy.special
from numpy.linalg import norm

import coord
import sphere
import plot






def std_kernel(mu, N):
    A = np.zeros_like(mu)

    for k in range(N + 1):
        Pk = sp.special.legendre(k)
        A += (2 * k + 1) / (4 * np.pi) * Pk(mu)

    return A


def even_kernel(mu, N):
    """Reproducing kernel.

    Compute the reproducing kernel for the
    even degree subspace of spherical harmonics of maximum
    degree N.

    Parameters
    ----------
    mu : float
        Cosine of the included angle between the kernel origin and a data point.
    N : int
        Maximum degree of spherical harmonic subspace.

    """
    A = np.zeros_like(mu)

    for k in range(2, N + 1, 2):
        Pk = sp.special.legendre(k)
        A += (2 * k + 1) / (4 * np.pi) * Pk(mu)

    return A


def inv_funk_radon_even_kernel(mu, N):
    """Q-space kernel.

    Calculate the inverse Funk-Radon transform and inverse
    spherical Laplacian of the reproducing kernel for the
    even degree subspace of spherical harmonics of maximum
    degree N, i.e.

    .. math::

       H(\mu) = \Delta^{-1} G^{-1} K_e(\mu)

    Parameters
    ----------
    mu : float
        Cosine of the included angle between the kernel origin and a data point.
    N : int
        Maximum degree of spherical harmonic subspace.

    """
    A = np.zeros_like(mu)

    for k in range(2, N + 1, 2):
        Pk = sp.special.legendre(k)
        A += (2 * k + 1) / (8 * np.pi**2 * Pk(0) * k * (k + 1)) * Pk(mu)

    return A


def inv_funk_radon_even_kernel_plus_ylm(mu, N):
    """Q-space kernel.

    Calculate the inverse Funk-Radon transform and inverse
    spherical Laplacian of the reproducing kernel for the
    even degree subspace of spherical harmonics from degree 4
    to a maximum degree N, i.e.

    .. math::

       H(\mu) = \Delta^{-1} G^{-1} Y^m_2 + \Delta^{-1} G^{-1} K_e(\mu)

    Parameters
    ----------
    mu : float
        Cosine of the included angle between the kernel origin and a data point.
    N : int
        Maximum degree of spherical harmonic subspace.

    """
    #A = np.zeros_like(mu)

    #for k in range(4, N + 1, 2):
    #    Pk = sp.special.legendre(k)
    #    A += (2 * k + 1) / (8 * np.pi**2 * Pk(0) * k * (k + 1)) * Pk(mu)
    
    A = 1.0
 
    return A




def kernel_reconstruct(kernels_theta, kernels_phi, weights,
                       grid_theta, grid_phi, kernel=std_kernel, N=8):
    """Kernel reconstruction of the PDF, on a grid on the sphere.

    Parameters
    ----------
    kernels_theta, kernels_phi : (N,) ndarray
        Positions of the kernels.
    weights : (N,) ndarray
        Weights of the kernels.
    grid_theta : (M,) ndarray
        Inclination angles of grid.
    grid_phi : (N,) ndarray
        Azimuth angles of grid.
    kernel : callable
        Kernel function used in the reconstruction.
    N : int
        Maximum order of kernel polynomials.

    Returns
    -------
    pdf : (M, N) ndarray
        Reconstruction of the PDF on the specified grid.

    """
    PDF_recon = np.zeros(np.broadcast(grid_theta, grid_phi).shape)

    for (k_theta, k_phi, w) in zip(kernels_theta, kernels_phi, weights):
        cos_theta = sphere.cos_inc_angle(grid_theta, grid_phi,
                                         k_theta, k_phi)

        PDF_recon += w * kernel(cos_theta, N)

    return PDF_recon


def kernel_matrix(s_theta, s_phi, q_theta, q_phi,
                  kernel=inv_funk_radon_even_kernel, N=18):
    """Construct the kernel matrix, A.

    The kernel projects sampling points to evaluation points.  Therefore,
    s_theta and s_phi are constrained by where you sampled, whereas q_theta and
    q_phi can be arbitrary (that's where the kernel is being evaluated).

    To phrase this another way, (s_theta, s_phi) define the location of the
    kernels [and, hence, the weights], whereas (q_theta, q_phi) is where we
    want to evaluate them.

    Parameters
    ----------
    s_theta, s_phi : (P,) ndarray
        Sampling points (inclination and azimuthal angles).
    q_theta, q_phi : (Q,) ndarray
        Evaluation points (inclination and azimuthal angles).
    N : int
        Maximum degree of spherical harmonic subspace.

    """
    P = len(s_theta)
    Q = len(q_theta)

    s_theta = s_theta[:, None]
    s_phi = s_phi[:, None]

    cos_theta = sphere.cos_inc_angle(s_theta, s_phi, q_theta, q_phi)

    return kernel(cos_theta, N)


def kernel_plot(kernel, grid_density=150, N=14):
    theta_grid = np.linspace(0, np.pi, grid_density)
    phi_grid = np.linspace(0, 2 * np.pi, grid_density)

    kernel_vals = kernel(sphere.cos_inc_angle(0, 0, theta_grid[:, None], phi_grid), N)

    plot.surf_grid_3D(kernel_vals, theta_grid, phi_grid, scale_radius=True)


if __name__ == "__main__":
    kernel_plot(even_kernel)
