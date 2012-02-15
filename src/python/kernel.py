import numpy as np
import scipy as sp

import sphere


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
    """Reproducing kernel.

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
    P, Q = len(theta_grid), len(phi_grid)
    PDF_recon = np.zeros((P, Q))

    for (k_theta, k_phi, w) in zip(kernels_theta, kernels_phi, weights):
        cos_theta = sphere.cos_inc_angle(grid_theta[:, None], grid_phi, k_theta, k_phi)
        PDF_recon += w * kernel(cos_theta, N)

    return PDF_recon


def kernel_matrix(s_theta, s_phi, q_theta, q_phi,
                  kernel=inv_funk_radon_even_kernel, N=18):
    """Construct the kernel matrix, A.

    Parameters
    ----------
    s_theta, s_phi : (P,) ndarray
        Sampling points (inclination and azimuthal angles).
    q_theta, q_phi : (Q,) ndarray
        Quadrature points (inclination and azimuthal angles).
    N : int
        Maximum degree of spherical harmonic subspace.

    """
    P = len(s_theta)
    Q = len(q_theta)

    s_theta = s_theta[:, None]
    s_phi = s_phi[:, None]

    cos_theta = sphere.cos_inc_angle(s_theta, s_phi, q_theta, q_phi)

    return kernel(cos_theta, N)
