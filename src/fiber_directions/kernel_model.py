__all__ = ['SparseKernelModel', 'SparseKernelFit']


import numpy as np
import scipy as sp
import scipy.special
from dipy.core.geometry import cart2sphere
from dipy.reconst.odf import OdfModel, OdfFit
from dipy.reconst.cache import Cache


def quadrature_points(N=72):
    """Load quadrature points on the sphere.

    Parameters
    ----------
    N : int, {72, 132, 492}
        A quadrature set with N points is loaded.

    Returns
    -------
    theta, phi : (N,) ndarray
        Quadrature point coordinates (inclination and azimuth).
    w : (N,) ndarray
        Quadrature weights.

    """
    import os
    basedir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data')

    quad_file = {72: 'qsph1-14-72DP.dat',
                 132: 'qsph1-16-132DP.dat',
                 492: 'qsph1-37-492DP.dat'}

    q_pts = np.loadtxt(os.path.join(basedir, quad_file[N]))
    q_theta, q_phi = cart2sphere(*q_pts[:, :3].T)[1:]
    q_w = q_pts[:, 3]

    return q_theta, q_phi, q_w


def cos_inc_angle(theta1, phi1, theta2, phi2):
    """
    Compute the cosine of the included angle between two vectors on the
    unit sphere, specified in spherical coordinates.

    Parameters
    ----------
    theta1, phi1 : float or ndarray
        First vector.
    theta2, phi2 : float or ndarray
        Second vector.

    Notes
    -----
    This is derived by taking the dot-product of the two vectors in Cartesian
    coordinates, then somewhat simplifying using trigonometric identities.

    More than one angle can be computed simultaneously, as long as theta1 and
    theta2, as well as phi1 and phi2 broadcast.

    """
    return np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2) \
           + np.cos(theta1) * np.cos(theta2)


def kernel_matrix(s_theta, s_phi, q_theta, q_phi, kernel, N=18):
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
    kernel : callable
        Kernel function used in the reconstruction.
    N : int
        Maximum degree of spherical harmonic subspace.

    """
    P = len(s_theta)
    Q = len(q_theta)

    s_theta = s_theta[:, None]
    s_phi = s_phi[:, None]

    cos_theta = cos_inc_angle(s_theta, s_phi, q_theta, q_phi)

    return kernel(cos_theta, N)


def kernel_reconstruct(kernels_theta, kernels_phi, weights,
                       grid_theta, grid_phi, kernel, N=8):
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
        cos_theta = cos_inc_angle(grid_theta, grid_phi,
                                  k_theta, k_phi)

        PDF_recon += w * kernel(cos_theta, N)

    return PDF_recon


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


def Linv(E):
    """
    Inverse of log(-log(.)) transform.

    """
    return np.exp(-np.exp(E))


class SparseKernelModel(OdfModel, Cache):
    def __init__(self, bvals, gradients, sh_order=8, qp=132,
                 loglog_tf=True, alpha=None, rho=None):
        """Sparse kernel model.

        Parameters
        ----------
        bvals : 1-D ndarray
            B-values.
        gradients : (N, 3) ndarray
            Gradient directions, xyz.
        sh_order : int
            Highest order of spherical harmonic fit.
        qp : {72, 132, 492}
            Number of kernels used to represent the signal.
        loglog_tf : bool
            Whether to perform ``log(-log(.))`` on the signal before fitting.
            In theory, this gives a better representation of the ODF (but does
            predict back the original signal).  Also, it seems not to work well
            for low b-values (<= 1500).

        alpha: float (optional)
            Parameter controlling the sparseness of ElasticNet. Defaults to 1

        rho: float (optional)
            Parameter controlling the sparseness of ElasticNet. Defaults to 0.5

        For a and b controlling the L1 and L2 norms of the weights:

        .. math ::

           a * L1 + b * L2

        set the input parameters `alpha` and `rho` to be:

        .. math ::

            alpha = a + b

            and

            rho = \frac{a}{a+b}


        See also
        --------
        sklearn.linear_model.ElasticNet

        """
        where_dwi = bvals > 0

        self.qp = qp
        self.sh_order = sh_order
        self.loglog_tf = loglog_tf
        self.gradient_theta, self.gradient_phi = \
                             cart2sphere(*gradients[where_dwi].T)[1:]

        self.kernel_theta, self.kernel_phi, _ = quadrature_points(N=qp)

        self.X = np.asfortranarray(
            kernel_matrix(self.gradient_theta, self.gradient_phi,
                          self.kernel_theta, self.kernel_phi,
                          kernel=inv_funk_radon_even_kernel,
                          N=self.sh_order)
            )

        if alpha is None:
            alpha = 1.0

        if rho is None:
            rho = 0.5

        self.alpha = alpha
        self.rho = rho

    def fit(self, signal):
        """Fit the model to the given signal.

        Parameter
        ---------
        signal : 1-D ndarray
            Signal measured at each b-vector.

        """
        if self.loglog_tf:
            y = -L(signal)
        else:
            y = signal

        from sklearn import linear_model
        lm = linear_model.ElasticNet(alpha=self.alpha, rho=self.rho,
                                     fit_intercept=True,
                                     copy_X=True)

        fit = lm.fit(self.X, y)
        beta = fit.coef_
        intercept = fit.intercept_

        return SparseKernelFit(beta=beta, intercept=intercept,
                               model=self)


class SparseKernelFit(OdfFit):
    def __init__(self, beta, intercept=0, model=None):
        self.beta = beta
        self.model = model
        self.intercept = intercept

    def odf(self, sphere):
        """Predict the ODF at the given vertices.

        """
        X = self.model.cache_get('odf_kernel_matrix',
                                 key=sphere)
        if X is None:
            X = kernel_matrix(sphere.theta, sphere.phi,
                              self.model.kernel_theta,
                              self.model.kernel_phi,
                              kernel=even_kernel,
                              N=self.model.sh_order)
            self.model.cache_set('odf_kernel_matrix', key=sphere, value=X)

        return np.dot(X, self.beta) + self.intercept

    def predict(self, sphere):
        """Predict the signal at the given vertices.

        """
        X = self.model.cache_get('predict_kernel_matrix')
        if X is None:
            X = kernel_matrix(sphere.theta, sphere.phi,
                              self.model.kernel_theta,
                              self.model.kernel_phi,
                              kernel=inv_funk_radon_even_kernel,
                              N=self.model.sh_order)
            self.model.cache_set('predict_kernel_matrix', key=sphere, value=X)

        E = np.dot(X, self.beta) + self.intercept

        if self.model.loglog_tf:
            E = Linv(-E)

        return E
