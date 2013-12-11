__all__ = ['SparseKernelModel', 'SparseKernelFit']


import numpy as np
import scipy as sp
import scipy.special

from numpy.linalg import norm
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
                 192: 'qsph1-23-192DP.dat',
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


def coherence(A):
    """Calculate the coherence of a given matrix A

    """
    inner_prods=np.dot(A.T, A)

    n_inner_prods = np.zeros_like(inner_prods)

    (n,n) = inner_prods.shape

    for i in range(n):
      for j in range(n):
        n_inner_prods[i,j] = np.abs(inner_prods[i,j])/(norm(A[:,i]) * norm(A[:,j]))
 


    coherence = n_inner_prods - np.diag(np.ones(n)) 

    return np.max(coherence)



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


def kernel(mu, N):
    """Reproducing kernel.

    Compute the reproducing kernel for the subspace
    of spherical harmonics of maximum degree N.

    Parameters
    ----------
    mu : float
        Cosine of the included angle between the kernel origin and a data point.
    N : int
        Maximum degree of spherical harmonic subspace.

    """
    A = np.zeros_like(mu)

    for k in range(N + 1):
        Pk = sp.special.legendre(k)
        A += (2 * k + 1) / (4 * np.pi) * Pk(mu)

    return A




def gaussian_rbf(mu, alpha):
    """Gaussian radial basis function: exp(-alpha r^2). In terms
    of mu, the cosine of the angle, exp(-alpha 2*(1-mu) ). 

    Compute the Gaussian radial basis function using the 
    spherical harmonic expansion

    Parameters
    ----------
    mu : float
        Cosine of the included angle between the kernel origin and a data point.
    alpha : float
        Shape parameter ``alpha > 0``.

    """
    A = np.zeros_like(mu)
    n_max = 40 

    for k in range(2, n_max, 2):
        Pk = sp.special.legendre(k)
        A += (2.0 * k + 1.0) * an(k, alpha) * Pk(mu)

    return A * 0.5


def an(n, alpha):
    """Calculate expansion coefficient for Gaussian RBF from Baxter and Hubbert.
    """
    return np.sqrt(np.pi / alpha) * np.exp(-2.0*alpha) * \
      sp.special.iv(n + 1.0/2.0, 2.0*alpha)


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


def inv_funk_radon_gaussian_rbf(mu, alpha):
    """Q-space kernel.

    Calculate the inverse Funk-Radon transform and inverse
    spherical Laplacian of the Gaussian RBF with shape 
    parameter alpha

    .. math::

       H(\mu) = \Delta^{-1} G^{-1} Exp(-alpha 2*(1-mu))

    Parameters
    ----------
    mu    : float
            Cosine of the included angle between the kernel origin and a data point.
    alpha : float
            shape parameter 

    """
    A = np.zeros_like(mu)
    n_max = 40
    
    for k in range(2, n_max, 2):
        Pk = sp.special.legendre(k)
        A +=  (((2.0 * k + 1.0) * an(k,alpha)) /  (4.0 * np.pi * Pk(0) * k * (k + 1.0)))  * Pk(mu)

    return A


def inv_funk_radon_even_kernel_Y2m(mu, N):
    """Q-space kernel.

    Calculate the inverse Funk-Radon transform and inverse
    spherical Laplacian of the reproducing kernel for the
    even degree subspace of spherical harmonics of degree 4 to maximum
    degree N plus a term involving the spherical harmonics of degree l=2.

    .. math::

       H(\Omega'\cdot\Omega) =\Delta^{-1} G^{-1}\sum_{|m|\le 2} + \Delta^{-1} G^{-1} K_e(\Omega'\cdot\Omega)

    Parameters
    ----------
    mu : float
        Cosine of the included angle between the kernel origin and a data point.
    N : int
        Maximum degree of spherical harmonic subspace.

    """
    A = np.zeros_like(mu)

    for k in range(4, N + 1, 2):
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
    def __init__(self, gtab, sh_order=8, qp=132,
                 loglog_tf=True, l1_ratio=None, alpha=None):
        """Sparse kernel model.

        Parameters
        ----------
        gtab : GradientTable
            B-values and gradient directions.
        sh_order : int
            Highest order of spherical harmonic fit.
        qp : {72, 132, 492}
            Number of kernels used to represent the signal.
        loglog_tf : bool
            Whether to perform ``log(-log(.))`` on the signal before fitting.
            In theory, this gives a better representation of the ODF (but does
            predict back the original signal).  Also, it seems not to work well
            for low b-values (<= 1500).
        l1_ratio : float (optional)
            Argument passed to sklearn's ElasticNet to control L1 vs L2
            penalization.  Should be > 0.01.
        alpha : float (optional)
            Argument passed to sklearn's ElasticNet.  Controls the weight of
            both L1 and L2 penalties.

        See also
        --------
        sklearn.linear_model.ElasticNet

        """
        mask = gtab.bvals > 0
        bvecs = gtab.bvecs[mask]

        self.qp = qp
        self.sh_order = sh_order
        self.loglog_tf = loglog_tf
        self.gradient_theta, self.gradient_phi = \
                             cart2sphere(*bvecs.T)[1:]

        self.kernel_theta, self.kernel_phi, _ = quadrature_points(N=qp)

        # Since our kernel is even, only use quadrature points above or on the
        # equator
        mask = self.kernel_theta >= (np.pi / 2)
        self.kernel_theta = self.kernel_theta[mask]
        self.kernel_phi = self.kernel_phi[mask]

        self.X = np.asfortranarray(
            kernel_matrix(self.gradient_theta, self.gradient_phi,
                          self.kernel_theta, self.kernel_phi,
                          kernel=inv_funk_radon_even_kernel,
                          N=self.sh_order)
            )

        if l1_ratio is None:
            l1_ratio = 1
        if alpha is None:
            alpha = 1

        self.l1_ratio = l1_ratio
        self.alpha = alpha

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
        lm = linear_model.ElasticNet(l1_ratio=self.l1_ratio, alpha=self.alpha,
                                     fit_intercept=True,
                                     copy_X=True)


	#from sklearn.linear_model import RandomizedLasso as rl
	#lasso = rl(alpha=0.01,verbose=False,fit_intercept=True,scaling=0.001,n_resampling=500)
	#rl_object = lasso.fit(self.X, y)

	#support = np.squeeze(np.where(rl_object.get_support()))       	

	#clf = linear_model.LinearRegression()
	#clf.fit(self.X[:,support],y)

        #beta = np.zeros(self.qp)
	#beta[support] = clf.coef_

	#intercept = clf.intercept_

	(rows,cols) = self.X.shape
        # Handle "nan" as missing observations
	#your_array = np.arange(rows)
	#np.random.shuffle(your_array)
        #mask = your_array[:34]

        mask = np.array([0, 32, 34, 39, 27, 17, 35, 58, 33, 25, 46, 61, 43, 23,  7, 31, 57, 
			42,  4, 41, 63, 20,  2, 47, 15, 22,  5, 38, 44, 52, 50,  8, 51, 48])

        #mask = ~np.isnan(y)
        fit = lm.fit(self.X[mask, :], y[mask])
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
