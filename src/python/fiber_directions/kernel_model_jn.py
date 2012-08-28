__all__ = ['SparseKernelModel', 'SparseKernelFit']


import numpy as np
import scipy as sp
import scipy.special
from dipy.core.geometry import cart2sphere


#add code to use RegressorMixin class
#------------------------------------------------------------------------------
class RegressorMixin(object):
    """ Mixin class for all regression estimators in the scikit learn
    """

    def score(self, X, y):
        """ Returns the coefficient of determination of the prediction

            Parameters
            ----------
            X : array-like, shape = [n_samples, n_features]
                Training set.

            y : array-like, shape = [n_samples]

            Returns
            -------
            z : float
        """
        return r2_score(y, self.predict(X))

#------------------------------------------------------------------------------

#add code to use BaseEstimator class
#------------------------------------------------------------------------------
class BaseEstimator(object):
    """ Base class for all estimators in the scikit learn

        Notes
        -----
        All estimators should specify all the parameters that can be set
        at the class level in their __init__ as explicit keyword
        arguments (no *args, **kwargs).

    """

    @classmethod
    def _get_param_names(cls):
        """ Get parameter names for the estimator
        """
        try:
            args, varargs, kw, default = inspect.getargspec(cls.__init__)
            assert varargs is None, (
                'scikit learn estimators should always specify their '
                'parameters in the signature of their init (no varargs).'
                )
            # Remove 'self'
            # XXX: This is going to fail if the init is a staticmethod, but
            # who would do this?
            args.pop(0)
        except TypeError:
            # No explicit __init__
            args = []
        args.sort()
        return args

    def _get_params(self, deep=True):
        """ Get parameters for the estimator

            Parameters
            ----------
            deep: boolean, optional
                If True, will return the parameters for this estimator and
                contained subobjects that are estimators.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, '_get_params'):
                deep_items = value._get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """ Set the parameters of the estimator.

        The method works on simple estimators as well as on nested
        objects (such as pipelines). The former have parameters of the
        form <component>__<parameter> so that it's possible to update
        each component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return
        valid_params = self._get_params(deep=True)
        for key, value in params.iteritems():
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                assert name in valid_params, ('Invalid parameter %s '
                                              'for estimator %s' %
                                             (name, self))
                sub_object = valid_params[name]
                assert hasattr(sub_object, '_get_params'), (
                    'Parameter %s of %s is not an estimator, cannot set '
                    'sub parameter %s' %
                        (sub_name, self.__class__.__name__, sub_name)
                    )
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                assert key in valid_params, ('Invalid parameter %s '
                                              'for estimator %s' %
                                             (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def _set_params(self, **params):
        if params != {}:
            warnings.warn("Passing estimator parameters to fit is deprecated;"
                          " use set_params instead",
                          category=DeprecationWarning)
        return self.set_params(**params)

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (
                class_name,
                _pprint(self._get_params(deep=False),
                        offset=len(class_name),
                ),
            )

    def __str__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (
                class_name,
                _pprint(self._get_params(deep=True),
                        offset=len(class_name),
                        printer=str,
                ),
            )

#------------------------------------------------------------------------------


#add code to use LinearModel class
#------------------------------------------------------------------------------
class LinearModel(BaseEstimator, RegressorMixin):
    """Base class for Linear Models"""

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Returns predicted values.
        """
        X = safe_asanyarray(X)
        return safe_sparse_dot(X, self.coef_.T) + self.intercept_

    @staticmethod
    def _center_data(X, y, fit_intercept, normalize=False):
        """
        Centers data to have mean zero along axis 0. This is here because
        nearly all linear models will want their data to be centered.

        WARNING : This function modifies X inplace :
            Use sklearn.utils.as_float_array before to convert X to np.float.
            You can specify an argument overwrite_X (default is False).
        """
        if fit_intercept:
            if sp.sparse.issparse(X):
                X_mean = np.zeros(X.shape[1])
                X_std = np.ones(X.shape[1])
            else:
                X_mean = X.mean(axis=0)
                X -= X_mean
                if normalize:
                    X_std = np.sqrt(np.sum(X ** 2, axis=0))
                    X_std[X_std==0] = 1
                    X /= X_std
                else:
                    X_std = np.ones(X.shape[1])
            y_mean = y.mean()
            y = y - y_mean
        else:
            X_mean = np.zeros(X.shape[1])
            X_std = np.ones(X.shape[1])
            y_mean = 0.
        return X, y, X_mean, y_mean, X_std

    def _set_intercept(self, X_mean, y_mean, X_std):
        """Set the intercept_
        """
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_std
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0	

#-----------------------------------------------------------------------------



#add code to use ElasticNet class
#------------------------------------------------------------------------------
#Import necessary packages
import sys
import warnings
import itertools
import operator
from abc import ABCMeta, abstractmethod

#from .base import LinearModel
#from ..base import RegressorMixin
#from .base import sparse_center_data
#from ..utils import as_float_array
#from ..cross_validation import check_cv
#from ..externals.joblib import Parallel, delayed
#from ..utils.extmath import safe_sparse_dot

#from . import cd_fast

#-------My Attempt at changing cd_fast code format from cython to python-------
import numpy as np
import numpy.linalg as linalg
import warnings

def fsign(f):
	if f==0:
		return 0
	elif f>0:
		return 1.0
	else:
		return -1.0
		
def fmax(x,y):
	if x>y:
		return x
	
	return y

def enet_coordinate_descent(w,alpha,beta,X,y,max_iter,tol,positive=False):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression

        We minimize

        1 norm(y - X w, 2)^2 + alpha norm(w, 1) + beta norm(w, 2)^2
        -                                         ----
        2                                           2

    """
				
    # get the data information into easy vars
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # compute norms of the columns of X
    norm_cols_X = (X**2).sum(axis=0)
				
				# initialize variable d_w_tol
    d_w_tol = tol

    if alpha == 0:
        warnings.warn("Coordinate descent with alpha=0 may lead to unexpected"
            " results and is discouraged.")

    R = y - np.dot(X, w)

    tol = tol * linalg.norm(y) ** 2

    for n_iter in range(max_iter):
        w_max = 0.0
        d_w_max = 0.0
        for ii in xrange(n_features):  # Loop over coordinates
            if norm_cols_X[ii] == 0.0:
                continue

            w_ii = w[ii]  # Store previous value

            if w_ii != 0.0:
                R += w_ii * X[:,ii]

            tmp = (X[:,ii]*R).sum()

            if positive and tmp < 0:
                w[ii] = 0.0
            else:
                w[ii] = fsign(tmp) * fmax(np.abs(tmp) - alpha, 0) \
                    / (norm_cols_X[ii] + beta)

            if w[ii] != 0.0:
                 R -=  w[ii] * X[:,ii] # Update residual

            # update the maximum absolute coefficient update
            d_w_ii = np.abs(w[ii] - w_ii)
            if d_w_ii > d_w_max:
                d_w_max = d_w_ii

            if np.abs(w[ii]) > w_max:
                w_max = np.abs(w[ii])

        if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
            # the biggest coordinate update of this iteration was smaller than
            # the tolerance: check the duality gap as ultimate stopping
            # criterion

            XtA = np.dot(X.T, R) - beta * w
            if positive:
                dual_norm_XtA = np.max(XtA)
            else:
                dual_norm_XtA = linalg.norm(XtA, np.inf)

            # TODO: use squared L2 norm directly
            R_norm = linalg.norm(R)
            w_norm = linalg.norm(w, 2)
            if (dual_norm_XtA > alpha):
                const = alpha / dual_norm_XtA
                A_norm = R_norm * const
                gap = 0.5 * (R_norm ** 2 + A_norm ** 2)
            else:
                const = 1.0
                gap = R_norm ** 2

            gap += alpha * linalg.norm(w, 1) - const * np.dot(R.T, y) + \
                  0.5 * beta * (1 + const ** 2) * (w_norm ** 2)

            if gap < tol:
                # return if we reached desired tolerance
                break

    return w, gap, tol
#---------------------------Back to ElasticNet code---------------------------

# ElasticNet model


class ElasticNet(LinearModel, RegressorMixin):
    """Linear Model trained with L1 and L2 prior as regularizer

    Minimizes the objective function::

            1 / (2 * n_samples) * ||y - Xw||^2_2 +
            + alpha * rho * ||w||_1 + 0.5 * alpha * (1 - rho) * ||w||^2_2

    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::

            a * L1 + b * L2

    where::

            alpha = a + b and rho = a / (a + b)
	    
	 The parameter rho corresponds to alpha in the glmnet R package while
    alpha corresponds to the lambda parameter in glmnet. Specifically, rho =
    1 is the lasso penalty. Currently, rho <= 0.01 is not reliable, unless
    you supply your own sequence of alpha.

    Parameters
    ----------
    alpha : float
        Constant that multiplies the penalty terms. Defaults to 1.0
        See the notes for the exact mathematical meaning of this
        parameter

    rho : float
        The ElasticNet mixing parameter, with 0 < rho <= 1. For rho = 0
        the penalty is an L1 penalty. For rho = 1 it is an L2 penalty.
	For 0 < rho < 1, the penalty is a combination of L1 and L2

    fit_intercept: bool
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    normalize : boolean, optional
        If True, the regressors X are normalized

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to 'auto' let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always True to preserve sparsity.

    max_iter: int, optional
        The maximum number of iterations

    copy_X : boolean, optional, default False
     	If True, X will be copied; else, it may be overwritten.

    tol: float, optional
        The tolerance for the optimization: if the updates are
        smaller than 'tol', the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than tol.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive: bool, optional
        When set to True, forces the coefficients to be positive.

    Attributes
    ----------
    coef_ : array, shape = [n_features]
        parameter vector (w in the cost function formula)
	
    sparse_coef_: scipy.sparse matrix, shape = [n_features, 1]
        sparse_coef_: is a readonly property derived from coef_

    intercept_ : float
        independent term in decision function.

    Notes
    -----
    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a fortran contiguous numpy array.
    """
    def __init__(self, alpha=1.0, rho=0.5, fit_intercept=True,
                 normalize=False, precompute='auto', max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False):
        self.alpha = alpha
        self.rho = rho
        self.coef_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
	self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.intercept_ = 0.0

    def fit(self, X, y, Xy=None, coef_init=None):
        """Fit Elastic Net model with coordinate descent

        Parameters
        -----------
        X: ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y: ndarray, (n_samples)
            Target
        Xy : array-like, optional
            Xy = np.dot(X.T, y) that can be precomputed. It is useful
            only when the Gram matrix is precomputed.
        coef_init: ndarray of shape n_features
		The initial coeffients to warm-start the optimization

        Notes
        -----

        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a fortran contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """

        fit = self._sparse_fit if sp.sparse.isspmatrix(X) else self._dense_fit
        fit(X, y, Xy, coef_init)
        return self

    def _dense_fit(self, X, y, Xy=None, coef_init=None):

        # X and y must be of type float64
	X = np.asanyarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape

        X_init = X
        X, y, X_mean, y_mean, X_std = self._center_data(X, y, 
		self.fit_intercept,self.normalize)
                #self.fit_intercept, self.normalize, copy=self.copy_X)
		
        precompute = self.precompute
        if X_init is not X and hasattr(precompute, '__array__'):
            # recompute Gram
            # FIXME: it could be updated from precompute and X_mean
            # instead of recomputed
            precompute = 'auto'
        if X_init is not X and Xy is not None:
            Xy = None  # recompute Xy

        if coef_init is None:
            if not self.warm_start or self.coef_ is None:
                self.coef_ = np.zeros(n_features, dtype=np.float64)
	else:
            if coef_init.shape[0] != X.shape[1]:
                raise ValueError("X and coef_init have incompatible " +
                                  "shapes.")
            self.coef_ = coef_init

        alpha = self.alpha * self.rho * n_samples
        beta = self.alpha * (1.0 - self.rho) * n_samples

        X = np.asfortranarray(X)  # make data contiguous in memory

        # precompute if n_samples > n_features
        if hasattr(precompute, '__array__'):
            Gram = precompute
        elif precompute == True or \
               (precompute == 'auto' and n_samples > n_features):
            Gram = np.dot(X.T, X)
        else:
            Gram = None
	if Gram is None:
            self.coef_, self.dual_gap_, self.eps_ = \
                    enet_coordinate_descent(self.coef_, alpha, beta,
                            X, y, self.max_iter, self.tol, self.positive)
        else:
            if Xy is None:
                Xy = np.dot(X.T, y)
            self.coef_, self.dual_gap_, self.eps_ = \
                    enet_coordinate_descent_gram(self.coef_, alpha,
                    beta, Gram, Xy, y, self.max_iter, self.tol, self.positive)

        self._set_intercept(X_mean, y_mean, X_std)

        if self.dual_gap_ > self.eps_:
            warnings.warn('Objective did not converge, you might want'
                          ' to increase the number of iterations')

        # return self for chaining fit and predict calls
        return self
      
    def _sparse_fit(self, X, y, Xy=None, coef_init=None):

        if not sp.sparse.isspmatrix_csc(X) or not np.issubdtype(np.float64, X):
            X = sp.sparse.csc_matrix(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes.\n" +
                             "Note: Sparse matrices cannot be indexed w/" +
                             "boolean masks (use `indices=True` in CV).")

        # NOTE: we are explicitly not centering the data the naive way to
        # avoid breaking the sparsity of X

        n_samples, n_features = X.shape[0], X.shape[1]

        if coef_init is None and \
            (not self.warm_start or self.coef_ is None):
            self.coef_ = np.zeros(n_features, dtype=np.float64)
	else:
            if coef_init.shape[0] != X.shape[1]:
                raise ValueError("X and coef_init have incompatible " +
                                  "shapes.")
            self.coef_ = coef_init

        alpha = self.alpha * self.rho * n_samples
        beta = self.alpha * (1.0 - self.rho) * n_samples
        X_data, y, X_mean, y_mean, X_std = sparse_center_data(X, y,
                                                       self.fit_intercept,
                                                       self.normalize)

        self.coef_, self.dual_gap_, self.eps_ = \
                cd_fast.sparse_enet_coordinate_descent(
                    self.coef_, alpha, beta, X_data, X.indices,
                    X.indptr, y, X_mean / X_std,
                    self.max_iter, self.tol, self.positive)

        self._set_intercept(X_mean, y_mean, X_std)	
		
	if self.dual_gap_ > self.eps_:
            warnings.warn('Objective did not converge, you might want'
                                'to increase the number of iterations')

        # return self for chaining fit and predict calls
        return self

    @property
    def sparse_coef_(self):
        """ sparse representation of the fitted coef """
        return sp.sparse.csr_matrix(self.coef_)

    def decision_function(self, X):
        """Decision function of the linear model

        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape [n_samples, n_features]

        Returns	
	-------
        array, shape = [n_samples] with the predicted real values
        """
        if sp.sparse.isspmatrix(X):
            return np.ravel(safe_sparse_dot(self.coef_, X.T, \
                                        dense_output=True) + self.intercept_)
        else:
            return super(ElasticNet, self).decision_function(X)
	    
#------------------------------------------------------------------------------

#Return to original code:			
		
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
    basedir = os.path.abspath(os.path.dirname(__file__))

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


class SparseKernelModel:
    def __init__(self, bvals, gradients, sh_order=8, qp=132,
                 loglog_tf=True):
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

        #from sklearn import linear_model

        aa = 0.0001 # L1 weight
        bb = 0.00001 # L2 weight
        alpha = aa + bb
        rho = aa / (aa + bb)
	
        #lm = linear_model.ElasticNet(alpha=alpha, rho=rho, fit_intercept=True,
                                    # copy_X=True)
	lm = ElasticNet(alpha=alpha, rho=rho, fit_intercept=True, copy_X=True)

        fit = lm.fit(self.X, y)
        beta = fit.coef_
        intercept = fit.intercept_

        return SparseKernelFit(beta=beta, intercept=intercept,
                               model=self)


class SparseKernelFit:
    def __init__(self, beta, intercept=0, model=None):
        self.beta = beta
        self.model = model
        self.intercept = intercept


    def odf(self, vertices=None, cache=None):
        """Predict the ODF at the given vertices.

        """
        if vertices is None:
            self._odf_kernel_matrix = cache._odf_kernel_matrix
        else:
            odf_theta, odf_phi = cart2sphere(*vertices.T)[1:]
            X = kernel_matrix(odf_theta, odf_phi,
                              self.model.kernel_theta,
                              self.model.kernel_phi,
                              kernel=even_kernel,
                              N=self.model.sh_order)
            self._odf_kernel_matrix = X

        return np.dot(self._odf_kernel_matrix, self.beta) + \
               self.intercept


    def predict(self, vertices=None, cache=None):
        """Predict the signal at the given vertices.

        """
        if vertices is None:
            self._pred_kernel_matrix = cache._pred_kernel_matrix
        else:
            pred_theta, pred_phi = cart2sphere(*vertices.T)[1:]
            X = kernel_matrix(pred_theta, pred_phi,
                              self.model.kernel_theta,
                              self.model.kernel_phi,
                              kernel=inv_funk_radon_even_kernel,
                              N=self.model.sh_order)
            self._pred_kernel_matrix = X

        E = np.dot(self._pred_kernel_matrix, self.beta) + \
            self.intercept

        if self.model.loglog_tf:
            E = Linv(-E)

        return E
