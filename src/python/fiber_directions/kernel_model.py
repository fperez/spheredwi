import numpy as np

import sys
sys.path.insert(0, '..')

from sphdif import coord, sphere
from sphdif.kernel import (kernel_matrix, kernel_reconstruct,
                           even_kernel, inv_funk_radon_even_kernel)


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


class SparseKernelModel:
    def __init__(self, bvals, gradients, sh_order=8, qp=132,
                 odf_vertices=None):
        where_dwi = bvals > 0

        self.sh_order = sh_order
        self.gradient_theta, self.gradient_phi, _ = \
                             coord.car2sph(*gradients[where_dwi].T)

        self.kernel_theta, self.kernel_phi, _ = sphere.quadrature_points(N=qp)

        self.X = kernel_matrix(self.gradient_theta, self.gradient_phi,
                               self.kernel_theta, self.kernel_phi,
                               kernel=inv_funk_radon_even_kernel,
                               N=self.sh_order)

        if odf_vertices is not None:
            self._odf_vertices = coord.car2sph(*odf_vertices.T)[:2]


    def fit(self, signal):
        y = -L(signal)

        from sklearn import linear_model

        aa = 0.0001 # L1 weight
        bb = 0.00001 # L2 weight
        alpha = aa + bb
        rho = aa / (aa + bb)
        lm = linear_model.ElasticNet(alpha=alpha, rho=rho, fit_intercept=True,
                                     copy_X=True)

        beta = lm.fit(self.X, y).coef_

        return SparseKernelFit(beta=beta, model=self)


class SparseKernelFit:
    def __init__(self, beta, model=None):
        self.beta = beta
        self.model = model

    def odf(self, odf_vertices=None):
        if odf_vertices is None:
            odf_theta, odf_phi = self.model._odf_vertices
        else:
            odf_theta, odf_phi = coord.car2sph(*odf_vertices.T)[:2]

        return kernel_reconstruct(self.model.kernel_theta,
                                  self.model.kernel_phi,
                                  self.beta,
                                  odf_theta, odf_phi,
                                  kernel=even_kernel,
                                  N=self.model.sh_order)
