"""Functions for spherical sampling and interpolation with proper quadratures.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Third-party
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Functions definitions
#-----------------------------------------------------------------------------

def interp_matrix(qpnts, spnts, npgrid, nsamp, deg_max):
    """Create matrix associated with kernel interpolation.

    Parameters
    ----------
    qpnts = quadrature points
    spnts = sample points
    npgrid = number of points in grid
    nsamp  = number of sample points
    deg_max = maximum degree of spherical harmonic subspace
    """
    # Initialize
    A = np.zeros((nsamp,npgrid))

    # Create matrix
    for i in xrange(nsamp):
        for j in xrange(npgrid):
            cosTheta = np.dot(spnts[i], qpnts[j])
            if(abs(cosTheta)>1):
                cosTheta = np.sign(cosTheta)
            A[i,j] = inv_funk_radon_kernel(cosTheta, deg_max)
    return A


def rand_sig(u, b, n, theta):
    """Create random signal on the sphere

    Parameters
    ----------
      u = unit vector
      b = ?
      n = # of signal components
      theta = ?

     lambda1 = 1700e-6 mm^2/s --typical #s
     lambda2 =  300e-6  "
     lambda3 =  300e-6  "
           b = 3000 s/mm^2
    """
    # Locally used names
    from numpy import dot, exp

    # Diffusion tensor parameters
    lambda1 = 1700e-6
    lambda2 = 300e-6
    lambda3 = 300e-6

    rotationMatrix = rotation3Dz(theta)

    # diagonal diffusion tensor for "prolate white matter"
    D1 = np.diag([lambda1, lambda2, lambda3])
    D2 = np.diag([lambda1, lambda2, lambda3])
    D3 = D1

    # orthonormal e-vectors of diffusion tensor
    V1 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float).reshape(3, 3)
    V2 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float).reshape(3, 3)
    V3 = V1

    V2 = dot(rotationMatrix, V2)
    V3 = dot(rotationMatrix, dot(rotationMatrix, V3))

    # Change basis to diagonalize diffusion tensor
    u1p = dot(V1.T, u)
    u2p = dot(V2.T, u)
    u3p = dot(V3.T, u)

    # XXX - check with cory these semantics
    if n==1:
        s = exp(-b * dot(u1p, dot(D1,u1p)) )   # Single mode
    elif n==2:
        s = 0.5 * (exp(-b * dot(u1p, dot(D1,u1p)) ) +
                  exp(-b * dot(u2p, dot(D2,u2p)) ) )
    elif n==3:
        s = (1.0/3) * (exp(-b * dot(u1p, dot(D1,u1p)) ) +
                       exp(-b * dot(u2p, dot(D2,u2p)) ) +
                       exp(-b * dot(u3p, dot(D3,u3p)) ) )
    return s


def inv_funk_radon_kernel(mu, N):
    """Reproducing kernel

    Calculate the inverse Funk-Radon transform of reproducing kernel for the
    space of spherical harmonics of maximum degree N.

    Parameters
    ----------
        mu = cos(theta)   (a scaler)
         N = maximum degree of subspace
    """

    # Check that -1 <= mu <= 1
    if abs(mu)>1.0:
        mu = np.sign(mu)

    # Need Legendre polynomials
    legPolys = legp(mu, N)
    p_at_zero = legp(0, N)
    coefs = 2*np.arange(0, N+1, 2) + 1
    ker = coefs*legPolys[::2]/p_at_zero[::2]
    return ker.sum() / (8*np.pi)


def legp(x, n):
    """Legendre polynomials: calculation of Legendre polynomials up degree N

    Legendre polynomials up to and including degree N evaluated at x =
    cos(theta)

    Parameters
    ----------
      x : scalar or 1d array
        cos(theta)
      n : int
        highest degree

    Returns
    -------

    Array of polynomial evaluations.  If x was a 1d array of length p, the
    return array has shape (n, p).
    """
    if isinstance(x, np.ndarray):
        assert x.ndim==1, "x must be scalar or 1d array"
        shape = (n+1, x.shape[0])
    else:
        shape = (n+1, )
    p = np.zeros(shape)
    p[0] = 1
    if n<=1:
        return p
    p[1] = x
    for i in range(1, n):
        p[i+1] = ((2*i + 1)*x*p[i] - i*p[i-1] ) / (i+1)
    return p


def rotation3Dz(theta):
    """Create a 3D  rotation matrix for rotation about z-axis.
    """
    rmat = np.zeros((3,3))
    rmat[0,0] = rmat[1,1] = np.cos(theta)
    rmat[0,1] = np.sin(theta)
    rmat[1,0] = -rmat[0,1]
    rmat[2,2] = 1
    return rmat
