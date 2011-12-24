#!/usr/bin/env python
"""Fiber reconstruction example script.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function, division

from enthought.mayavi import mlab

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import cvxmod as cvx

# Local imports
from sphdif import sphquad as sph
reload(sph)  # For interactive development


# Make global some frequently used functions
from numpy import dot
from numpy.linalg import norm

#-----------------------------------------------------------------------------
# Main script
#-----------------------------------------------------------------------------

# Load quadrature
qsph1_37_492DP = np.loadtxt('data/qsph1-37-492DP.dat')
quad_pnts = qsph1_37_492DP[:, :3]
N = 18         # maximum degree of subspace
n_qpnts = 492  # number of points in quadrature

#quad_pnts_up_hem = quad_pnts[(np.where(quad_pnts[:,2]>0))]
#n_qpnts,nn = quad_pnts_up_hem.shape
#quad_pnts = quad_pnts_up_hem

## # Alternative quadrature points
## from dipy.data import get_sphere
## data_file = get_sphere('symmetric362') # 'symmetric642'
## quad_pnts = np.load(data_file)['vertices']
## N = 18
## n_qpnts = len(quad_pnts)

# Sample signal on lower degree quadrature
qsph1_16_132DP = np.loadtxt('data/qsph1-16-132DP.dat')
sample_pnts  = qsph1_16_132DP[:, :3]
n_sample_pnts = 132

#sample_pnts_up_hem = sample_pnts[(np.where(sample_pnts[:,2]>0))]
#n_sample_pnts,nn = sample_pnts_up_hem.shape
#sample_pnts = sample_pnts_up_hem


# Create reproducing-kernel (sparse representation) matrix
nA = sph.interp_matrix_new(quad_pnts, sample_pnts, n_qpnts, n_sample_pnts, N)

# Create signal
print('Creating signal...')
n_fibers = 2                      # number of Gaussian components (max n=3)
b        = 4000                   # s/mm^2
r_angle  = -np.pi/2
signal   =  np.zeros(n_sample_pnts)
for i in range(n_sample_pnts):
    signal[i] = sph.rand_sig(sample_pnts[i, :3].T, b, n_fibers, r_angle)

SNR = []

nRealizations = 1
for kk in range(nRealizations):
    # Make Rician noise
    sigma  = 0.0                          # standard deviation
    noiseR = sigma * np.random.randn(*signal.shape)
    noiseI = sigma * np.random.randn(*signal.shape)
    noise  = noiseR + 1j*noiseI

    SNR.append(10 * np.log10(norm(signal,2)/norm(noise,2)))
    print('Signal to noise ratio: %0.5g' % SNR[kk])

    # Add noise to signal
    rSig = signal + noise
    rSig = abs(rSig)                 #phase is not used in MRI
    
    #Take ln(-ln(signal))
    #for jj in range(n_sample_pnts):
    #  rSig[jj] = sph.ilog(rSig[jj],0.001)


    # Choose regularization parameter
    # lambda > lambda_max -> zero solution
    lambda_max = 2*norm(dot(nA.T, rSig.T), np.inf) 
    #lamb = 0.015*lambda_max

    lamb = 0.5125*lambda_max


    print('Solving L1 penalized system with cvxmod...')
    # For reference, original specification of the convex optimization problem
    # using the matlab cvxopt syntax.
    """
    variable ndCoefsl1(nQpnts)
    cvx_precision('low')
    minimize( norm( nA * ndCoefsl1 - rSig.T,2) + lamb*norm(ndCoefsl1,1) )
    subject to
      0.0 <= ndCoefsl1
      """
    ndCoefsl1 = cvx.optvar('ndCoefsl1', n_qpnts)
    nAp = cvx.param('nA', value=cvx.matrix(nA))
    rSigp = cvx.param('rSig', value=cvx.matrix(rSig))
    objective = cvx.minimize(cvx.norm2( nAp * ndCoefsl1 - rSigp) +
                             lamb*cvx.norm1(ndCoefsl1) )
    constraints = [0.0 <= ndCoefsl1]
    prob = cvx.problem(objective, constraints)

    # Call the solver
    prob.solve()

    # Convert the cvxmod objects to plain numpy arrays for further processing
    nd_coefs_l1 = np.array(ndCoefsl1.value).squeeze()

    # Cutoff those coefficients that are less than cutoff
    cutoff =  nd_coefs_l1.mean() + 2.5*nd_coefs_l1.std(ddof=1)
    nd_coefs_l1_trim = np.where(nd_coefs_l1 > cutoff, nd_coefs_l1, 0)

    # Get indices needed for sorting coefs, in reverse order.
    sortedIndex = nd_coefs_l1_trim.argsort()[::-1]
    # number of significant coefficients
    nSig = (nd_coefs_l1_trim > 0).sum()
    print('Precentage compression: %0.5g' % (100*(1.00 - (nSig/(1.0*n_qpnts)))))

    # Used for taking only some of the points---now using the whole sphere
    # Let -1.5 -> 0 and get only the hemisphere with x>0
    cond  = np.where(quad_pnts[sortedIndex[:nSig], 0] >= -1.5)
    indexPos = sortedIndex[cond]
    points   = quad_pnts[indexPos, :3]
   
    coefs    = nd_coefs_l1_trim[indexPos]

    np.savetxt('recon_data.dat', points)


    #--Visualize signal
    # Create a spherical mesh
    npts = 101
    r    = 1.0 
    pi   = np.pi
    cos  = np.cos
    sin  = np.sin
    theta, phi = np.mgrid[0:pi:npts*1j, 0:2*pi:npts*1j]

    x = r*sin(theta)*cos(phi)
    y = r*sin(theta)*sin(phi)
    z = r*cos(theta)

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 400))

    k = np.zeros(x.shape)

    for i in range(npts):
      for j in range(npts):
        k[i,j] = sph.even_pODF(np.array([x[i,j], y[i,j], z[i,j]]),points,coefs,N)

    s = k
                      
    mlab.mesh(k*x, k*y, k*z, scalars=s, colormap='jet')
    mlab.show()

    
    #--Start clustering -- maybe need to use a different set of nodes to evaluate pODF
    #
    # Sample reconstructed pODF using rejection technique -- it's not strictly non-negative!!
    nsamples = 100
    sampled_points = sph.sample_pODF(nsamples,points,coefs,N)




