#!/usr/bin/env python
"""Fiber reconstruction example script.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function, division

#from enthought.mayavi import mlab

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import cvxpy as cvx
from scipy.optimize import fmin_bfgs

# Local imports
from sphdif import sphquad as sph
reload(sph)  # For interactive development

# Load Fortran kernels
from sphdif import even_pODF_f   as epODF
from sphdif import sample_pODF_f as spODF


# Make global some frequently used functions
from numpy import dot
from numpy.linalg import norm as norm

#from scipy.cluster.vq import vq, kmeans, kmeans2, whiten
import Pycluster as pyc

#-----------------------------------------------------------------------------
# Main script
#-----------------------------------------------------------------------------


viz_fibers = False

# Load quadrature
qsph1_37_492DP = np.loadtxt('data/qsph1-37-492DP.dat')
quad_pnts = qsph1_37_492DP[:, :3]
#qsph1_44_672DP = np.loadtxt('data/qsph1-44-672DP.dat')
#quad_pnts = qsph1_44_672DP[:, :3]
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
n_fibers = 1                      # number of Gaussian components (max n=3)
b        = 4000                   # s/mm^2
#rangle  = -np.pi/2
#signal  =  np.zeros(n_sample_pnts)
#for i in range(n_sample_pnts):
#    signal[i] = sph.rand_sig(sample_pnts[i, :3].T, b, n_fibers, rangle)

SNR = []
nRealizations = 1
nAngles       = 1

angles_true  = np.zeros(nAngles)
angles_mean  = np.zeros(nAngles)
angles_spect = np.zeros((nRealizations,))
angles_kmean = np.zeros((nRealizations,))


for jj in range(nAngles):

  #Generate signal
  rangle  = -np.pi/2.0 
  signal  =  np.zeros(n_sample_pnts)
  angles_true[jj] = rangle * 180 / np.pi
 
  for i in range(n_sample_pnts):
    signal[i] = sph.rand_sig(sample_pnts[i, :3].T, b, n_fibers, rangle)




  for kk in range(nRealizations):
    print('Realization: %2g' % kk)
    # Make Rician noise
    sigma  = 0.0                       # standard deviation
    noiseR = sigma * np.random.randn(*signal.shape)
    noiseI = sigma * np.random.randn(*signal.shape)
    noise  = noiseR + 1j*noiseI

    SNR.append(10 * np.log10(norm(signal,2)/norm(noise,2)))
    print('Signal to noise ratio: %0.5g' % SNR[kk])

    # Add noise to signal
    rSig = signal + noise
    rSig = abs(rSig)                 #phase is not used in MRI


    # Choose regularization parameter
    # lambda > lambda_max -> zero solution
    lambda_max = 2*norm(dot(nA.T, rSig.T), np.inf) 

    lamb = 0.65*lambda_max
    
    print('Solving L1 penalized system with cvxpy...')

    coefs = cvx.variable(n_qpnts,1)
    A     = cvx.matrix(nA)
    rhs   = cvx.matrix(rSig).T

    objective = cvx.minimize(cvx.norm2(A*coefs - rhs) +
                             lamb*cvx.norm1(coefs) )
    constraints = [cvx.geq(coefs,0.0)]
    prob = cvx.program(objective, constraints)

    # Call the solver
    prob.solve(quiet=True)  #Use quiet=True to suppress output


    # Convert the cvxmod objects to plain numpy arrays for further processing
    nd_coefs_l1 = np.array(coefs.value).squeeze()

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
  
    #cond     = np.where(quad_pnts[:, 0] >= -quad_pnts[:, 1])
    cond  = np.where(quad_pnts[sortedIndex[:nSig], 0] >= -1.5)
    indexPos = sortedIndex[cond]
    points   = quad_pnts[indexPos, :3]
   
    coefs    = nd_coefs_l1_trim[indexPos]
    (ncoefs,) = coefs.shape
  


    args = (points,coefs,N)
    angles = np.zeros((nSig,2))
  
    for i in range(nSig):
      print('point: %0.5g' % i)
      x0 = sph.car2sph(points[i,0],points[i,1],points[i,2])[1:3]
      xopt = fmin_bfgs(sph.even_pODF_opt,x0,fprime=sph.even_pODF_opt_grad,args=args,gtol=1.0e-5)
      print('xopt theta: %0.5g' % xopt[0]) 
      print('xopt phi: %0.5g' % xopt[1]) 
      angles[i,:] = xopt

    mpoints = np.zeros_like(points)
    for i in range(nSig):
      mpoints[i,:] = np.array([np.sin(angles[i,0])*np.cos(angles[i,1]),np.sin(angles[i,0])*np.sin(angles[i,1]), np.cos(angles[i,0])])

    if viz_fibers == True:
      #---------------------------------------------------------------------------
      #--Visualize signal
      # Create a spherical mesh
      npts = 151
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
          k[i,j] = epODF.even_podf_f(np.array([x[i,j], y[i,j], z[i,j]]),points,coefs,N,ncoefs)
      s = k
      mlab.mesh(k*x, k*y, k*z, scalars=s, colormap='jet')
      mlab.show()


    #---------------------------------------------------------------------------
    #--Start clustering -- maybe need to use a different set of nodes to evaluate pODF
    #
    # Sample reconstructed pODF using rejection technique -- it's not strictly non-negative!!
    nsamples = 1000
    sampled_points = spODF.sample_podf_f(nsamples,N,points,coefs,ncoefs) 

    cond     = np.where(sampled_points[:, 0] >= -sampled_points[:, 1])
    c_points = sampled_points[cond, :][0]
    
    #im = np.where(c_points[:,3] > c_points[:,3].mean())
    #c_points = c_points[im,0:3][0]

    #k-means 
    nclusters = 2
    (indx,error,nf) = pyc.kcluster(c_points,nclusters=nclusters,npass=500,dist='u')
    (cdata,cmask) = pyc.clustercentroids(c_points,np.ones_like(c_points),indx)

    i_cl1 = np.where(indx == 0)
    i_cl2 = np.where(indx == 1)

    cl1   = c_points[i_cl1]
    cl2   = c_points[i_cl2]  

    cl1m  = np.array([cl1[:,0].mean(), cl1[:,1].mean(), cl1[:,2].mean()]) 
    cl1m  = cl1m / norm(cl1m)

    cl2m  = np.array([cl2[:,0].mean(), cl2[:,1].mean(), cl2[:,2].mean()]) 
    cl2m  = cl2m / norm(cl2m)
    
    angles_kmean[kk] = np.arccos(np.dot(cl1m,cl2m))*180.0/np.pi


    #Spectral clustering
    L = sph.laplacian(c_points,1.0)
    (ev, v) = la.eig(L)
    evals = np.abs(ev)
    index = evals.argsort()
    e_sorted = evals[index]
    n_evals  = e_sorted/e_sorted[-1]
 
    indx2 = np.where(np.abs(n_evals - 1) > 0.15 )
    (ncls,) = indx2[0].shape

    v_sorted = v[:,index]

    v0 = np.real(v_sorted[:,0])
    v1 = np.real(v_sorted[:,1])

    pos = np.where(v1 > 0.0)
    neg = np.where(v1 < 0.0)
  
    g1  = c_points[pos,:][0]
    g2  = c_points[neg,:][0]
    
    g1m = np.array([g1[:,0].mean(), g1[:,1].mean(), g1[:,2].mean()]) 
    g1m = g1m / norm(g1m)

    g2m = np.array([g2[:,0].mean(), g2[:,1].mean(), g2[:,2].mean()]) 
    g2m = g2m / norm(g2m)    
  
    angles_spect[kk] = np.arccos(np.dot(g1m,g2m))*180.0/np.pi
     
    

  #angles_mean[jj] = angles_real.mean() 
  #print('Mean angle: %5.5g' % angles_mean[jj])
  


