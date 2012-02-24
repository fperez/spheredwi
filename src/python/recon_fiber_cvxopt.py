#!/usr/bin/env python
"""Fiber reconstruction example script.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function, division

try:
    from mayavi import mlab
except ImportError:
    from enthought.mayavi import mlab

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import cvxpy as cvx
from scipy.optimize import fmin_bfgs, fmin

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
#qsph1_37_492DP = np.loadtxt('data/qsph1-37-492DP.dat')
#quad_pnts = qsph1_37_492DP[:, :3]
#N = 18         # maximum degree of subspace
#n_qpnts = 492  # number of points in quadrature
#
qsph1_19_132DP = np.loadtxt('data/qsph1-19-132DP.dat')
quad_pnts = qsph1_19_132DP[:, :3]
N = 9          # maximum degree of subspace
n_qpnts = 132  # number of points in quadrature



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
qsph1_14_72DP = np.loadtxt('data/qsph1-14-72DP.dat')
sample_pnts  = qsph1_14_72DP[:, :3]
n_sample_pnts = 72

#sample_pnts_up_hem = sample_pnts[(np.where(sample_pnts[:,2]>0))]
#n_sample_pnts,nn = sample_pnts_up_hem.shape
#sample_pnts = sample_pnts_up_hem


# Create reproducing-kernel (sparse representation) matrix
nA = sph.interp_matrix_new(quad_pnts, sample_pnts, n_qpnts, n_sample_pnts, N)

# Create signal
n_fibers = 2                      # number of Gaussian components (max n=3)
b        = 4000                   # s/mm^2

SNR = []
nRealizations = 20
nAngles       = 3

angles_true  = np.zeros(nAngles + 1)
angles_mean  = np.zeros(nAngles + 1)
angles_std   = np.zeros(nAngles + 1)
angles_spect = np.zeros((nRealizations,))
angles_clust = np.zeros((nRealizations,6))



for jj in range(nAngles,-1,-1): #range(nAngles + 1):

  #Generate signal
  rangle  =  np.arccos(1.0/2.0**0.25) + (np.pi/4.0)*(jj/(1.0*nAngles))   
  signal  =  np.zeros(n_sample_pnts)
  #angles_true[jj] = rangle * 180 / np.pi
  
  print('\n')
  print('Creating signal...')
  for i in range(n_sample_pnts):
    (angles_true[jj],signal[i]) = sph.rand_sig(sample_pnts[i, :3].T, b, n_fibers, rangle)


  s_energy = norm(signal,2)
  snr = 20.0
  tau = 10.0**(-snr/10.0) * s_energy
  print('Signal to noise ratio: %0.5g' % snr)


  #Start Monte Carlo simulations
  for kk in range(nRealizations):
    print('\n')
    print('Realization: %2g' % kk)

    # Make Rician noise
    noiseR = np.random.randn(*signal.shape)
    noiseI = np.random.randn(*signal.shape)
    noise  = noiseR + 1j*noiseI
    
    # Normalize to get desired SNR
    noise = noise*(tau/norm(noise))
   
    # Add noise to signal
    rSig = signal + noise
    rSig = abs(rSig)                 #phase is not used in MRI


    # Choose regularization parameter
    # lambda > lambda_max -> zero solution
    lambda_max = 2*norm(dot(nA.T, rSig.T), np.inf) 

    lamb = 1.0e-8*lambda_max
    
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
    cutoff =  nd_coefs_l1.mean() + 2.0*nd_coefs_l1.std(ddof=1)
    nd_coefs_l1_trim = np.where(nd_coefs_l1 > cutoff, nd_coefs_l1, 0)

    # Get indices needed for sorting coefs, in reverse order.
    sortedIndex = nd_coefs_l1_trim.argsort()[::-1]
    # number of significant coefficients
    nSig = (nd_coefs_l1_trim > 0).sum()
    print('Precentage compression: %0.5g' % (100*(1.00 - (nSig/(1.0*n_qpnts)))))

    # Used for taking only some of the points---now using the whole sphere
    # Let -1.5 -> 0 and get only the hemisphere with x>0
  
    #cond     = np.where(quad_pnts[:, 0] >= -quad_pnts[:, 1])
    cond      = np.where(quad_pnts[sortedIndex[:nSig], 0] >= -2.5)
    indexPos  = sortedIndex[cond]
    points    = quad_pnts[indexPos, :3]
   
    coefs     = nd_coefs_l1_trim[indexPos]
    (ncoefs,) = coefs.shape
  

    print('Start maximization step...')
    #Start maximization process
    args = (points,coefs,N)
    angles = np.zeros((nSig,2))  
    for i in range(nSig):
      x0 = sph.car2sph(points[i,0],points[i,1],points[i,2])[1:3]
#      xopt = fmin_bfgs(sph.even_pODF_opt,x0,fprime=sph.even_pODF_opt_grad,args=args,gtol=1.0e-4,maxiter=100)
      xopt = fmin(sph.even_pODF_opt,x0,args=args, xtol = 0.00001, ftol = 0.00001,disp=0)
      angles[i,:] = xopt

    #convert back to (x,y,z)
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



    print('Start clustering step...')
    #---------------------------------------------------------------------------
    #--Start clustering -- maybe need to use a different set of nodes to evaluate pODF
    #
    nclusters = 4
    (indx,error,nf) = pyc.kcluster(mpoints,nclusters=nclusters,npass=500,dist='u')
    (cmdata,cmmask) = pyc.clustercentroids(mpoints,np.ones_like(mpoints),indx)
    
    angles_clust[kk,0] = np.arccos(np.dot(cmdata[0,:],cmdata[1,:]))*180/np.pi
    angles_clust[kk,1] = np.arccos(np.dot(cmdata[0,:],cmdata[2,:]))*180/np.pi
    angles_clust[kk,2] = np.arccos(np.dot(cmdata[0,:],cmdata[3,:]))*180/np.pi
    angles_clust[kk,3] = np.arccos(np.dot(cmdata[1,:],cmdata[2,:]))*180/np.pi
    angles_clust[kk,4] = np.arccos(np.dot(cmdata[1,:],cmdata[3,:]))*180/np.pi
    angles_clust[kk,5] = np.arccos(np.dot(cmdata[2,:],cmdata[3,:]))*180/np.pi
    print('Done with realization.')





  angles_clust.sort()  
  angles_mean[jj] = angles_clust[:,0].mean() 
  angles_std[jj]  = angles_clust[:,0].std() 
   


  print('\n')
  print('Mean angle: %5.5g' % angles_mean[jj])
  print('Std dev   : %5.5g' % angles_std[jj])
  print('True angle: %5.5g' % angles_true[jj])
  print('\n')
  
    

