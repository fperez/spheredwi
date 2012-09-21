import numpy as np
import sys
sys.path.insert(0, '..')

from kernel_model_fmin_jn import SparseKernelModel
from kernel_model_fmin_jn import quadrature_points
from sphdif.linalg import rotation_around_axis

from dipy.sims.voxel import single_tensor
from numpy.linalg import norm as norm

from dipy.core.subdivide_octahedron import create_unit_hemisphere

from kernel_model_fmin_jn import even_kernel
import os
import Pycluster as pyc
from dipy.core.geometry import cart2sphere
from scipy.optimize import fmin

#Code trying to bring over even_pODF_opt from recon_fiber_cvxopt:

def even_pODF_opt(angles,*args): # qpoints, c, N):
    """Given the coefficients, evaluate model at a specific direction (theta,phi)


    Parameters
    ----------
     angles  = (theta,phi) -- polar angle, azimuthal angle
         N   = maximum degree of subspace
         c   = coefficients from minimization problem
     qpoints = quadrature points corresponding to coefficients c
    """

    qpoints = args[0]
    c       = args[1]
    N       = args[2]

    n,m = qpoints.shape

    theta,phi = angles[0], angles[1]
    omega = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])

    sum = 0.0
    for i in range(n):
      mu = np.dot(omega,qpoints[i,:])
      mu = np.clip(mu, -1.0, 1.0)

      sum += c[i]*even_kernel(mu, N)
    

    return -(N+1)**2 * sum
    
#---------------------------------------------------------------------------------

def two_fiber_signal(bvals, bvecs, angle, w=[0.5, 0.5], SNR=0):
    R0 = rotation_around_axis([0, 1, 0], 0)
    R1 = rotation_around_axis([0, 1, 0], np.deg2rad(angle))

    E = w[0] * single_tensor(gradients=bvecs, bvals=bvals, S0=1, evecs=R0, snr=SNR)
    E += w[1] * single_tensor(gradients=bvecs, bvals=bvals, S0=1, evecs=R1, snr=SNR)

    return E


from dipy.data import get_data
img, bvals, bvecs = get_data('small_64D')
bvals = np.load(bvals)
bvecs = np.load(bvecs)
where_dwi = bvals > 0
bvecs = bvecs[where_dwi]
bvals = bvals[where_dwi] * 3

tot_its = 1 #number of time 3 random measurements are discarded
nRealizations = 1 #number of MC simulations
sig_noise = 20 #SNR value

error = np.zeros((13,1))
mean_means = np.zeros((13,1))
mean_stds = np.zeros((13,1))
angles_clust = np.zeros((nRealizations,6))


for ii in range(0,tot_its):
	
	angles_mean = []
	angles_std = []
	
#-------------Code to randomly throw out one measurement-----------------------

	num_disc = 3 #Change this variable to adjust the number of measurements discarded
	
	if num_disc == 0:
		new_bvecs = bvecs
		new_bvals = bvals
	else:
		random_pnt = np.random.random_integers(1,62,num_disc)
		new_bvecs = np.zeros((64-num_disc)*3).reshape(64-num_disc,3)
		new_bvals = np.zeros(64-num_disc)
		print
		print "Measurement numbers thrown out:"

		for i in range(0,num_disc+1):
			if i==0:
				new_bvecs[0:random_pnt[i]-1,:] = bvecs[0:random_pnt[i]-1,:]
				new_bvals[0:random_pnt[i]-1] = bvals[0:random_pnt[i]-1]
				print random_pnt[i]
			elif i==num_disc:
				new_bvecs[random_pnt[i-1]-1:64-num_disc,:] = bvecs[random_pnt[i-1]+(i-1):64]
				new_bvals[random_pnt[i-1]-1:64-num_disc] = bvals[random_pnt[i-1]+(i-1):64]
			else:
				new_bvecs[random_pnt[i-1]-1:random_pnt[i]-1,:] = bvecs[random_pnt[i-1]+(i-1):random_pnt[i]+(i-1),:]
				new_bvals[random_pnt[i-1]-1:random_pnt[i]-1] = bvals[random_pnt[i-1]+(i-1):random_pnt[i]+(i-1)]
				print random_pnt[i]

		print

#------------------------------------------------------------------------------

	
	hsphere = create_unit_hemisphere(5)

	sk = SparseKernelModel(new_bvals, new_bvecs, sh_order=8)

	#angles = [50]
	angles = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

	SNR = None
	new_bvals = np.ones_like(new_bvals) * 3000

	cache = None
	odf_verts = hsphere.vertices

	for angle in angles:	
		
		print "Analyzing angle", angle
		
		#start MC simulation here:
		for kk in range(0,nRealizations):
			
			E = two_fiber_signal(new_bvals, new_bvecs, angle, SNR=SNR)
    
    			#create and add in Rician noise
    			noiseR = np.random.randn(*E.shape)
    			noiseI = np.random.randn(*E.shape)
    			noise = noiseR + 1j*noiseI
    			tau = (10.0**(-sig_noise/10.0)*norm(E,2)) #here we are using an snr of 15.0
    			noise = noise*(tau/norm(noise)) #normalize to get desired snr
    			E = E+noise #add noise to signal
    			E = abs(E) #phase is not used in MRI
    
    			fit, coefs = sk.fit(E)
			
			coefs_trim = np.array(coefs).squeeze()
			cutoff = coefs.mean() + 2.0*coefs.std(ddof=1)
			coefs_trim = np.where(coefs>cutoff, coefs, 0)

			q_theta, q_phi, q_points = quadrature_points(132)
			
			basedir = os.path.abspath(os.path.dirname(__file__))
			q_points = np.loadtxt(os.path.join(basedir, 'qsph1-16-132DP.dat'))
			q_points = q_points[:, :3]
			
			max_sub_deg = 8
			nSig = (coefs_trim>0).sum()
			sortedIndex = coefs_trim.argsort()[::-1]
			cond = np.where(q_points[sortedIndex[:nSig],0] >= -2.5)
			indexPos = sortedIndex[cond]
			points = q_points[indexPos,:3]
			coefs = coefs_trim[indexPos]
			
			args = (points, coefs,max_sub_deg)
			new_angle_test = np.zeros((nSig,2))
			mpoints = np.zeros_like(points)
			
			for ii in range(nSig):
				x0 = cart2sphere(points[ii,0],points[ii,1],points[ii,2])[1:3]
				xopt = fmin(even_pODF_opt,x0,args=args,xtol = 0.00001, ftol = 0.00001, disp=0)
				new_angle_test[ii,:] = xopt
				mpoints[ii,:] = np.array([np.sin(new_angle_test[ii,0])*np.cos(new_angle_test[ii,1]), np.sin(new_angle_test[ii,0])*np.sin(new_angle_test[ii,1]), np.cos(new_angle_test[ii,0])])
			
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
 
			#end MC simulation

		angles_clust.sort()  
  		angles_mean.append(angles_clust[:,0].mean()) 
  		angles_std.append(angles_clust[:,0].std())
	
	error += (np.abs(angles - np.array(angles_mean))).reshape((13,1))
	mean_means += (np.array(angles_mean)).reshape((13,1))
	mean_stds += (np.array(angles_std)).reshape((13,1))
		
		
angles_mean = mean_means/tot_its
angles_std = mean_stds/tot_its
error = error/tot_its
result = np.column_stack((angles, angles_mean, angles_std, error))
np.savetxt('result_fmin.out', result)

print 
print "     Angle in", "      Angle out (mean)", "    STD", "             Error"
print result

