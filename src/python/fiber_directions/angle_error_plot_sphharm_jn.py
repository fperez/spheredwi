import numpy as np
import sys
sys.path.insert(0, '..')

from kernel_model_sphharm_jn import SparseKernelModel
from kernel_model_sphharm_jn import quadrature_points
from sphdif.linalg import rotation_around_axis

from dipy.sims.voxel import single_tensor
from numpy.linalg import norm as norm

from kernel_model_sphharm_jn import even_kernel
import os
import Pycluster as pyc

#------My code changing local_maxima code format from cython to python------
def local_maxima(codf,cedges):
    """Given a function, odf, and neighbor pairs, edges, finds the local maxima

    If a function is evaluated on some set of points where each pair of
    neighboring points is in edges, the function compares each pair of
    neighbors and returns the value and location of each point that is >= all
    its neighbors.

    Parameters
    ----------
    odf : array_like
        The odf of some function evaluated at some set of points
    edges : array_like (N, 2)
        every edges(i,:) is a pair of neighboring points

    Returns
    -------
    peaks : ndarray
        odf at local maximums, orders the peaks in descending order
    inds : ndarray
        location of local maximums, indexes to odf array so that
        odf[inds[i]] == peaks[i]

    Note
    ----
    Comparing on edges might be faster then comparing on faces if edges does
    not contain repeated entries. Additionally in the event that some function
    is symmetric in some way, that symmetry can be exploited to further reduce
    the domain of the search and the number of input edges. This is done in the
    create_half_unit_sphere function of dipy.core.triangle_subdivide for
    functions with antipodal symmetry.

    See Also
    --------
    create_half_unit_sphere

    """

    np.ones(len(codf))
    lenedges = len(cedges)
    cpeak = np.ones(len(codf),'uint8')
   
    for i in range(lenedges):

        find0 = cedges[i,0]
        find1 = cedges[i,1]
        odf0 = codf[find0]
        odf1 = codf[find1]

        if odf0 > odf1:
            cpeak[find1] = 0
        elif odf0 < odf1:
            cpeak[find0] = 0
        elif (odf0 != odf0) or (odf1 != odf1):
            raise ValueError("odf cannot have nans")

    peakidx = cpeak.nonzero()[0]
    peakvalues = codf[peakidx]
    order = peakvalues.argsort()[::-1]
    return peakvalues[order], peakidx[order]
    
#---------Back to original code------------------------------------------------

def two_fiber_signal(bvals, bvecs, angle, w=[0.5, 0.5], SNR=0):
    R0 = rotation_around_axis([0, 1, 0], 0)
    R1 = rotation_around_axis([0, 1, 0], np.deg2rad(angle))

    E = w[0] * single_tensor(gradients=bvecs, bvals=bvals, S0=1, evecs=R0, snr=SNR)
    E += w[1] * single_tensor(gradients=bvecs, bvals=bvals, S0=1, evecs=R1, snr=SNR)

    return E


def angle_from_odf(odf, verts, edges):
    # Find angles
    p, i = local_maxima(odf, edges)

    mask = p > 0
    p = p[mask]
    i = i[mask]

    if len(p) < 2:
        return 0

    w = np.dot(verts[i[0]], verts[i[1]])
    return np.rad2deg(np.arccos(np.abs(w)))


from dipy.data import get_data
img, bvals, bvecs = get_data('small_64D')
bvals = np.load(bvals)
bvecs = np.load(bvecs)
where_dwi = bvals > 0
bvecs = bvecs[where_dwi]
bvals = bvals[where_dwi] * 3

arr_len = 13 #13 to test all angles, 1 to test 1 angle

error = np.zeros((arr_len,1))
mean_means = np.zeros((arr_len,1))
mean_stds = np.zeros((arr_len,1))

tot_its = 1 #number of time 3 random measurements are discarded
nRealizations = 1 #number of MC simulations
sig_noise = 20 #SNR value

for ii in range(0,tot_its):

	angles_mean = []
	angles_std = []

#-------------Code to randomly throw out one measurement-----------------------

	num_disc = 0 #Change this variable to adjust the number of measurements discarded
	
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

	from dipy.core.subdivide_octahedron import create_unit_hemisphere
	hsphere = create_unit_hemisphere(5)

	sk = SparseKernelModel(new_bvals, new_bvecs, sh_order=8)

	if arr_len == 1:
		angles = [50]
	else:
		angles = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
	
	recovered_angle = []

	SNR = None
	new_bvals = np.ones_like(new_bvals) * 3000

	cache = None
	odf_verts = hsphere.vertices

	for angle in angles:	
		
		#print "Analyzing angle", angle
		recovered_angle = []
		
		#start MC simulation here:
		for kk in range(0,nRealizations):
			
			E = two_fiber_signal(new_bvals, new_bvecs, angle, SNR=SNR)
    
    			#create and add in Rician noise
    			noiseR = np.random.randn(*E.shape)
    			noiseI = np.random.randn(*E.shape)
    			noise = noiseR + 1j*noiseI
    			tau = (10.0**(-sig_noise/10.0)*norm(E,2)) 
    			noise = noise*(tau/norm(noise)) #normalize to get desired snr
    			E = E+noise #add noise to signal
    			E = abs(E) #phase is not used in MRI
    
    			fit = sk.fit(E)
			
			#Pass out odf matrix, beta, and intercept matrices before computation of 
			#odf in order to compute FA and only consider portion of matrix 
			#if necessary
			 
			
    			odf_mat, beta, intercept= fit.odf(vertices=odf_verts, cache=cache)
			
			
			FA = (np.linalg.norm(beta[0:132])**2)/((np.linalg.norm(beta[132:138])**2)+(np.linalg.norm(beta[0:132])**2))
			#print "Fractional Anisotropy:", FA 
			
			#if FA >= 1:
				#odf = np.dot(odf_mat[:,0:132],beta[0:132])+intercept
			#elif FA < 1:
				#odf = np.dot(odf_mat[:,132:138],beta[132:138])+intercept
			
			odf = np.dot(odf_mat,beta) #+ intercept
			#odf = np.clip(odf, 0, None)
    			#odf = np.abs(odf)

    			# Use cache from now on
    			cache = fit
    			odf_verts = None

			#from dipy.viz import show_odfs
			#show_odfs([[[odf]]], (verts, faces))
			
    			recovered_angle.append(angle_from_odf(odf,hsphere.vertices, hsphere.edges))
			
			#end MC simulation
		
		recovered_angle.sort()
		angles_mean.append(np.array(recovered_angle).mean())
		angles_std.append(np.array(recovered_angle).std())
		
	error += (np.abs(angles - np.array(angles_mean))).reshape((arr_len,1))
	mean_means += (np.array(angles_mean)).reshape((arr_len,1))
	mean_stds += (np.array(angles_std)).reshape((arr_len,1))
		


angles_mean = mean_means/tot_its
angles_std = mean_stds/tot_its
error = error/tot_its
result = np.column_stack((angles, angles_mean, angles_std, error))
np.savetxt('result.out', result)

print
print "     Angle in", "      Angle out (mean)", "    STD", "             Error"
print result

#print beta
