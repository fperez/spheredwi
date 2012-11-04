import numpy as np
import sys
sys.path.insert(0, '..')

from kernel_model import SparseKernelModel
from kernel_model import quadrature_points
from sphdif.linalg import rotation_around_axis

from dipy.sims.voxel import single_tensor
from numpy.linalg import norm as norm

import os

# Number of Monte-Carlo simulations
realizations = 1

# Signal to noise ratio
sig_noise = 20

# Crossing angles
angles = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
                  dtype=float)

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

error = np.zeros_like(angles)
mean_means = np.zeros_like(angles)
mean_stds = np.zeros_like(angles)

from dipy.core.sphere import unit_icosahedron
sphere = unit_icosahedron.subdivide(5)


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

sk = SparseKernelModel(new_bvals, new_bvecs, sh_order=8,
                       alpha=0.00011, rho=0.9)
sk.direction_finder.config(sphere=sphere, min_separation_angle=10)

recovered_angle = []

SNR = None
new_bvals = np.ones_like(new_bvals) * 3000

for angle in angles:

    print "Analyzing angle", angle
    recovered_angle = []

    #start MC simulation here:
    for kk in range(0, realizations):

        E = two_fiber_signal(new_bvals, new_bvecs, angle, SNR=SNR)

        ## Create and add in Rician noise
        noiseR = np.random.randn(*E.shape)
        noiseI = np.random.randn(*E.shape)
        noise = noiseR + 1j * noiseI
        tau = (10.0**(-sig_noise / 10.0)*norm(E, 2))
        noise = noise * (tau / norm(noise)) # Normalize to get desired SNR
        E = E + noise # Add noise to signal
        E = abs(E) # Phase is not used in MRI

        fit = sk.fit(E)
        xyz = fit.directions

        if len(xyz) > 1:
            w = np.dot(xyz[0], xyz[1])
        else:
            w = 1

        recovered_angle.append(np.rad2deg(np.arccos(np.abs(w))))

        # End MC simulation

    recovered_angle.sort()
    angles_mean.append(np.array(recovered_angle).mean())
    angles_std.append(np.array(recovered_angle).std())

error += (np.abs(angles - np.array(angles_mean)))
mean_means += (np.array(angles_mean))
mean_stds += (np.array(angles_std))

result = np.column_stack((angles, angles_mean, angles_std, error))
np.savetxt('result.out', result)

np.set_printoptions(precision=2, suppress=True)
print
print "Angle in / Angle out (mean) / STD / Error"
print result
