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
realizations = 50

# Randomly discarded measurements
discarded = 3

# Signal to noise ratio
sig_noise = 5

# Crossing angles
angles = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
                  dtype=float)

# B-value
B = 3000

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
bvals = bvals[where_dwi]

bvals = np.ones_like(bvals) * B

error = np.zeros_like(angles)
mean_means = np.zeros_like(angles)
mean_stds = np.zeros_like(angles)

from dipy.core.sphere import unit_icosahedron
sphere = unit_icosahedron.subdivide(5)


angles_mean = []
angles_std = []

mask = np.zeros_like(bvals, dtype=bool)
mask[:discarded] = True

#------------------------------------------------------------------------------

recovered_angle = []

SNR = None

sk = SparseKernelModel(bvals, bvecs, sh_order=8, l1_ratio=0.5, alpha=0.0001)
sk.direction_finder.config(sphere=sphere, min_separation_angle=10)

recovered_angle = np.zeros((len(angles), realizations))

for n, angle in enumerate(angles):

    print "Angle %.2f, %d realizations, discarding %d measurements" % \
          (angle, realizations, discarded)

    # MC simulation loop
    for kk in range(0, realizations):
        E = two_fiber_signal(bvals, bvecs, angle, SNR=SNR)

        ## Create and add in Rician noise
        noiseR = np.random.random(E.shape)
        noiseI = np.random.random(E.shape)
        noise = noiseR + 1j * noiseI
        tau = (10.0**(-sig_noise / 10.0) * norm(E, 2))
        noise = noise * (tau / norm(noise)) # Normalize to get desired SNR

        E = E + noise # Add noise to signal
        E = abs(E) # Measured amplitude signal

        # Drop data
        np.random.shuffle(mask)
        E[mask] = np.nan

        fit = sk.fit(E)
        xyz = fit.directions

        if len(xyz) > 1:
            w = np.dot(xyz[0], xyz[1])
        else:
            w = 1

        recovered_angle[n, kk] = np.rad2deg(np.arccos(np.abs(w)))

        # End MC simulation


np.savetxt('_recovered_angle.npy', recovered_angle)

np.set_printoptions(precision=2, suppress=True)
print
print "Angle in / Angle out (mean) / STD / Error"

u = recovered_angle.mean(axis=1)
s = recovered_angle.std(axis=1)
e = np.abs(angles[:, None] - recovered_angle)
e_u = e.mean(axis=1)
e_s = e.std(axis=1)

print np.column_stack((angles, u, s, e_u))

import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.errorbar(angles, e_u, yerr=e_s)
ax.set_xlabel('Angle')
ax.set_ylabel('Reconstructed angle error')
ax.set_title('Reconstruction, %d iterations, %d discarded, SNR (power defn.) of %.2f' % (realizations, discarded, sig_noise))
plt.show()
