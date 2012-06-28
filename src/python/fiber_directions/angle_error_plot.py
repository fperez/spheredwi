import numpy as np
import sys
sys.path.insert(0, '..')

from kernel_model import SparseKernelModel
from sphdif.linalg import rotation_around_axis

from dipy.reconst.recspeed import local_maxima
from dipy.sims.voxel import single_tensor

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

from dipy.core.triangle_subdivide import create_half_unit_sphere
verts, edges, sides = create_half_unit_sphere(6)
faces = edges[sides, 0]

sk = SparseKernelModel(bvals, bvecs, sh_order=8, odf_vertices=verts)

angles = [25, 30, 35, 40, 45, 50, 55, 60]
recovered_angle = []

for angle in angles:
    print "Analyzing angle", angle

    E = two_fiber_signal(bvals, bvecs, angle, SNR=None)
    odf = sk.fit(E).odf()
    odf = np.clip(odf, 0, None)

#    from dipy.viz import show_odfs
#    show_odfs([[[odf]]], (verts, faces))

    recovered_angle.append(angle_from_odf(odf, verts, edges))

result = np.column_stack((angles, recovered_angle))

print
print "Angle in", "Angle out"
print result
