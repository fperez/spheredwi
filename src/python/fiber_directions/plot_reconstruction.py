import sys
sys.path.insert(0, '..')

import numpy as np
import glob
import os

from sphdif import sph_io
from sphdif import plot
from sphdif.kernel import (kernel_matrix, std_kernel, even_kernel,
                           inv_funk_radon_even_kernel, kernel_reconstruct)
from sphdif.coord import car2sph, sph2car

from dipy.sims.voxel import multi_tensor_odf
from dipy.reconst.shm import SlowAdcOpdfModel, MonoExpOpdfModel, QballOdfModel

coords = sph_io.load('sphere_pts.npz')

#from dipy.data import get_sphere
#verts, faces = get_sphere('symmetric724')

## from dipy.core.triangle_subdivide import create_unit_sphere
## verts, edges, sides = create_unit_sphere(5)
## faces = edges[sides, 0]

from dipy.core.triangle_subdivide import create_half_unit_sphere
verts, edges, sides = create_half_unit_sphere(5)
faces = edges[sides, 0]

theta, phi, r = car2sph(*verts.T)

sampling_xyz = np.column_stack(
    sph2car(coords['gradient_theta'], coords['gradient_phi'])
    )


## verts = np.column_stack(sph2car(theta, phi))
## from dipy.core.meshes import faces_from_vertices
## faces = faces_from_vertices(verts)

def separation_from_odf(odf):
    # Find angles
    from dipy.reconst.recspeed import local_maxima
    p, i = local_maxima(odf, edges)
    p = p[:2]
    i = i[:2]
    print "Peaks:", p
    print "Angular separation:", np.rad2deg(np.arccos(np.abs(np.dot(verts[i[0]], verts[i[1]]))))



ODFs = []
#m = SlowAdcOpdfModel(coords['b'], sampling_xyz, sh_order=8, odf_vertices=verts)
m = MonoExpOpdfModel(coords['b'], sampling_xyz, sh_order=8, odf_vertices=verts)

for k, fn in enumerate(sorted(glob.glob(os.path.join(sph_io.data_path,
                                                     'odf_coeffs_*.npz')))):
    data = sph_io.load(os.path.basename(fn))

    print "Quadrature model..."
    odf = kernel_reconstruct(coords['odf_theta'], coords['odf_phi'], data['beta'],
                             theta, phi,
                             kernel=even_kernel, N=data['kernel_N'])
    odf[odf < 0] = 0
    ODFs.append(odf)

    print "Actual angle:", np.rad2deg(data['separation'])
    separation_from_odf(odf)

    ODFs.append(multi_tensor_odf(verts, data['weights'], mevecs=data['mevecs']))

    print "Dipy model..."
    odf = m.evaluate_odf(data['signal'])
    ODFs.append(odf)
    separation_from_odf(odf)

N = 3
ODFs = np.array([[ODFs]]).reshape(len(ODFs) // N, 1, N, -1)

from dipy.viz import show_odfs, get_mlab

show_odfs(ODFs, (verts, faces), scale=2, radial_scale=True)
