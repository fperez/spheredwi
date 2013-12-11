"""
=========================================================
Reconstruct with Sparse Model (adapted from DiPy example)
=========================================================
"""

import numpy as np
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from kernel_model import SparseKernelModel

# Download Stanford data
fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

# Construct model
model = SparseKernelModel(gtab, sh_order=6, loglog_tf=False, l1_ratio=0.99, alpha=0.0005)
#model = CsaOdfModel(gtab, 4)
#model = ConstrainedSphericalDeconvModel(gtab, None)

"""
img contains a nibabel Nifti1Image object (data) and gtab contains a GradientTable
object (gradient information e.g. b-values). For example to read the b-values
it is possible to write print(gtab.bvals).

Load the raw diffusion data and the affine.
"""

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
Remove most of the background using dipy's mask module.
"""

# Instantiate model
from dipy.reconst.multi_voxel import multi_voxel_fit
SparseKernelModel.fit = multi_voxel_fit(SparseKernelModel.fit)

#from dipy.segment.mask import median_otsu
#maskdata, mask = median_otsu(data, 3, 1, True,
#                             vol_idx=range(10, 50), dilate=2)

# Select segment of data to work with
data_small = data[13:43, 44:74, 28:29]

fit = model.fit(data_small)

sphere = get_sphere('symmetric724')
odfs = fit.odf(sphere)

# It is common for some models to produce negative values, we can remove those
# using ``np.clip``
odfs = np.clip(odfs, 0, np.max(odfs, -1)[..., None])

# Visualie

from dipy.viz import fvtk
r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(odfs, sphere, colormap='jet'))

fn = model.__class__.__name__ + '_odfs.png'
print('Saving illustration as %s' % fn)
fvtk.record(r, n_frames=1, out_path=fn, size=(600, 600))
