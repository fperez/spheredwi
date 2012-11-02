import sys
sys.path.insert(0, '..')

import numpy as np

from sphdif import plot, coord

from dipy.core.subdivide_octahedron import create_unit_sphere
s = create_unit_sphere(3)

from mayavi import mlab
mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

mask = s.theta <= np.pi/2.
plot.scatter_3D(s.theta[mask], s.phi[mask], color=(0, 0, 1))

#ef = s.edges.ravel()
#for e in s.edges:
#    mlab.plot3d(s.x[e], s.y[e], s.z[e], tube_radius=None)

N = 20
mlab.quiver3d([s.x[N]], [s.y[N]], [s.z[N]], color=(1, 0, 0), mode='2darrow')

plot.show()
