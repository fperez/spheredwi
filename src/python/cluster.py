"""Find clusters.

Run this file after running the recon_fiber script with %run -i in ipython.
"""

import itertools
from math import floor

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scikits.learn import mixture

import sphdif.sphquad as sph
reload(sph)

norm = np.linalg.norm

# Globals
X = points
n, m = X.shape

#cvtype = 'full'
#cvtype = 'spherical'
cvtype = 'diag'
#cvtype =  'tied'

# Try flipping the data in the lower hemisphere and folding it over
xneg_idx = X[:, 0] < 0
xpos_idx = np.logical_not(xneg_idx)
xfit = np.empty_like(X)
xfit[xneg_idx] = -(X[xneg_idx])
xfit[xpos_idx] = X[xpos_idx]
X = xfit
#X = X[xpos_idx]
n_states = 2

gmm = mixture.GMM(n_states=n_states, cvtype=cvtype)
gmm.fit(X)

splot = plt.subplot(111, aspect='equal')
color_iter = itertools.cycle (['r', 'g', 'b', 'c'])
marker_iter  = itertools.cycle(['o', 's', '^', 'd', 'p', 'h', '8'])

Y_ = gmm.predict(X)

for i, (mean, covar, color, marker) in \
        enumerate(zip(gmm.means, gmm.covars, color_iter, marker_iter)):
                      
    v, w = np.linalg.eigh(covar)
    u = w[0] / np.linalg.norm(w[0])
    plt.scatter(X[Y_==i, 0], X[Y_==i, 1], color=color, marker=marker, s=40)
    angle = np.arctan(u[1]/u[0])
    angle = 180 * angle / np.pi # convert to degrees
    ell = mpl.patches.Ellipse (mean, v[0], v[1], 180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    splot.grid(True)

plt.show()

print 'means\n', gmm.means
print 'covars\n', [np.diag(c) for  c in gmm.covars]
print 'angle', sph.angle(gmm.means[0], gmm.means[1], True)
