"""Make all figures for hbm2011 poster.
"""

import matplotlib.pyplot as plt

import dwicoverage as cov
reload(cov)

vmin = 0.45

cov.main_coverage('qsph1-14-72DP.dat', vmin=vmin)
plt.savefig('../../posters/hbm-2011/coverage_72_full_range.eps')

cov.main_coverage('bvecs', symm=True, vmin=vmin)
plt.savefig('../../posters/hbm-2011/coverage_64_full_range.eps')


plt.show()
