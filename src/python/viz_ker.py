"""Visualize inverse Funk-Radon transform elements with mayavi.
"""
from sphdif.kernel import kernel_plot, inv_funk_radon_even_kernel
from sphdif.plot import show

kernel_plot(inv_funk_radon_even_kernel, N=14)
show()
