"""Visualize kernel in ODF and signal space.
"""
from sphdif.kernel import even_kernel, inv_funk_radon_even_kernel
from dipy.viz import show_odfs
from dipy.core.subdivide_octahedron import create_unit_sphere

sphere = create_unit_sphere(6)

# sphere.z = np.dot([0, 0, 1], [x, y, z]) = cos(mu)
kernel_odf = even_kernel(sphere.z, N=8)
kernel_signal = inv_funk_radon_even_kernel(sphere.z, N=8)

#show_odfs([[[kernel_odf, kernel_signal]]], (sphere.vertices, sphere.faces))
show_odfs([[[kernel_odf]]], (sphere.vertices, sphere.faces))
