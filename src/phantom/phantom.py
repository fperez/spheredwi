import numpy as np

from dipy.sims.phantom import orbital_phantom, add_rician_noise
from dipy.viz import fvtk

def f1(t):
    x = np.sin(t)
    y = np.linspace(-1, 1, len(x))
    z = y
    return x, y, z

def f2(t):
    x = -np.sin(t)
    y = np.linspace(-1, 1, len(x))
    z = y
    return x, y, z

def f3(t):
    x = np.linspace(-1, 1, len(t))
    y = np.sin(t)
    z = x
    return x, y, z


t = np.linspace(0, 2 * np.pi, 100)

# kissing fibre #1
vol = orbital_phantom(func=f1, t=t)

# kissing fibre #2
vol += orbital_phantom(func=f2, t=t)

# kissing fibre #3
vol += orbital_phantom(func=f3, t=t)

#vol = add_rician_noise(vol)

r=fvtk.ren()
fvtk.add(r, fvtk.volume(vol[...,0]))
fvtk.show(r)
