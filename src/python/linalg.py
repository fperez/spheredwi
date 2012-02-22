import numpy as np

def rotation_around_axis(v, theta):
    """Return the matrix that rotates 3D data an angle of theta around
    the axis v.

    Parameters
    ----------
    v : (3,) ndarray
        Axis of rotation.
    theta : float
        Angle of rotation in radians.

    References
    ----------
    http://en.wikipedia.org/wiki/Rodrigues'_rotation_formula

    """
    v = np.asarray(v)
    if not v.size == 3:
        raise ValueError("Axis of rotation should be 3D vector.")
    if not np.isscalar(theta):
        raise ValueError("Angle of rotation must be scalar.")

    v = v / np.linalg.norm(v)
    C = np.array([[ 0,   -v[2], v[1]],
                  [ v[2], 0,   -v[0]],
                  [-v[1], v[0], 0]])

    return np.eye(3) + np.sin(theta) * C + (1 - np.cos(theta)) * C.dot(C)
