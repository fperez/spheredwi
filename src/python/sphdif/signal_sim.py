from __future__ import division
import numpy as np

diffusion_evals = np.array([1700e-6, 300e-6, 300e-6])

def single_tensor(gradients, bvals, S0=1, evals=None, rotation=None, SNR=None):
    """Simulated Q-space signal with a single tensor.

    Parameters
    -----------
    gradients : (N, 3) or (M, N, 3) ndarray
        Measurement gradients / directions, also known as b-vectors, as 3D unit
        vectors (either in a list or on a grid).
    bvals : (N,) array
        B-values for measurements.  The b-value is also ``b = \tau |q|^2``,
        where ``\tau`` is the time allowed for attenuation and ``q`` is the
        measurement position vector in signal-space (Q-space or Fourier-space).
        If b is too low, there is not enough attenuation to measure.  With b
        too high, the signal to noise ratio increases.
    S0 : double,
        Strength of signal in the presence of no diffusion gradient (also
        called the ``b=0`` value).
    evals : (3,) ndarray
        Eigenvalues of the diffusion tensor.  By default, values typical for
        prolate white matter are used.
    rotation : (3, 3) ndarray
        Rotation matrix for transforming the direction of the tensor.
    SNR : float
        Signal to noise ratio, assuming gaussian noise.  None implies no noise.

    Returns
    --------
    S : (N,) ndarray
        Simulated signal: ``S(q, tau) = S_0 e^(-b g^T R D R.T g)``.

    Notes
    -----
    Based on ``dipy.sims.voxel``.  TODO: Contribute back.

    References
    ----------
    .. [1] M. Descoteaux, "High Angular Resolution Diffusion MRI: from Local
           Estimation to Segmentation and Tractography", PhD thesis,
           University of Nice-Sophia Antipolis, p. 42, 2008.
    .. [2] E. Stejskal and J. Tanner, "Spin diffusion measurements: spin echos
           in the presence of a time-dependent field gradient", Journal of
           Chemical Physics, nr. 42, pp. 288--292, 1965.

    """
    if evals is None:
        evals = diffusion_evals

    if rotation is None:
        rotation = np.eye(3)

    out_shape = gradients.shape[:gradients.ndim - 1]

    gradients = gradients.reshape(-1, 3)
    R = np.asarray(rotation)
    S = np.zeros(len(gradients))
    D = R.dot(np.diag(evals)).dot(R.T)

    for (i, g) in enumerate(gradients):
        S[i] = S0 * np.exp(-bvals[i] * g.T.dot(D).dot(g))

    if SNR is not None:
        std = S0 / SNR
        S = S + np.random.randn(len(S)) * std

    return S.reshape(out_shape)

def single_tensor_ODF(r, evals=None, rotation=None):
    """Simulated ODF with a single tensor.

    Parameters
    ----------
    r : (N,3) or (M,N,3) ndarray
        Measurement positions in (x, y, z), either as a list or on a grid.
    evals : (3,)
        Eigenvalues of diffusion tensor.
    rotation : (3, 3) ndarray
        Rotation matrix to orient the diffusion tensor.

    Returns
    -------
    ODF : (N,) ndarray
        The diffusion probability at ``r`` after time ``tau``.

    References
    ----------
    .. [1] Aganj et al., "Reconstruction of the Orientation Distribution
           Function in Single- and Multiple-Shell q-Ball Imaging Within
           Constant Solid Angle", Magnetic Resonance in Medicine, nr. 64,
           pp. 554--566, 2010.

    """
    if evals is None:
        evals = diffusion_evals

    if rotation is None:
        rotation = np.eye(3)

    out_shape = r.shape[:r.ndim - 1]

    R = np.asarray(rotation)
    Di = np.linalg.inv(R.dot(np.diag(evals)).dot(R.T))
    r = r.reshape(-1, 3)
    P = np.zeros(len(r))
    for (i, u) in enumerate(r):
        P[i] = u.T.dot(Di).dot(u)**(3 / 2)

    return  (1 / (4 * np.pi * np.prod(evals)**1/2 * P)).reshape(out_shape)

if __name__ == "__main__":
    import sphere, coord, plot

    npts = 150
    theta, phi = sphere.mesh(npts)
    xyz = np.dstack(coord.sph2car(theta, phi))

    ODF = single_tensor_ODF(xyz, rotation=None)
    signal = single_tensor(gradients=xyz,
                           bvals=1000 * np.ones(npts * npts), rotation=None,
                           S0=1, SNR=None)

    plot.surf_grid_3D(ODF, theta, phi, scale_radius=True)
    plot.show()

    plot.surf_grid_3D(signal, theta, phi, scale_radius=True)
    plot.show()
