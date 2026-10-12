"""Core utilities for the jPCA / traveling-wave demo notebooks.

Implements the Churchland et al. 2012 pipeline (preprocessing -> PCA -> jPCA)
plus synthetic data generators used to demonstrate:
  1. what jPCA does on motor-cortex-like population data,
  2. why pure sequential / peak-time-wave structure also yields rotations
     (Lebedev et al. 2019; Kuzmina, Kriukov & Lebedev 2024),
  3. the standing-vs-traveling wave discriminator in physical space.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# preprocessing (Churchland et al. 2012 style)
# ---------------------------------------------------------------------------

def preprocess(rates, bin_ms=10, smooth_sd_ms=20, epsilon=5.0,
               subtract_condition_mean=True):
    """rates: (n_neurons, n_conditions, n_times) firing rates (spikes/s).

    Steps, following Churchland et al. 2012:
      1. Gaussian smoothing in time (firing-rate estimate).
      2. Soft-normalization per neuron: divide by (range + epsilon) so that
         high-rate neurons do not dominate the covariance.
      3. Subtract the cross-condition mean per neuron, focusing the analysis
         on condition-dependent structure.
    """
    rates = np.asarray(rates, dtype=float)
    if smooth_sd_ms > 0:
        rates = gaussian_filter1d(rates, smooth_sd_ms / bin_ms, axis=2,
                                  mode='nearest')
    spread = rates.max(axis=(1, 2)) - rates.min(axis=(1, 2))
    rates = rates / (spread + epsilon)[:, None, None]
    if subtract_condition_mean:
        rates = rates - rates.mean(axis=1, keepdims=True)
    return rates


def stack_conditions(rates):
    """(N, C, T) -> (N, C*T): concatenate conditions along time."""
    n, c, t = rates.shape
    return rates.reshape(n, c * t)


def unstack(X, n_conditions):
    """(D, C*T) -> (D, C, T)."""
    d, ct = X.shape
    return X.reshape(d, n_conditions, ct // n_conditions)


# ---------------------------------------------------------------------------
# jPCA
# ---------------------------------------------------------------------------

def jpca(rates, n_pcs=6, dt_s=0.01, refine_skew=True, align_axes=True):
    """Run PCA then jPCA on preprocessed rates (N, C, T).

    Fits dX/dt = M X by least squares, keeps the skew-symmetric part
    (purely rotational flow), and extracts rotation planes from the
    imaginary-conjugate eigenvalue pairs of M_skew. Defaults follow the
    original Churchland et al. 2012 toolbox (and the bantin Python port).

    refine_skew: if True (default), M_skew is the exact constrained
        least-squares fit (conjugate gradients over skew-symmetric
        matrices), as in the original jPCA MATLAB toolbox. If False, the
        closed-form skew projection (M - M^T)/2 of the unconstrained fit
        is used (this is the initialization of the constrained fit).
    align_axes: if True (default), rotate the basis within each jPC plane
        so the across-condition distribution of initial (first-bin, e.g.
        preparatory) states lies along the plane's first axis — the display
        convention of the original toolbox (align_jpcs in bantin's port).
        Pure in-plane rotation: plane variance and fit R^2 are unchanged.
        Needs >= 2 conditions; otherwise skipped.

    Returns a dict with the fitted model, R^2 of the rotational fit,
    eigenvalues, orthonormal plane bases (n_pcs/2 planes), and rotation
    frequencies in Hz.
    """
    n, c, t = rates.shape
    n_pcs = min(n_pcs, n)
    X = stack_conditions(rates)                       # (N, C*T)
    pca = PCA(n_components=n_pcs)
    Z = pca.fit_transform(X.T).T                      # (n_pcs, C*T)

    dZ = np.diff(Z, axis=1)                           # (n_pcs, C*T-1)
    Z0 = Z[:, :-1]
    if refine_skew:
        M_skew = _skew_sym_regress(Z0.T, dZ.T)
    else:
        M_full = dZ @ np.linalg.pinv(Z0)              # unconstrained fit
        M_skew = 0.5 * (M_full - M_full.T)            # project onto skew-symmetric
    resid = dZ - M_skew @ Z0
    r2 = 1.0 - np.sum(resid ** 2) / np.sum(dZ ** 2)

    eigvals, eigvecs = np.linalg.eig(M_skew)
    planes, freqs_hz = [], []
    for i in np.argsort(-eigvals.imag):               # one eigenvector per conjugate pair
        if eigvals[i].imag <= 1e-10:
            break                                     # remaining pairs are ~zero
        v = eigvecs[:, i]
        Q, _ = np.linalg.qr(np.column_stack([v.real, v.imag]))
        planes.append(Q)
        freqs_hz.append(eigvals[i].imag / (2 * np.pi * dt_s))
    if align_axes and c >= 2:
        planes = _align_planes_to_start(planes, Z, c)
    return dict(pca=pca, Z=Z, M_skew=M_skew, r2=r2, eigvals=eigvals,
                planes=planes, freqs_hz=freqs_hz, n_conditions=c, dt_s=dt_s)


def _skew_sym_regress(X, X_dot, tol=1e-4):
    """Exact constrained skew-symmetric least squares: minimize
    ||X_dot - X @ M.T||_F over skew-symmetric M by conjugate gradients,
    initialized from the closed-form skew projection of the unconstrained
    fit. Port of skew_sym_regress in the bantin jPCA package (which mirrors
    the original MATLAB toolbox). X, X_dot: (samples, dims)."""
    from scipy.optimize import minimize
    D = X.shape[1]
    iu = np.triu_indices(D, k=1)
    M_lstq = np.linalg.lstsq(X, X_dot, rcond=None)[0].T
    M0 = 0.5 * (M_lstq - M_lstq.T)

    def vec2mat(h):
        M = np.zeros((D, D))
        M[iu] = h
        return M - M.T

    def objective(h):
        M = vec2mat(h)
        return 0.5 * np.linalg.norm(X @ M.T - X_dot, 'fro') ** 2

    def gradient(h):
        M = vec2mat(h)
        dM = (X.T @ X @ M.T) - (X.T @ X_dot)
        return (dM.T - dM)[iu]

    out = minimize(objective, M0[iu], jac=gradient, method='CG',
                   options=dict(maxiter=10000, gtol=tol))
    return vec2mat(out.x)


def _align_planes_to_start(planes, Z, n_conditions):
    """Rotate the basis within each jPC plane so the across-condition
    distribution of initial (first-bin, e.g. preparatory) states lies along
    the plane's first axis, with the dominant second-axis excursion positive.
    Port of align_jpcs from the bantin jPCA package (itself mirroring the
    MATLAB toolbox). Pure in-plane rotation: variance and fit quality are
    unchanged. planes: list of (D, 2) bases; Z: (D, C*T)."""
    Zc = unstack(Z, n_conditions)
    aligned = []
    for Q in planes:
        coords = np.einsum('ij,jct->ict', Q.T, Zc)
        prep = coords[:, :, 0].T                      # (C, 2) initial states
        rot = PCA(n_components=2).fit(prep).components_
        if np.linalg.det(rot) < 0:                    # keep the basis right-handed
            rot[1] = -rot[1]
        new = np.einsum('ij,jct->ict', rot, coords)
        if np.abs(new[1]).max() > new[1].max():
            rot = -rot
        aligned.append(Q @ rot.T)
    return aligned


def project_planes(res):
    """Project the reduced trajectories onto each jPC plane.

    Returns a list of (trajectories, frac_var, r2_in_plane), ordered by the
    variance captured by each plane. trajectories has shape (2, C, T);
    r2_in_plane is the skew-symmetric fit restricted to that 2-D plane — the
    honest rotation metric when other latent dimensions are translational.
    """
    Z, C = res['Z'], res['n_conditions']
    total = np.sum(Z ** 2)
    out = []
    for Q in res['planes']:
        coords = Q.T @ Z                              # (2, C*T)
        dz = np.diff(coords, axis=1)
        z0 = coords[:, :-1]
        m = dz @ np.linalg.pinv(z0)
        m = 0.5 * (m - m.T)
        r2 = 1.0 - np.sum((dz - m @ z0) ** 2) / np.sum(dz ** 2)
        out.append((unstack(coords, C), np.sum(coords ** 2) / total, r2))
    out.sort(key=lambda x: -x[1])
    return out


def start_ordered_colors(coords, cmap_name='RdBu'):
    """Per-condition colors ordered by the starting jPC1 coordinate of each
    trajectory — the coloring convention of the original jPCA MATLAB toolbox
    (and the bantin Python port's plot_projections): trajectories are colored
    by where their initial (preparatory) state sits in the rotation plane,
    not by arbitrary condition index. coords: (2, C, T). Returns (C, 4) RGBA.
    """
    import matplotlib.pyplot as plt
    C = coords.shape[1]
    order = np.argsort(coords[0, :, 0])
    cols = np.empty((C, 4))
    cols[order] = plt.get_cmap(cmap_name)(np.linspace(0, 1, C))
    return cols


def rotation_r2(rates, n_pcs=6, dt_s=0.01):
    """Headline 'how rotational is this dataset' number: R^2 of the
    skew-symmetric fit dX/dt = M_skew X in the top-PCA subspace."""
    return jpca(rates, n_pcs=n_pcs, dt_s=dt_s)['r2']


# ---------------------------------------------------------------------------
# synthetic data generators
# ---------------------------------------------------------------------------

def simulate_reaching(n_neurons=150, n_conditions=8, T=120, bin_ms=10,
                      f_rot_hz=2.0, decay_tau_s=1.2, seed=0):
    """Churchland-style synthetic motor cortex population.

    Per condition, latent dynamics = 2 rotational dimensions (a slowly decaying
    oscillation starting at movement onset; shared plane and direction across
    conditions, strongly condition-dependent radius) + 2 weak condition-tuned
    ramp dimensions. Neurons are random linear mixtures of the latents.

    Returns rates (N, C, T) in spikes/s, reach directions, and time axis (s).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T) * bin_ms / 1000.0
    dirs = np.linspace(0, 2 * np.pi, n_conditions, endpoint=False)
    lat = np.zeros((4, n_conditions, T))
    move = t >= 0.2
    tm = np.clip(t - 0.2, 0, None)
    # steady oscillation during movement: radius decays slowly from its
    # preparatory value; phase advances at f_rot_hz
    env = np.zeros(T)
    env[move] = np.exp(-tm[move] / 2.5)
    prep = 0.5 * (1 - np.cos(np.pi * np.clip(t / 0.2, 0, 1))) * (~move)
    for c in range(n_conditions):
        # strong radius modulation across conditions (survives condition-mean
        # subtraction); sign flips are harmless -- the flow is linear.
        amp = 0.4 + 1.6 * abs(np.cos(dirs[c]))
        # preparatory state: a condition-specific point on a ring -- the
        # "initial condition" that sets radius and entry phase of the rotation
        px, py = amp * np.cos(dirs[c]), amp * np.sin(dirs[c])
        rot_x = amp * env * np.cos(2 * np.pi * f_rot_hz * tm + dirs[c])
        rot_y = amp * env * np.sin(2 * np.pi * f_rot_hz * tm + dirs[c])
        lat[0, c] = np.where(move, rot_x, prep * px)
        lat[1, c] = np.where(move, rot_y, prep * py)
        lat[2, c] = 0.3 * np.cos(dirs[c]) * tm        # weak direction-tuned ramps
        lat[3, c] = 0.3 * np.sin(dirs[c]) * tm
    W = rng.standard_normal((n_neurons, 4)) / 2
    rates = 20 + 15 * (W @ lat.reshape(4, -1)).reshape(n_neurons, n_conditions, T)
    rates += 0.5 * rng.standard_normal(rates.shape)
    return np.clip(rates, 0.1, None), dirs, t


def simulate_sequences(n_neurons=200, n_conditions=8, T=120, bin_ms=10,
                       sigma_ms=30, spacing_ms=3.0, gain_vary=True, seed=1):
    """Pure feedforward sequences: neuron j fires one Gaussian bump at time
    tau_j = spacing_ms * j. There is no recurrent dynamics and no attractor
    anywhere in this generator. Conditions differ only in overall gain
    (or sequence speed if spacing_jitter is used downstream).

    Returns rates (N, C, T) and time axis in ms.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T) * bin_ms                          # ms
    tau = 100.0 + spacing_ms * np.arange(n_neurons)    # fixed recruitment order
    rates = np.zeros((n_neurons, n_conditions, T))
    for c in range(n_conditions):
        gain = 0.6 + 0.5 * c / max(n_conditions - 1, 1) if gain_vary else 1.0
        rates[:, c] = gain * 25.0 * np.exp(-(t[None, :] - tau[:, None]) ** 2
                                           / (2 * sigma_ms ** 2))
    rates += 1.5 * rng.standard_normal(rates.shape)
    return np.clip(rates + 2.0, 0.0, None), t


def simulate_physical_wave(n_neurons=120, T=400, dt_s=0.001, freq_hz=10.0,
                          k_rad_per_cell=0.2, kind='traveling', seed=3):
    """Neurons arranged on a 1D physical line (positions 0..n-1), oscillating.

    traveling: r(x,t) = 1 + cos(2*pi*f*t - k*x)   -- phase gradient across space
    standing:  r(x,t) = (1 + cos(2*pi*f*t)) * (0.5 + 0.5*cos(k*x))  -- separable,
               one spatial envelope, one global temporal phase (rank-1 field)

    Returns rates (N, 1, T), positions, and time axis (s).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T) * dt_s
    x = np.arange(n_neurons)
    phase_t = 2 * np.pi * freq_hz * t
    if kind == 'traveling':
        field = 1.0 + np.cos(phase_t[None, :] - k_rad_per_cell * x[:, None])
    elif kind == 'standing':
        spatial = 0.5 + 0.5 * np.cos(k_rad_per_cell * x)
        field = (1.0 + np.cos(phase_t))[None, :] * spatial[:, None]
    else:
        raise ValueError(kind)
    rates = 10.0 * field + 0.5 * rng.standard_normal(field.shape)
    return np.clip(rates, 0.0, None)[:, None, :], x, t


def ellipticity(rates_probe, subtract_mean=True):
    """Standing-vs-traveling meter from the shape of the state-space loop.

    PCA to 2D on the probe neurons, then ratio of minor-to-major axis of the
    trajectory covariance: ~0 = line (standing wave), ~1 = circle (clean
    traveling wave / quadrature pair).
    """
    X = stack_conditions(rates_probe)
    if subtract_mean:
        X = X - X.mean(axis=1, keepdims=True)
    Z = PCA(n_components=2).fit_transform(X.T).T
    s = np.linalg.svd(np.cov(Z), compute_uv=False)
    return float(np.sqrt(max(s[1], 0.0) / s[0]))


def phase_vs_position(rates_probe, freq_hz, dt_s):
    """Dominant-cycle phase of each probe neuron (FFT phase at freq_hz),
    unwrapped, versus physical position. A traveling wave gives a line whose
    slope is the spatial phase gradient; a standing wave gives a flat line.
    Returns positions, phases (rad)."""
    X = stack_conditions(rates_probe)
    X = X - X.mean(axis=1, keepdims=True)
    n, T = X.shape
    freqs = np.fft.rfftfreq(T, dt_s)
    i0 = np.argmin(np.abs(freqs - freq_hz))
    phases = np.angle(np.fft.rfft(X, axis=1)[:, i0])
    return np.arange(n), np.unwrap(phases)
