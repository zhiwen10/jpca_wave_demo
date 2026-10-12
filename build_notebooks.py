# -*- coding: utf-8 -*-
"""Builds the demo notebooks (with markdown + code cells) using nbformat.
Run once with the iblwave env python; then execute the notebooks via
jupyter nbconvert --execute (see README.md).
"""
import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


HEADER = ("import numpy as np\n"
          "import matplotlib.pyplot as plt\n"
          "from jpca_core import *\n"
          "%matplotlib inline\n"
          "plt.rcParams['figure.dpi'] = 110")


# ===========================================================================
# Notebook 1
# ===========================================================================

nb1_cells = [
md("""# jPCA on motor-cortex population data — the method, and what it means

This notebook reproduces, on synthetic data, the analysis of **Churchland et al. 2012
(*Nature*, "Neural population dynamics during reaching")** — the paper that made
"rotational dynamics" famous.

**The setup.** During a reaching task, hundreds of neurons in motor/premotor cortex are
recorded across several reach directions (conditions). Single neurons show complex,
multiphasic, condition-dependent responses — hard to read directly. jPCA asks a
collective question instead: *does the population state rotate through state space?*

**The pipeline** (each step matters — Notebook 2 shows how much):

1. firing rates per neuron per condition (Gaussian smoothing of spike trains),
2. soft-normalization per neuron (so high-rate neurons don't dominate),
3. subtract the cross-condition mean (focus on condition-dependent structure),
4. PCA to a handful of latent dimensions (Churchland used 6),
5. fit a linear dynamical system **dX/dt = M·X** in that subspace,
6. constrain **M to be skew-symmetric** (M = −Mᵀ) — a matrix that produces *purely
   rotational* flow — and read off the rotation planes from its eigenvalue pairs.

The skew-symmetry constraint is the whole trick: a skew-symmetric matrix has purely
imaginary eigenvalues ±iω, each conjugate pair defining one plane in which the state
rotates with frequency ω/2π. Nothing decays, nothing grows — rotation only."""),

code(HEADER + """

# --- Generate Churchland-style synthetic data -------------------------------
# 8 reach directions; per condition, latent dynamics = a decaying 2 Hz oscillation
# (shared plane, condition-dependent radius) + 2 direction-tuned ramps.
# 150 neurons are random linear mixtures of these 4 latents.
rates, dirs, t = simulate_reaching(n_neurons=150, n_conditions=8, f_rot_hz=2.0)
print('rates (neurons, conditions, time):', rates.shape)

fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex=True)
colors = plt.cm.hsv(np.linspace(0, 1, rates.shape[1], endpoint=False))
for ax, u in zip(axes.flat, [3, 40, 71, 96, 120, 133]):
    for c in range(rates.shape[1]):
        ax.plot(t, rates[u, c], color=colors[c], lw=1)
    ax.set_title(f'neuron {u}', fontsize=9)
    ax.set_xlabel('time (s)'); ax.set_ylabel('sp/s')
axes.flat[0].legend([f'{int(np.degrees(d))}°' for d in dirs], fontsize=6, ncol=2)
fig.suptitle('Single neurons: multiphasic, condition-tuned, unreadable one by one')
fig.tight_layout(); plt.show()"""),

md("""**What single neurons look like.** Each neuron mixes the latents: some show the
oscillation, some the ramps, most a blend. No single neuron "rotates." Rotation, if it
exists, is a property of the population state X(t) — the vector of all neurons at once.

Now run the preprocessing. The next cell walks through the three steps one at a time,
so you can see exactly what each one does, before applying them in a single call.
Note that every choice here (smoothing width, the +5 sp/s normalization constant,
subtracting the condition mean) *shapes what the rotation will look like later*.
Kuzmina et al. 2024 showed that a 10→20 ms change in smoothing can flip a dataset
from "rotating" to "not rotating." Treat these as hypotheses, not defaults."""),

code("""# --- What each preprocessing step does, before -> after ---------------------
from scipy.ndimage import gaussian_filter1d

smoothed = gaussian_filter1d(rates, 20 / 10, axis=2, mode='nearest')   # 1. smooth in time
spread = smoothed.max(axis=(1, 2)) - smoothed.min(axis=(1, 2))         # per-neuron range
normed = smoothed / (spread + 5.0)[:, None, None]                      # 2. soft-normalize
final = normed - normed.mean(axis=1, keepdims=True)                    # 3. subtract condition mean

hi, lo = int(np.argmax(spread)), int(np.argmin(spread))   # most / least modulated neuron
print(f'per-neuron range before normalization: {spread.min():.1f} .. {spread.max():.1f} sp/s')
print(f'neuron {hi}: range {spread[hi]:.1f} sp/s  ->  {np.ptp(normed[hi]):.2f} after normalization')
print(f'neuron {lo}: range {spread[lo]:.1f} sp/s  ->  {np.ptp(normed[lo]):.2f} after normalization')
print(f'the +5 sp/s "soft" constant: without it neuron {lo} would be scaled up '
      f'{spread[hi] / spread[lo]:.1f}x more than neuron {hi}; with it, only '
      f'{(spread[hi] + 5) / (spread[lo] + 5):.1f}x')

fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
c0 = 0
axes[0].plot(t, rates[hi, c0], 'o-', color='0.75', ms=2, lw=0.6, label='raw')
axes[0].plot(t, smoothed[hi, c0], color='k', lw=1.6, label='smoothed')
axes[0].set_xlim(0.25, 0.65)
axes[0].set_title(f'1. Gaussian smoothing, 20 ms SD\\n(neuron {hi}, one condition, zoomed)',
                  fontsize=9)
axes[0].legend(fontsize=7); axes[0].set_ylabel('sp/s')

for u, ls in [(hi, '-'), (lo, '--')]:
    for c in range(rates.shape[1]):
        axes[1].plot(t, rates[u, c], color=colors[c], lw=0.9, ls=ls)
        axes[2].plot(t, normed[u, c], color=colors[c], lw=0.9, ls=ls)
        axes[3].plot(t, final[u, c], color=colors[c], lw=0.9, ls=ls)
axes[1].set_title(f'raw: neuron {hi} (solid) fills the axis,\\nneuron {lo} (dashed) is invisible',
                  fontsize=9)
axes[2].set_title('2. soft-normalized: both neurons\\nnow on a comparable scale', fontsize=9)
axes[3].axhline(0, color='k', lw=0.5)
axes[3].set_title('3. condition mean removed: only\\ncondition-dependent structure remains', fontsize=9)
for ax in axes:
    ax.set_xlabel('time (s)')
fig.tight_layout(); plt.show()

rates_p = preprocess(rates, bin_ms=10, smooth_sd_ms=20)   # all three steps in one call
print('check: step-by-step == preprocess():', np.allclose(final, rates_p))"""),

code("""from sklearn.decomposition import PCA
X = stack_conditions(rates_p)                     # (N, C*T)
pca_full = PCA().fit(X.T)
cum = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(4.5, 3))
plt.plot(np.arange(1, 31), cum[:30], 'o-', ms=3)
plt.axhline(0.9, color='k', ls=':', lw=1)
plt.xlabel('# PCs'); plt.ylabel('cumulative variance')
plt.title('PCA: how many latents carry the structure?'); plt.show()
print('variance captured by 6 PCs: %.2f' % cum[5])"""),

md("""**The jPCA step, mathematically.** After PCA we have reduced trajectories Z(t) in a
6-D latent space. We fit the best linear flow:

> dZ/dt ≈ M·Z by least squares → M_full = dZ·Z⁺

then keep only the rotational part of the flow. The closed-form projection

> M_skew = (M_full − M_fullᵀ) / 2

is the initialization; the original toolbox (and our `jpca()` by default,
`refine_skew=True`) then refines it by minimizing **‖dZ − M·Z‖² directly over
skew-symmetric matrices** (conjugate gradients — the same constrained fit as
the 2012 MATLAB code). We quantify the rotational fit with
R² = 1 − ‖dZ − M_skew·Z‖² / ‖dZ‖².

Eigendecomposing M_skew yields imaginary-conjugate pairs ±iω. Each pair's eigenvector
v = a + ib defines a rotation plane span{a, b} — the "jPC planes." Projecting the data
into those planes is what produces the famous figure. (For display, the basis within
each plane is rotated so the initial — preparatory — states lie along jPC1; this is
purely cosmetic and changes no variances or fits.)

**A metric subtlety Churchland's figures hide:** the whole-space R² measures how well
rotation explains the derivative across *all six* latent dimensions — including
non-rotational ones (the direction-tuned ramps are translational, not rotational, so
their derivative cannot be fit by any skew-symmetric M). The field's actual claim rests
on the *top rotation planes*: how much variance they capture and how well rotation fits
*within* them. We report both below."""),

code("""res = jpca(rates_p, n_pcs=6, dt_s=0.01)
print('whole-space skew-symmetric fit R^2 = %.3f  (diluted by the ramp dimensions)' % res['r2'])
print('plane frequencies (Hz):', np.round(res['freqs_hz'], 2))

projs = project_planes(res)
for k, (_, frac, r2p) in enumerate(projs, 1):
    print('jPC plane %d: %4.1f%% of variance | in-plane rotation R^2 = %+.2f | %.2f Hz'
          % (k, 100 * frac, r2p, res['freqs_hz'][k - 1]))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
ev = res['eigvals']
axes[0].plot(ev.real, ev.imag, 'o')
axes[0].axhline(0, color='k', lw=0.5); axes[0].axvline(0, color='k', lw=0.5)
axes[0].set_xlabel('real (growth/decay)'); axes[0].set_ylabel('imaginary (rotation)')
axes[0].set_title('eigenvalues of M_skew: pure imaginary pairs')
axes[1].bar(range(len(projs)), [p[1] for p in projs])
axes[1].set_xlabel('jPC plane'); axes[1].set_ylabel('fraction of variance')
axes[1].set_title('variance captured per rotation plane')
fig.tight_layout(); plt.show()"""),

code("""# --- The classic figure: trajectories in the top two jPC planes -------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, (coords, frac, r2p), k in zip(axes, projs, [1, 2]):
    for c in range(rates_p.shape[1]):
        ax.plot(coords[0, c], coords[1, c], color=colors[c], lw=1.5)
        ax.plot(coords[0, c, 0], coords[1, c, 0], 'o', color=colors[c], ms=4)
        # arrowhead mid-trajectory to show the common rotation direction
        i = int(coords.shape[2] * 0.45)
        dx = coords[0, c, i+1] - coords[0, c, i]; dy = coords[1, c, i+1] - coords[1, c, i]
        ax.arrow(coords[0, c, i], coords[1, c, i], dx, dy, color=colors[c],
                 head_width=0.02*np.abs(coords).max(), length_includes_head=True)
    ax.set_title(f'jPC plane {k}  ({frac:.0%} of var, in-plane R2={r2p:.2f})')
    ax.set_xlabel('jPC1'); ax.set_ylabel('jPC2'); ax.set_aspect('equal')
fig.suptitle('Population trajectories per reach direction — the "rotational dynamics" figure')
fig.tight_layout(); plt.show()"""),

md("""## What it means — and what it does not mean

**What the plot shows:** the population state moves through a *quadrature pair* of latent
patterns — pattern A, then pattern B (90° later), then −A, then −B. Every condition does
this in the same direction, at a condition-dependent radius. Churchland's interpretation:
motor cortex is a **pattern generator** — the rotation is the dynamical machinery that
produces time-varying muscle commands (position + velocity must both be represented, and
"variable + its derivative" plotted in state space *is* a rotation; that part is nearly
kinematically inevitable).

**What the plot does *not* show:**

- **Stability or attractor-ness.** These are transient, driven trajectories — the state
  passes through the rotation once and leaves. Nothing here says the rotation recovers
  from a perturbation. (Churchland et al. never claimed an attractor; later work —
  Sussillo 2015, Michaels 2016, Lebedev 2019 — showed rotations arise without one.)
- **Uniqueness of mechanism.** jPCA *maximizes* rotation by construction — it searches
  for the most rotational view of the data. So the plot's existence is not evidence;
  the evidence is the fit *relative to shuffles/surrogate data*, and whether the rotation
  plane/frequency mean anything physically.
- **Even the R² needs reading.** Whole-space and in-plane R² answer different questions;
  a dataset can be "rotational" in its top planes while its bulk flow is not.
- **Physical location.** jPCA lives entirely in latent space. It says nothing about where
  in the brain the quadrature pair sits, or whether the rotation corresponds to anything
  spatial — which is the opening Notebook 2 exploits.

**Next:** Lebedev's doubt. If neurons simply fire one after another, jPCA will show you
a rotation — no dynamical machinery required. → Notebook 2."""),
]

nb1 = nbf.v4.new_notebook(cells=nb1_cells)
nb1.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python',
                              'name': 'python3'}
nbf.write(nb1, '01_jpca_motor_rotation.ipynb')


# ===========================================================================
# Notebook 2
# ===========================================================================

nb2_cells = [
md("""# The Lebedev / Kuzmina challenge: rotations from sequences and peak-time waves

**Lebedev, Ostrovski & Druckmann 2019** (*eNeuro*) made a deflating observation:
if neurons fire in sequence — same response shape, shifted in time — then PCA/jPCA
renders that population as a *rotation*. Rotations and sequences are two views of one
structure. No recurrent dynamics, no pattern generator, no attractor required.

**Kuzmina, Kriukov & Lebedev 2024** (*Scientific Reports*) pushed it further across 59
recordings from M1, PMd, S1, SMA, PFC and parietal cortex: rotational structure is
"consistently accounted for by a traveling-wave pattern" — a wave that travels across
the population when neurons are **sorted by their peak-firing time**. Their conclusion:
rotational dynamics and traveling waves "are typically the same phenomena," and the
rotations provide "*no insight into the underlying dynamical system*."

Two things to notice before any code:

- **The circularity.** Sorting neurons by peak time *guarantees* a wave-looking raster.
  The substantive claim is not "a wave exists" but "the peak-time order is consistent
  across conditions" — and where it isn't (their grasping datasets), rotations die.
- **No physical space anywhere.** The wave travels across a *ranking constructed from
  the data*, not across cortex. Whether that functional order coincides with an
  anatomical path is a question their data cannot ask.

This notebook demonstrates all of this in code, then does the one thing they couldn't:
put the wave in physical space."""),

code(HEADER + """

# --- Model A: pure sequences (Lebedev 2019) ---------------------------------
# 200 neurons; neuron j fires ONE Gaussian bump at time tau_j = 100 + 3*j ms.
# There is no dynamics here at all -- just a fixed recruitment order.
rates_seq, t_ms = simulate_sequences(n_neurons=200, spacing_ms=3.0)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
peak_order = np.argsort(np.argmax(rates_seq.mean(axis=1), axis=1))
axes[0].imshow(rates_seq[:, 3].T, aspect='auto', cmap='viridis',
               extent=[0, 199, t_ms[-1], t_ms[0]])
axes[0].set_title('neurons in original (arbitrary) order')
axes[0].set_xlabel('neuron (index)'); axes[0].set_ylabel('time (ms)')
axes[1].imshow(rates_seq[peak_order][:, 3].T, aspect='auto', cmap='viridis',
               extent=[0, 199, t_ms[-1], t_ms[0]])
axes[1].set_title('same data, sorted by peak time  ->  "a traveling wave"')
axes[1].set_xlabel('neuron (sorted by peak time)')
fig.suptitle('One condition of the sequence model')
fig.tight_layout(); plt.show()
print('The wave in the right panel exists BY CONSTRUCTION: we sorted by peak time.')"""),

md("""Now run the **full Churchland pipeline** on this pure sequence data — same
preprocessing, same PCA, same skew-symmetric fit. If jPCA reports rotation here, then
rotation cannot be evidence of a rotational dynamical engine: this generator has no
recurrence, no oscillation, no attractor — just an order."""),

code("""rates_seq_p = preprocess(rates_seq, bin_ms=10, smooth_sd_ms=20)
res_seq = jpca(rates_seq_p, n_pcs=6)
print('pure-sequence data:  whole-space rotation R^2 = %.3f' % res_seq['r2'])
coords, frac, r2p = project_planes(res_seq)[0]
print('top plane: %.0f%% of variance | %.2f Hz | in-plane rotation R^2 = %+.2f'
      % (100 * frac, res_seq['freqs_hz'][0], r2p))

C = rates_seq_p.shape[1]
gains = [0.6 + 0.5 * c / (C - 1) for c in range(C)]   # per-condition gains (see simulate_sequences)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# left panel: ONE condition, color = time (so the rotation direction is visible)
c_show = C - 1                                        # the largest loop (gain 1.10)
x = coords[0, c_show]
y = coords[1, c_show]
ax = axes[0]
ax.plot(x, y, color='0.7', linewidth=1, zorder=1)           # gray path underneath
sc = ax.scatter(x, y, c=t_ms, cmap='plasma', s=12, zorder=2)  # dots colored by time
ax.plot(x[0], y[0], 'o', color='k', markersize=8, zorder=3)   # black dot = start
for frac_i in [0.2, 0.5, 0.8]:                        # direction arrows along the way
    i = int(len(t_ms) * frac_i)
    ax.arrow(x[i], y[i], x[i + 1] - x[i], y[i + 1] - y[i], color='k',
             head_width=0.03 * np.abs(coords).max(), length_includes_head=True, zorder=3)
fig.colorbar(sc, ax=ax, label='time (ms)')
ax.set_title('one condition (gain %.2f), color = time\\n(the state swings around the origin)'
             % gains[c_show])
ax.set_xlabel('jPC1'); ax.set_ylabel('jPC2'); ax.set_aspect('equal')

# right panel: all 8 conditions, color = gain
ax = axes[1]
colors = plt.cm.viridis(np.linspace(0.05, 0.9, C))   # dark -> light as gain grows
for c in range(C):
    ax.plot(coords[0, c], coords[1, c], color=colors[c], linewidth=1.6,
            label='gain %.2f' % gains[c])
    ax.plot(coords[0, c, 0], coords[1, c, 0], 'o', color=colors[c], markersize=5)
    i = int(coords.shape[2] * 0.35)                  # arrow partway along the trajectory
    ax.arrow(coords[0, c, i], coords[1, c, i],
             coords[0, c, i + 1] - coords[0, c, i],
             coords[1, c, i + 1] - coords[1, c, i],
             color=colors[c], head_width=0.03 * np.abs(coords).max(),
             length_includes_head=True)
ax.set_title('all 8 conditions: the SAME loop at 8 radii\\n(dot = start, arrow = direction of motion)')
ax.set_xlabel('jPC1'); ax.set_ylabel('jPC2'); ax.set_aspect('equal')
ax.legend(fontsize=7, loc='upper right')

fig.suptitle('jPCA of PURE SEQUENCES: still rotates — %.0f%% of var | in-plane R2 = %.2f | %.2f Hz'
             % (100 * frac, r2p, res_seq['freqs_hz'][0]))
fig.tight_layout(); plt.show()"""),

md("""The rotation appears: every condition swings around the origin in the same direction,
and the whole-space fit is strong (R² ≈ 0.68). Note the honest numbers too: the in-plane
fit is modest (R² ≈ 0.19) and the loops are open arcs, not clean circles — a wave that
passes through the population once rotates *while it lasts*, noisily. This is Lebedev's
doubt, running.

**Why it happens (one line of math):** PCA of a family of time-shifted bumps returns the
bump and its own derivative as the first two components — a quadrature pair — and a
quadrature pair plotted against itself is a circle. The "rotation" is the display format
of an order.

## The Kuzmina sweep: rotation needs a phase gradient, and collapses without it

Kuzmina et al. parameterized the wave by the spacing *b* between successive neurons'
peaks. Two degenerate limits kill the rotation — and both are instructive:

- **b = 0** — all neurons fire together (a *standing* pattern / global synchrony). The
  population state oscillates along a single direction: a line segment in state space,
  rank-1, no quadrature pair, no rotation. (Their covariance-singularity argument is the
  same fact.)
- **b → large** — peaks so spread out that neurons are effectively uncorrelated: no
  collective structure at all.

Only an intermediate spacing — a real phase gradient across the population — rotates."""),

code("""spacings = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 80.0]
r2s = []
for b in spacings:
    r_b, _ = simulate_sequences(n_neurons=200, spacing_ms=b, seed=1)
    r2s.append(rotation_r2(preprocess(r_b, bin_ms=10), n_pcs=6))
    print(f'peak spacing b = {b:5.1f} ms   ->   rotation R^2 = {r2s[-1]:+.3f}')

plt.figure(figsize=(5, 3.5))
plt.semilogx(np.array(spacings) + 0.1, r2s, 'o-')
plt.xlabel('peak spacing b (ms)  [0.1 = synchronous]')
plt.ylabel('rotation R$^2$')
plt.title('Rotation lives only at intermediate spacing\\n(standing wave and decorrelation both kill it)')
plt.show()"""),

md("""## The part they couldn't do: the wave in physical space

Kuzmina's wave travels across a *ranking*. Your question is different: does the timing
order coincide with **anatomical position** — is the functional wave a physical wave?

Here neurons sit on a 1D line and oscillate. Compare two regimes:

- **traveling wave** — phase shifts steadily across space: r(x,t) = cos(2πft − kx)
- **standing wave** — one spatial envelope oscillating in time: r(x,t) = cos(2πft)·cos(kx)

A "local electrode" (20 adjacent neurons) records each. The standing wave must NOT
produce rotation (rank-1 field: one pattern, one phase). The traveling wave must.
And the diagnostic fingerprint: in a traveling wave, each neuron's oscillation *phase*
is a linear function of its *physical position*.

**Metric note.** For a pure oscillation sampled at 1 ms, the flow-fit R² is
noise-dominated and meaningless — the derivative between adjacent bins is tiny. The
honest rotation statistics here are the *shape* of the state-space loop (ellipticity:
0 = line/standing, 1 = circle/clean quadrature) and the phase–position relationship.
Choosing the metric is part of choosing the hypothesis."""),

code("""probe = slice(45, 65)   # the local electrode: 20 adjacent neurons

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for col, kind in enumerate(['traveling', 'standing']):
    rates_w, x, t_s = simulate_physical_wave(kind=kind, freq_hz=10.0)
    rates_probe = rates_w[probe]
    res_w = jpca(preprocess(rates_probe, bin_ms=1, subtract_condition_mean=False),
                 n_pcs=4, dt_s=0.001)
    coords, frac, r2p = project_planes(res_w)[0]
    ell = ellipticity(rates_probe)
    pos, ph = phase_vs_position(rates_probe, freq_hz=10.0, dt_s=0.001)
    ppcorr = np.corrcoef(pos, ph)[0, 1]

    ax = axes[col]
    ax.imshow(rates_w[:, 0].T, aspect='auto', cmap='RdBu_r',
              extent=[0, 119, t_s[-1]*1000, 0])
    ax.axhline(45, color='k', ls='--', lw=1); ax.axhline(65, color='k', ls='--', lw=1)
    ax.set_title(f'{kind} wave (dashed = probe rows)'); ax.set_xlabel('position x')
    ax.set_ylabel('time (ms)')

    axes[2].plot(np.degrees(ph) + 400 * col, pos, 'o', ms=4,
                 label=f'{kind}: phase–pos r={ppcorr:+.2f}, ellipticity={ell:.2f}')
axes[2].set_xlabel('oscillation phase (deg, offset per case)')
axes[2].set_ylabel('physical position on the line')
axes[2].legend(fontsize=8)
axes[2].set_title('the fingerprint: phase orders along anatomy\\nonly for the traveling wave')
fig.tight_layout(); plt.show()

for kind in ['traveling', 'standing']:
    rates_w, _, _ = simulate_physical_wave(kind=kind)
    rp = rates_w[probe]
    pos, ph = phase_vs_position(rp, 10.0, 0.001)
    print(f'{kind:9s} | ellipticity = {ellipticity(rp):.3f} | '
          f'phase-position correlation = {np.corrcoef(pos, ph)[0,1]:+.3f}')"""),

md("""The traveling wave gives the local probe a circular state-space loop
(ellipticity → 1) and — the key figure — **phase orders linearly along physical
position**. The standing wave gives neither: ellipticity → 0 (a line segment), and phase
is flat across space. So a local jPCA rotation rules out a purely standing pattern by
itself; what it cannot tell apart, without anatomy, is *which* traveling order produced
it — the physical wave, or any other recruitment order. That is precisely the gap your
simultaneous widefield + dense-electrophysiology design closes.

## The fragility Kuzmina documented

Rotation strength is sensitive to preprocessing choices. Same sequence data, varying
only the smoothing kernel:"""),

code("""widths = [5, 10, 20, 40, 80, 150]
r2s = [rotation_r2(preprocess(rates_seq, bin_ms=10, smooth_sd_ms=w)) for w in widths]
for w, r in zip(widths, r2s):
    print(f'smoothing = {w:4d} ms   ->   rotation R^2 = {r:+.3f}')
plt.figure(figsize=(5, 3.5)); plt.plot(widths, r2s, 'o-')
plt.xlabel('smoothing kernel width (ms)'); plt.ylabel('rotation R$^2$')
plt.title('Same data, different preprocessing, different "rotation"')
plt.show()"""),

md("""## Takeaways

1. **jPCA rotation = phase-ordered structure in the population.** It is a display format,
   not a mechanism. Sequences produce it (Lebedev 2019); peak-time waves produce it
   (Kuzmina 2024); a physical traveling wave produces it. Recurrence, attractors, and
   pattern-generation machinery are *not* required.
2. **The rotation collapses for standing/synchronous patterns** (no phase gradient, no
   quadrature pair) and for decorrelated ones. Its presence tells you an order exists;
   its shape (ellipticity) tells you how clean the quadrature is.
3. **The open scientific question is spatial, not mathematical.** Everyone can find an
   ordering of neurons that makes a wave. The question nobody could ask until now:
   does that order coincide with a *physical anatomical path* — and does perturbing the
   physical wave move the latent rotation? That is the shadow test.
4. **Preprocessing is a hypothesis.** Rotation strength moves with smoothing width,
   normalization, and condition count. Report the pipeline with the result, and fix it
   before looking at the answer."""),
]

nb2 = nbf.v4.new_notebook(cells=nb2_cells)
nb2.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python',
                              'name': 'python3'}
nbf.write(nb2, '02_lebedev_sequences_waves.ipynb')


# ===========================================================================
# Notebook 0 — real data: Churchland et al. 2012 exampleData.mat
# ===========================================================================

nb0_cells = [
md("""# jPCA on the real Churchland et al. 2012 data

This notebook runs **our own pipeline** (`jpca_core.py`) on the actual data distributed
with the original jPCA code release (`exampleData.mat`, Churchland lab website;
mirrored in the nwb4edu repo). Everything in Notebooks 1–2 is synthetic; this is the
real thing.

**What the data is** (from the 2012 paper): M1/PMd **array** recordings during the
"maze task" — straight reaches and reaches curved around barriers. 108 conditions
(4 sets of 27 target/barrier arrangements), 218 units, firing rates sampled every
10 ms starting 50 ms before movement onset. With 218 units it matches the paper's
"N-array" dataset (monkey N; Figure 3f) — J-array had 146 isolations. Note: this is
the *maze* task data — the 8-target center-out sets are different datasets (see
README for links); the straight reaches are a subset of the 108 conditions here.

The data is already distributed as smoothed firing rates, so we skip the smoothing
step of `preprocess` and keep only soft-normalization + cross-condition mean subtraction."""),

code(HEADER + """

# --- Load the real data (scipy only; the .mat predates v7.3) ----------------
from scipy.io import loadmat

# squeeze_me collapses the (1, 108) struct array to a plain (108,) record
# array; all conditions share one time axis, so the whole dataset stacks in
# a single call -- no per-condition loop.
Data = loadmat('data/exampleData.mat', squeeze_me=True)['Data']  # fields: A, times
rates_ctn = np.stack(Data['A'])                   # (C, T, N) = (108, 61, 218)
times = np.asarray(Data[0]['times']).ravel()      # time axis in ms (taken from condition 0)
for c in range(Data.size):                        # check that every condition uses this same axis
    assert np.array_equal(np.asarray(Data[c]['times']).ravel(), times)

C, T, N = rates_ctn.shape
rates_real = rates_ctn.transpose(2, 0, 1)   # reorder to (N, C, T) -- like MATLAB permute(x, [3 1 2])
print(f'{C} conditions | {N} neurons | {T} time points')
print('time axis (ms):', times[0], '...', times[-1], 'step', times[1]-times[0])

# --- nwb4edu-style exploration: example neurons across conditions ----------
# (colors here only distinguish conditions; the state-space figures below use
# the paper's convention: color = sorted starting jPC1, i.e. preparatory state)
neurons_to_show = [0, 40, 80, 120, 160, 200]      # 6 example neurons, one per panel
conds_to_show = list(range(0, C, 14))             # every 14th condition -> 8 conditions

fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex=True)   # grid of 2 rows x 3 columns
for panel in range(6):
    ax = axes[panel // 3, panel % 3]              # panel 0..5 -> its (row, col) in the grid
    neuron = neurons_to_show[panel]
    for k in range(len(conds_to_show)):
        cond = conds_to_show[k]
        # color from the blue->red colormap: 0.0 = blue (first condition), 1.0 = red (last)
        color = plt.cm.RdBu(k / (len(conds_to_show) - 1))
        ax.plot(times, rates_real[neuron, cond, :], color=color, linewidth=1)
    ax.set_title(f'neuron {neuron}', fontsize=9)
    ax.set_xlabel('time from movement onset (ms)')

legend_labels = []
for cond in conds_to_show:
    legend_labels.append(f'cond {cond}')
axes[0, 0].legend(legend_labels, fontsize=6, ncol=2)
fig.suptitle('Real M1/PMd single neurons across maze-task conditions')
fig.tight_layout(); plt.show()"""),

md("""**The peak-time-sorted raster, on real data.** Before any jPCA: average each neuron
across all 108 conditions, sort neurons by their peak time, and look at one condition.
The "traveling wave" appears — this is exactly the Kuzmina et al. construction, on the
very dataset Churchland analyzed. Keep this circularity in mind when the rotation
appears below (Notebook 2 unpacks it)."""),

code("""mean_rates = rates_real.mean(axis=1)                      # (N, T) across conditions
peak_order = np.argsort(np.argmax(mean_rates, axis=1))
plt.figure(figsize=(6, 4))
plt.imshow(rates_real[peak_order][:, 10].T, aspect='auto', cmap='viridis',
           extent=[0, N, times[-1], times[0]])
plt.xlabel('neuron (sorted by peak time)'); plt.ylabel('time (ms)')
plt.title('Real data, one condition, neurons sorted by peak time\\n— the "wave" appears by construction')
plt.colorbar(label='sp/s'); plt.show()"""),

md("""**Preprocessing on the real data, before → after.** Same steps as Notebook 1,
except the smoothing step is skipped (`smooth_sd_ms=0`): this file already contains
smoothed firing rates. Watch the two active steps — soft-normalization and
condition-mean subtraction — on the most strongly modulated unit and a
quiet-but-responsive one (10th percentile of range):"""),

code("""# --- What the preprocessing does to the real data, before -> after ----------
spread = rates_real.max(axis=(1, 2)) - rates_real.min(axis=(1, 2))     # per-unit range
normed = rates_real / (spread + 5.0)[:, None, None]                    # 2. soft-normalize
final = normed - normed.mean(axis=1, keepdims=True)                    # 3. subtract condition mean

hi = int(np.argmax(spread))                       # most modulated unit
lo = int(np.argsort(spread)[len(spread) // 10])   # 10th percentile: quiet but responsive
print(f'per-unit range before normalization: {spread.min():.1f} .. {spread.max():.1f} sp/s')
print(f'unit {hi}: range {spread[hi]:.1f} sp/s  ->  {np.ptp(normed[hi]):.2f} after normalization')
print(f'unit {lo}: range {spread[lo]:.1f} sp/s  ->  {np.ptp(normed[lo]):.2f} after normalization')
print(f'the +5 sp/s "soft" constant: without it unit {lo} would be scaled up '
      f'{spread[hi] / spread[lo]:.1f}x more than unit {hi}; with it, only '
      f'{(spread[hi] + 5) / (spread[lo] + 5):.1f}x')

fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
for u, ls in [(hi, '-'), (lo, '--')]:
    for k, cond in enumerate(conds_to_show):
        color = plt.cm.RdBu(k / (len(conds_to_show) - 1))   # same conditions as above
        axes[0].plot(times, rates_real[u, cond], color=color, lw=0.9, ls=ls)
        axes[1].plot(times, normed[u, cond], color=color, lw=0.9, ls=ls)
        axes[2].plot(times, final[u, cond], color=color, lw=0.9, ls=ls)
axes[0].set_title(f'raw: unit {hi} (solid) fills the axis,\\nunit {lo} (dashed) is invisible',
                  fontsize=9)
axes[1].set_title('soft-normalized: both units\\nnow on a comparable scale', fontsize=9)
axes[2].axhline(0, color='k', lw=0.5)
axes[2].set_title('condition mean removed: only\\ncondition-dependent structure remains',
                  fontsize=9)
for ax in axes:
    ax.set_xlabel('time from movement onset (ms)')
axes[0].set_ylabel('sp/s')
fig.tight_layout(); plt.show()

rates_p = preprocess(rates_real, bin_ms=10, smooth_sd_ms=0)   # one call, already smoothed
print('check: step-by-step == preprocess():', np.allclose(final, rates_p))"""),

md("""**jPCA on the real data, step by step.** The same treatment for the analysis itself:
instead of calling `jpca()` as a black box, the next cell spells out each step —
stack conditions → PCA to 6 dimensions → fit linear dynamics `dZ = M·Z` with `M`
restricted to be skew-symmetric (rotation only) → read the rotation planes and their
frequencies off the eigenvalues of `M` → align the axes within each plane. It collects
the results in `res_steps`, laid out like the dictionary `jpca()` returns; the cell after
it runs the one-line `jpca()` and compares the two side by side."""),

code("""# --- jPCA step by step: what the one-line jpca() call does internally -------
from sklearn.decomposition import PCA
from scipy.optimize import minimize

n_pcs, dt_s = 6, 0.01                             # 6 PCs -> 3 planes; 10 ms bins

# 1. Stack conditions end to end: every (condition, time) pair is one sample.
X = rates_p.reshape(N, C * T)                     # (N, C*T)

# 2. PCA down to the top 6 PCs. Z is the population state in PC space; jPCA
#    never leaves this 6-D space, it only picks new axes inside it.
pca = PCA(n_components=n_pcs)
Z = pca.fit_transform(X.T).T                      # (6, C*T)
print('1-2. PCA: %d neurons -> %d PCs holding %.0f%% of the variance'
      % (N, n_pcs, 100 * pca.explained_variance_ratio_.sum()))

# 3. State and its change over one time step. (As in jpca(), the difference
#    runs along the stacked axis, so it also includes the C-1 jumps from the
#    end of one condition to the start of the next.)
Z0 = Z[:, :-1]                                    # state now
dZ = np.diff(Z, axis=1)                           # state next - state now

# 4. Fit linear dynamics dZ = M Z0, twice.
#    a) unconstrained least squares: M_full may stretch, shrink and rotate
M_full = np.linalg.lstsq(Z0.T, dZ.T, rcond=None)[0].T
r2_full = 1.0 - np.sum((dZ - M_full @ Z0) ** 2) / np.sum(dZ ** 2)
#    b) skew-symmetric only (M = -M^T): such a matrix can only rotate.
#       (M_full - M_full^T)/2 is the closed-form starting guess; conjugate
#       gradients then finds the best skew-symmetric M. Only the entries above
#       the diagonal are free (15 numbers for 6x6), so we optimize those.
iu = np.triu_indices(n_pcs, k=1)

def vec2mat(h):                                   # 15 numbers -> 6x6 skew matrix
    M = np.zeros((n_pcs, n_pcs))
    M[iu] = h
    return M - M.T

def objective(h):                                 # squared error of dZ = M Z0
    return 0.5 * np.sum((Z0.T @ vec2mat(h).T - dZ.T) ** 2)

def gradient(h):
    dM = (Z0 @ Z0.T @ vec2mat(h).T) - (Z0 @ dZ.T)
    return (dM.T - dM)[iu]

M_start = 0.5 * (M_full - M_full.T)
fit = minimize(objective, M_start[iu], jac=gradient, method='CG',
               options=dict(maxiter=10000, gtol=1e-4))
M_skew = vec2mat(fit.x)

# 5. How much of the dynamics does pure rotation explain?
r2 = 1.0 - np.sum((dZ - M_skew @ Z0) ** 2) / np.sum(dZ ** 2)
print('4-5. fit R^2: any linear dynamics %.3f | rotation only %.3f' % (r2_full, r2))

# 6. Eigen-decomposition. A skew-symmetric matrix has purely imaginary
#    eigenvalues in conjugate pairs +-i*w: each pair is one rotation plane and
#    w is its angular speed (radians per bin). The real and imaginary parts of
#    the eigenvector span the plane; QR makes them an orthonormal 2-D basis.
eigvals, eigvecs = np.linalg.eig(M_skew)
print('6.   eigenvalues (imaginary part):', np.round(np.sort(eigvals.imag), 4))
planes, freqs_hz = [], []
for i in np.argsort(-eigvals.imag):               # fastest rotation first
    if eigvals[i].imag <= 1e-10:
        break                                     # the rest are the -i*w partners
    v = eigvecs[:, i]
    Q, _ = np.linalg.qr(np.column_stack([v.real, v.imag]))
    planes.append(Q)                              # (6, 2): PC space -> this plane
    freqs_hz.append(eigvals[i].imag / (2 * np.pi * dt_s))   # rad/bin -> Hz

# 7. Display convention: the two axes inside a plane are arbitrary up to a
#    rotation, so turn them until the starting (first-bin, preparatory) states
#    spread along jPC1. Pure in-plane rotation: nothing about the fit changes.
Zc = Z.reshape(n_pcs, C, T)
for k, Q in enumerate(planes):
    coords = np.einsum('ij,jct->ict', Q.T, Zc)    # (2, C, T) in-plane trajectories
    start = coords[:, :, 0].T                     # (C, 2) starting states
    rot = PCA(n_components=2).fit(start).components_
    if np.linalg.det(rot) < 0:                    # keep the basis right-handed
        rot[1] = -rot[1]
    new = np.einsum('ij,jct->ict', rot, coords)
    if np.abs(new[1]).max() > new[1].max():       # largest jPC2 excursion positive
        rot = -rot
    planes[k] = Q @ rot.T

# Same dict layout that jpca() returns; compared with the one-line call below.
res_steps = dict(pca=pca, Z=Z, M_skew=M_skew, r2=r2, eigvals=eigvals,
                 planes=planes, freqs_hz=freqs_hz, n_conditions=C, dt_s=dt_s)
print('7.   rotation frequencies (Hz):', np.round(freqs_hz, 2))"""),

code("""# --- Compact jpca() vs the step-by-step version, side by side ---------------
res = jpca(rates_p, n_pcs=6, dt_s=0.01)   # default: original-toolbox fit + axis alignment

projs_steps = project_planes(res_steps)           # planes ordered by variance
projs_compact = project_planes(res)

print('%-32s %14s %10s' % ('', 'step-by-step', 'jpca()'))
print('%-32s %14.4f %10.4f' % ('whole-space rotation R^2', res_steps['r2'], res['r2']))
for k in range(len(res['freqs_hz'])):
    print('%-32s %14.3f %10.3f' % (f'rotation frequency {k + 1} (Hz)',
                                   res_steps['freqs_hz'][k], res['freqs_hz'][k]))
for k in range(len(projs_compact)):
    print('%-32s %14.1f %10.1f' % (f'jPC plane {k + 1}: % of variance',
                                   100 * projs_steps[k][1], 100 * projs_compact[k][1]))
    print('%-32s %14.3f %10.3f' % (f'jPC plane {k + 1}: in-plane R^2',
                                   projs_steps[k][2], projs_compact[k][2]))
print('largest difference in M_skew: %.1e' % np.abs(res_steps['M_skew'] - res['M_skew']).max())
print('largest difference in plane axes: %.1e'
      % max(np.abs(P - Pr).max() for P, Pr in zip(res_steps['planes'], res['planes'])))
print('check: step-by-step == jpca():',
      np.allclose(res_steps['M_skew'], res['M_skew']) and np.isclose(res_steps['r2'], res['r2'])
      and all(np.allclose(P, Pr) for P, Pr in zip(res_steps['planes'], res['planes'])))

# left column: step-by-step, right column: jpca(); one row per jPC plane
fig, axes = plt.subplots(2, 2, figsize=(10, 9.2))
for col, (name, projs_m) in enumerate([('step-by-step', projs_steps), ('jpca()', projs_compact)]):
    for row in range(2):
        coords, frac, r2p = projs_m[row]
        cols_k = start_ordered_colors(coords)      # color = starting jPC1 (preparatory)
        ax = axes[row, col]
        for c in range(C):
            ax.plot(coords[0, c], coords[1, c], color=cols_k[c], lw=0.8, alpha=0.8)
            ax.plot(coords[0, c, 0], coords[1, c, 0], 'o', color=cols_k[c], ms=3)
        ax.set_title(f'{name}: jPC plane {row + 1}  ({frac:.0%} of var, in-plane R2={r2p:.2f})',
                     fontsize=9)
        ax.set_xlabel('jPC1'); ax.set_ylabel('jPC2'); ax.set_aspect('equal')
fig.suptitle('Same data, two routes to jPCA: spelled-out steps (left) vs one-line jpca() (right)')
fig.tight_layout(); plt.show()"""),

code("""# --- Our pipeline on the real data (res from the jpca() call above) ----------
print('whole-space skew-symmetric fit R^2 = %.3f' % res['r2'])
print('plane frequencies (Hz):', np.round(res['freqs_hz'], 2))
projs = project_planes(res)
for k, (_, frac, r2p) in enumerate(projs, 1):
    print('jPC plane %d: %4.1f%% of variance | in-plane rotation R^2 = %+.2f | %.2f Hz'
          % (k, 100 * frac, r2p, res['freqs_hz'][k - 1]))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, (coords, frac, r2p), k in zip(axes, projs, [1, 2]):
    cols_k = start_ordered_colors(coords)      # color = starting jPC1 (preparatory)
    hw = 0.03 * np.abs(coords).max()
    for c in range(C):
        ax.plot(coords[0, c], coords[1, c], color=cols_k[c], lw=0.8, alpha=0.8)
        ax.plot(coords[0, c, 0], coords[1, c, 0], 'o', color=cols_k[c], ms=3)
        ax.arrow(coords[0, c, -1], coords[1, c, -1],
                 coords[0, c, -1] - coords[0, c, -2],
                 coords[1, c, -1] - coords[1, c, -2],
                 color=cols_k[c], alpha=0.8, head_width=hw, length_includes_head=True)
    ax.set_title(f'REAL DATA jPC plane {k}  ({frac:.0%} of var, in-plane R2={r2p:.2f})')
    ax.set_xlabel('jPC1'); ax.set_ylabel('jPC2'); ax.set_aspect('equal')
fig.suptitle('Churchland 2012 maze-task data — color = sorted starting jPC1 '
             '(preparatory state), matching the original paper convention')
fig.tight_layout(); plt.show()"""),

md("""## Same data, different lens: PCA vs jPCA

The classic 8-target "starburst" figures in the reaching literature are plain **PCA**
projections — axes that maximize variance. jPCA uses the *same* data and the *same*
initial PCA reduction, then re-rotates within that 6-D PC space so the axes align with
rotational dynamics instead of variance. Both views below come from the identical
preprocessed rates (`rates_p`): the left panel is simply the top two PCs
(`res['Z'][:2]`), the right is jPC plane 1.

Watch what changes: in the variance view, conditions spread apart (condition identity
dominates variance — the "spokes" / initial states). In the dynamics view, the same
trajectories wind into nested loops rotating in a common direction. As the 2012 paper
puts it, the rotations *"were always present in the top PCs, but were difficult to see
because they were not axis aligned."*"""),

code("""# --- PCA (variance axes) vs jPCA (dynamics axes), same data -----------------
pc_coords = unstack(res['Z'][:2], C)             # top-2 PC projection, (2, C, T)
pc_frac = np.sum(res['Z'][:2] ** 2) / np.sum(res['Z'] ** 2)
cols0 = start_ordered_colors(projs[0][0])  # one condition->color map for both panels

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for c in range(C):
    axes[0].plot(pc_coords[0, c], pc_coords[1, c], color=cols0[c], lw=0.8, alpha=0.8)
    axes[0].plot(pc_coords[0, c, 0], pc_coords[1, c, 0], 'o', color=cols0[c], ms=3)
axes[0].set_title(f'PC1-PC2 projection ({pc_frac:.0%} of var) — variance axes')
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2'); axes[0].set_aspect('equal')

coords, frac, r2p = projs[0]
for c in range(C):
    axes[1].plot(coords[0, c], coords[1, c], color=cols0[c], lw=0.8, alpha=0.8)
    axes[1].plot(coords[0, c, 0], coords[1, c, 0], 'o', color=cols0[c], ms=3)
axes[1].set_title(f'jPC plane 1 ({frac:.0%} of var) — dynamics axes')
axes[1].set_xlabel('jPC1'); axes[1].set_ylabel('jPC2'); axes[1].set_aspect('equal')

fig.suptitle('Same trajectories, same color per condition — two orthonormal bases of the same PC space')
fig.tight_layout(); plt.show()"""),

md("""## Why doesn't our figure look exactly like the paper's Figure 3?

Two reasons remain, in order of visual impact:

1. **Time window.** The paper plots only the **first 200 ms of movement-related
   activity** per condition; everything above used the full 600 ms supplied in the
   file, so each loop includes the movement-offset / return segment, which doubles
   back and tangles the picture. Below we refit jPCA on just −50 to 150 ms — the
   arcs immediately resolve into the sweeping, radiating structure of the paper's
   panels.
2. **Which dataset.** The clean textbook loops are Figure 3a–b: monkeys B and A,
   28 *straight-reach* conditions recorded with single electrodes. This file is the
   108-condition **array** dataset (curved + straight reaches mixed), whose paper
   panels (e–f) are visibly messier too.

Three further conventions of the original pipeline are now adopted here, after a
side-by-side comparison with the reference Python port (`bantin/jPCA`,
`simple_jPCA_example.ipynb`). Both are the **default** in our `jpca()`, so all
three notebooks follow the original paper's analysis:

- **Axis alignment** (`align_axes`, on by default). The basis *within* a jPC
  plane is arbitrary up to rotation, and the toolbox rotates it so the
  first-bin — preparatory — states lie along the jPC1 axis. Without this step
  the preparatory cloud leaks into both axes: across conditions, the
  trial-start states have std 0.34 / 0.31 in jPC1 / jPC2 before alignment, vs
  0.45 / 0.05 after — the same cloud, just rotated. This, not the data, is
  what makes the paper's preparatory points look compact and ours did not.
- **Constrained skew fit** (`refine_skew`, on by default). The toolbox
  minimizes ‖dX − M·X‖ over skew-symmetric M directly (conjugate gradients);
  the closed-form `(M − Mᵀ)/2` shortcut is only its initialization. The
  difference is small (‖ΔM‖/‖M‖ ≈ 0.13 here) but this notebook matches the
  reference pipeline exactly.
- **Coloring.** Trajectories are colored by sorted starting jPC1 coordinate —
  the preparatory state — via `start_ordered_colors`, as in the paper and the
  MATLAB toolbox. Axes are still defined up to an overall sign.

Honest caveat: even in the paper's window, the in-plane skew-symmetric fit is
weak for this dataset — the rotation is visually consistent but the linear
model in that plane explains little. The paper's quantitative claim rests on
the fit in the full 6-D PC space, not on clean in-plane fits."""),

code("""# --- Refit jPCA on the paper's analysis window (first 200 ms) ---------------
k200 = int(np.searchsorted(times, 150))          # -50 .. 150 ms
res200 = jpca(rates_p[:, :, :k200], n_pcs=6, dt_s=0.01)   # same default pipeline
coords200, frac200, r2p200 = project_planes(res200)[0]
print('200 ms window: plane 1 %.0f%% of var | in-plane rotation R^2 = %+.2f | freqs %s Hz'
      % (100 * frac200, r2p200, np.round(res200['freqs_hz'], 2)))

fig, ax = plt.subplots(figsize=(6, 5.4))
cols200 = start_ordered_colors(coords200)      # color = sorted starting jPC1
hw = 0.03 * np.abs(coords200).max()
for c in range(C):
    ax.plot(coords200[0, c], coords200[1, c], color=cols200[c], lw=0.8, alpha=0.85)
    ax.plot(coords200[0, c, 0], coords200[1, c, 0], 'o', color=cols200[c], ms=3)
    ax.arrow(coords200[0, c, -1], coords200[1, c, -1],
             coords200[0, c, -1] - coords200[0, c, -2],
             coords200[1, c, -1] - coords200[1, c, -2],
             color=cols200[c], alpha=0.85, head_width=hw, length_includes_head=True)
ax.set_title('Paper window (first 200 ms), color = preparatory state (sorted starting jPC1)')
ax.set_xlabel('jPC1'); ax.set_ylabel('jPC2'); ax.set_aspect('equal')
fig.tight_layout(); plt.show()"""),

md("""## Compare with the synthetic version

Put this figure next to Notebook 1's. The real data shows the same qualitative
structure — nested, consistently directed loops in the top planes — which is why the
synthetic generator in `simulate_reaching` was built the way it was. The differences
are equally instructive: the real trajectories are messier, the plane variance is split
less cleanly, and with 108 heterogeneous maze conditions the "radius per condition"
organization is only approximate.

Then ask the Lebedev/Kuzmina question of *this* dataset: the sorted raster above shows
the peak-time wave is already there in the raw rates. jPCA did not discover a hidden
engine — it re-displayed an order that exists in the data. Whether that order has a
physical, anatomical address is the question no electrode-only dataset can answer —
and the subject of the proposed experiments. → Notebook 2."""),
]

nb0 = nbf.v4.new_notebook(cells=nb0_cells)
nb0.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python',
                              'name': 'python3'}
nbf.write(nb0, '00_real_data_churchland2012.ipynb')


# ===========================================================================
# Notebook 3 — jPCA explained from scratch (hands-on companion)
# ===========================================================================

nb3_cells = [
md("""# jPCA explained from scratch — a hands-on companion

This notebook is a slow walk through the method used in notebooks 00–02. No real
data, no statistics — just toy systems small enough to see through completely.
Every idea comes twice: once in text, once as code you can run and poke.
(Best read *before* the other notebooks, despite the number.)

**The whole method in one sentence:** jPCA fits the simplest possible dynamical
system to the population's trajectory — *dX/dt = M·X* — and then keeps only the
part of M that **rotates**.

1. the state-space picture — what "population state" even means
2. fitting a law of motion to a trajectory (it is just regression)
3. why *skew-symmetric* means *pure rotation*
4. how one fitted matrix becomes "rotation planes" and the famous loop figure
5. what a rotation means biologically: patterns chasing patterns
6. the catch: jPCA always finds rotations — so what counts as evidence?

**Python reading hints for MATLAB eyes** (used throughout):

- `@` is matrix multiply (MATLAB `*`); plain `*` is element-wise (MATLAB `.*`)
- `X.T` is transpose (MATLAB `X.'`); `X[:, :-1]` = all columns except the last
- indexing starts at 0, and ranges exclude the last element (`0:3` ≈ `range(3)`)
- `np.linalg.pinv`, `np.linalg.eig`, `scipy.linalg.expm` = MATLAB `pinv`, `eig`, `expm`"""),

md("""## 1. The state-space picture

Forget individual neurons for a moment. At each instant the whole population is
ONE point:

- the space has one axis per neuron (218 in the real data, 3 in the demo below),
- the coordinates of the point at time t are the firing rates of all neurons at t,
- one trial / condition = one **trajectory** traced by that point as time passes.

Two views of the SAME data below: on the left, three familiar rate-vs-time traces;
on the right, the same three numbers plotted as one dot moving through "neuron
space." jPCA lives entirely in the right-hand view."""),

code(HEADER + """

# --- One trial, two views ---------------------------------------------------
# A 3-neuron toy population watched for 2 s: three rate traces on the left;
# THE SAME data as one moving dot in "neuron space" on the right.
t = np.arange(0, 2.0, 0.01)            # time axis in seconds, 10 ms steps
r1 = 10 + 5 * np.cos(2 * np.pi * t)    # neuron 1: 1 Hz cosine around 10 sp/s
r2 = 10 + 5 * np.sin(2 * np.pi * t)    # neuron 2: same wave, a quarter-cycle later
r3 = 10 + 3 * np.sin(4 * np.pi * t)    # neuron 3: a faster 2 Hz wiggle

fig = plt.figure(figsize=(11, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(t, r1, label='neuron 1')
ax1.plot(t, r2, label='neuron 2')
ax1.plot(t, r3, label='neuron 3')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('firing rate (sp/s)')
ax1.set_title('view 1: three rate traces')
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')   # 3-D panel: axes = neuron rates
ax2.plot(r1, r2, r3, color='k', linewidth=1)
ax2.plot(r1[0], r2[0], r3[0], 'o', color='tab:green')    # green dot = start
ax2.plot(r1[-1], r2[-1], r3[-1], 's', color='tab:red')   # red square = end
ax2.set_xlabel('neuron 1 rate')
ax2.set_ylabel('neuron 2 rate')
ax2.set_zlabel('neuron 3 rate')
ax2.set_title('view 2: ONE dot moving in neuron space\\n(green = start, red = end)')
fig.tight_layout()
plt.show()
print("Same data, two views. jPCA asks about the SHAPE of the moving dot's path.")"""),

md("""## 2. Fitting a law of motion = regression, nothing more

The simplest possible law of motion for the state X(t):

> **dX/dt = M·X** — "where the state is heading = one fixed matrix × where it is"

If we knew M we would know the rules of the system. We don't — but we can estimate
it from data. Measure the state in every time bin (X), and the change between
adjacent bins (dX = diff(X)). Then find the M that makes dX ≈ M·X as closely as
possible over all bins. That is an ordinary least-squares problem; in MATLAB you
would write `M = dX / X`.

The demo cooks up a trajectory from a KNOWN matrix, then checks that regression
recovers it. (In the real pipeline this fit happens after PCA, in a 6-D latent
space, with all conditions stacked end to end — same idea, more columns.)"""),

code("""# --- Cook up a trajectory from a KNOWN matrix, then recover M by regression ---
from scipy.linalg import expm          # matrix exponential -- same as MATLAB's expm

dt = 0.01                              # time step (s)
t = np.arange(0, 2.0, dt)
omega = 2 * np.pi * 1.0                # 1 Hz, in radians per second
M_true = np.array([[0.0, -omega],
                   [omega,  0.0]])     # the hidden law of motion (a pure rotator, section 3)

# Generate the trajectory: x(t+dt) = expm(M*dt) @ x(t), starting from (1, 0).
step = expm(M_true * dt)               # expm is the exact update rule for x' = Mx
X = np.zeros((2, len(t)))
X[:, 0] = [1.0, 0.0]
for k in range(len(t) - 1):
    X[:, k + 1] = step @ X[:, k]

# Now forget M_true and estimate it from the trajectory alone:
# dX/dt approx M X   ->   M = (dX/dt) pinv(X)     [MATLAB: M = (dX/dt) / X]
X0 = X[:, :-1]                         # state in every bin except the last
dX = np.diff(X, axis=1)                # change per bin, like MATLAB diff(X, 1, 2)
M_hat = (dX / dt) @ np.linalg.pinv(X0) # least-squares fit of the law of motion
print('the hidden matrix M was:')
print(np.round(M_true, 3))
print('recovered from the trajectory alone:')
print(np.round(M_hat, 3))
print('largest error: %.4f' % np.max(np.abs(M_hat - M_true)))
print('-> fitting a linear dynamical system is just regression (least squares).')"""),

md("""You may notice the recovered matrix is not *perfect*: the diagonal came out
−0.197 instead of 0. That is not a bug — it is the finite-difference
approximation. `diff(X)` measures the *chord* between adjacent bins, and for
circular motion the chord points slightly *inward* from the true tangent — so the
fit sees a tiny fake decay (of size ≈ ω²·dt/2) on top of the rotation. Smaller
time bins shrink it. The same subtlety is why the real pipeline keeps the bins
small, and why notebook 02 warns that derivative-based fits get fragile when the
per-bin change is tiny."""),

md("""## 3. Why "skew-symmetric" is exactly "pure rotation"

Any square matrix splits into two pieces by plain algebra:

> M = (M + Mᵀ)/2  +  (M − Mᵀ)/2  =  symmetric part  +  skew-symmetric part

The two pieces do geometrically different jobs:

- the **symmetric part** stretches or shrinks the state along fixed axes —
  growth, decay, convergence onto attractors;
- the **skew-symmetric part** (defined by M = −Mᵀ) **rotates** the state, and
  does nothing else.

Why is skew the same as rotation? Circular motion at frequency ω is
x(t) = r·cos(ωt), y(t) = r·sin(ωt). Differentiate:

> dx/dt = −ω·y,   dy/dt = ω·x   →   dX/dt = [[0, −ω], [ω, 0]]·X

and that matrix is exactly skew-symmetric. The converse also holds: **every**
skew-symmetric matrix, in any dimension, generates pure rotation — its eigenvalues
are purely imaginary (±iω), and expm(M) is a rotation matrix: it turns vectors
without changing their length.

jPCA's move: fit M to data, keep only its skew-symmetric part, and ask how much of
the motion that part alone explains (that fraction is the R² printed in notebooks
00–02). Below: rotation only, decay only, and both together (a spiral) — so you
can see what "keeping only the rotation" throws away."""),

code("""# --- Same machinery, three trajectories: rotation only, decay only, both ------
M_rot   = np.array([[0.0, -omega],
                    [omega,  0.0]])    # skew-symmetric: M == -M.T (flip + negate)
M_decay = np.array([[-1.0, 0.0],
                    [0.0, -1.0]])      # symmetric: shrinks x and y equally
M_both  = M_decay + M_rot              # any M = symmetric part + skew part

mats  = [M_rot, M_decay, M_both]
names = ['skew part only:\\npure rotation (a circle)',
         'symmetric part only:\\npure decay (collapse to 0)',
         'both parts together:\\na spiral']
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for i in range(3):
    step = expm(mats[i] * dt)
    X = np.zeros((2, len(t)))
    X[:, 0] = [1.0, 0.0]
    for k in range(len(t) - 1):
        X[:, k + 1] = step @ X[:, k]
    axes[i].plot(X[0, :], X[1, :])
    axes[i].plot(X[0, 0], X[1, 0], 'o', color='tab:green')
    axes[i].set_title(names[i], fontsize=10)
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    axes[i].set_aspect('equal')
fig.tight_layout()
plt.show()

one_turn = expm(M_rot * 1.0)           # advance the rotator by exactly one period
print('rotator applied over one full period (should be ~ the identity matrix):')
print(np.round(one_turn, 6))
print('-> a skew matrix turns vectors WITHOUT changing their length. That is all it does.')"""),

md("""## 4. How ONE matrix becomes "rotation planes" and the loop figure

This is the step everyone gets stuck on. Take it in two sizes.

**In 3D, where you can see it.** In 3D a skew-symmetric matrix is an old friend:
[[0, −c, b], [c, 0, −a], [−b, a, 0]] times a vector is exactly the cross product
(a, b, c) × v. So 3-D skew dynamics = rotation **about an axis**, and the motion
happens **in the plane perpendicular to that axis**. Its eigenvalues are
{0, +iω, −iω}: the 0 belongs to the axis (points on it don't move); the ±iω pair
describes the circling in the plane. The eigenvector of +iω is *complex*,
v = a + ib — and its real part a and imaginary part b are two ordinary real
vectors that span exactly the rotation plane.

**In 6D there is no single axis — instead, up to three independent planes.** A 6×6
skew matrix can rotate in three 2-D planes *at once*, each at its own frequency,
like three gears inside one machine. `eig(M_skew)` hands you all three: the
eigenvalues come in pure-imaginary pairs ±iω₁, ±iω₂, ±iω₃, and each pair's
eigenvector v = a + ib supplies that plane's two spanning directions. The
frequency you read off, ω/2π, is the number printed as "Hz" in notebooks 00–02.

**The loop figure is then just a change of coordinates.** For each time point,
project the state onto the plane's two directions: coordinate 1 = a·Z(t),
coordinate 2 = b·Z(t) — two dot products per time point. Plot coordinate 2 against
coordinate 1 as time runs. If the dynamics in that plane really is a rotation, the
plot traces a loop. That plot *is* the famous jPCA figure.

Below: first the 3-D case, then a 6-D system built BY HAND with two rotation
planes, so we can watch `eig()` recover them and draw the loop figure ourselves."""),

code("""# --- 3D: a skew matrix = rotation about an axis -------------------------------
# Rotation about the z-axis at 1 Hz: x and y chase each other, z does nothing.
M3 = np.array([[0.0, -omega, 0.0],
               [omega,  0.0, 0.0],
               [0.0,   0.0,  0.0]])
eigvals3, eigvecs3 = np.linalg.eig(M3)   # like MATLAB [V, D] = eig(M)
print('eigenvalues of the 3-D skew matrix:', np.round(eigvals3, 3))
print('one is zero         -> the rotation AXIS (points on it do not move)')
print('two are +/- i*omega -> the rotation PLANE and its speed')

i = np.argmax(eigvals3.imag)             # position of the +i*omega eigenvalue
v = eigvecs3[:, i]                       # its (complex) eigenvector
a3 = v.real / np.linalg.norm(v.real)     # real part      -> one unit vector
b3 = v.imag / np.linalg.norm(v.imag)     # imaginary part -> a second unit vector
print()
print('eigenvector v =', np.round(v, 3))
print('real part a =', np.round(a3, 3))
print('imag part b =', np.round(b3, 3))
print('a . b = %.3f -> perpendicular' % (a3 @ b3))
print('both have zero z-component: the plane they span is the xy-plane,')
print('the plane perpendicular to the rotation axis (z).')"""),

code("""# --- 6D: eig() finds the planes automatically -- the loop figure, by hand -----
# Build a 6x6 skew matrix containing TWO independent rotation blocks:
omega1 = 2 * np.pi * 1.0               # plane A (coordinates 1-2): 1.0 Hz
omega2 = 2 * np.pi * 0.4               # plane B (coordinates 3-4): 0.4 Hz
M6 = np.zeros((6, 6))                  # coordinates 5-6: no dynamics at all
M6[0, 1] = -omega1
M6[1, 0] = omega1
M6[2, 3] = -omega2
M6[3, 2] = omega2

# Simulate a 6-D trajectory starting from a point inside both planes
X6 = np.zeros((6, len(t)))
X6[:, 0] = [1.0, 0.0, 0.7, 0.0, 0.0, 0.0]
step6 = expm(M6 * dt)
for k in range(len(t) - 1):
    X6[:, k + 1] = step6 @ X6[:, k]

# This is exactly what jpca() does at jpca_core.py:97 -- eigendecompose M_skew.
eigvals6, eigvecs6 = np.linalg.eig(M6)
print('eigenvalues of M6:', np.round(eigvals6, 2))
print('two pure-imaginary pairs (the two rotation planes) + two zeros (dead directions)')

fastest_first = np.argsort(-eigvals6.imag)   # sort by rotation speed, fastest first
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
for i in range(2):
    v = eigvecs6[:, fastest_first[i]]        # eigenvector of one +/-i*omega pair
    a = v.real / np.linalg.norm(v.real)      # plane direction 1
    b = v.imag / np.linalg.norm(v.imag)      # plane direction 2
    coord1 = a @ X6                          # project the trajectory into the plane:
    coord2 = b @ X6                          # one dot product per direction per time point
    freq_hz = eigvals6.imag[fastest_first[i]] / (2 * np.pi)
    axes[i].plot(coord1, coord2)
    axes[i].plot(coord1[0], coord2[0], 'o', color='tab:green')
    axes[i].set_title('plane %d: rotation at %.1f Hz' % (i + 1, freq_hz))
    axes[i].set_xlabel('coordinate along a')
    axes[i].set_ylabel('coordinate along b')
    axes[i].set_aspect('equal')
fig.suptitle('The jPCA loop figure, built by hand: data projected into the eigen-planes of M6')
fig.tight_layout()
plt.show()
print('Every loop figure in notebooks 00-02 is made exactly this way.')"""),

md("""## 5. What a rotation means biologically: patterns chasing patterns

A "pattern" is one number per neuron — a population-wide shape of activity
("neurons 4–8 loud, the rest quiet" is a pattern). Suppose the state rotates in
the plane spanned by two patterns a and b:

> X(t) = a·cos(ωt) + b·sin(ωt)

Read that aloud: at phase 0 the population activity IS pattern a. A quarter-cycle
later cos has faded and sin has grown, so the population IS pattern b. Then −a
(the same shape, sign-flipped), then −b, then back to a. **A rotation in state
space = the population endlessly morphing through a → b → −a → −b.** The first
figure below freezes one cycle at 8 moments so you can watch the morphing.

Now the single-neuron view. Neuron i's rate is

> rate_i(t) = a_i·cos(ωt) + b_i·sin(ωt) = R_i·cos(ωt − φ_i)

The last step is the trig identity from school: any mixture A·cos + B·sin is a
single cosine with some amplitude R and phase φ. So **every neuron oscillates at
the same frequency, but each at its own phase** — each neuron peaks at its own
time. That is the second figure, left panel.

And now the punchline. Neurons peaking one after another *is what a sequence is*.
Sort the neurons by their phase (= peak time) and stack their rates as an image:
a traveling wave appears (middle panel). The same data in state space is a loop
(right panel). **"The population state rotates" and "the neurons fire in a
phase-shifted sequence" are two descriptions of ONE object.**

That is why jPCA reports rotations for pure feedforward sequences (notebook 02),
and why the loop figure alone cannot tell you whether any dynamical machinery
produced them. It is a display format for phase-ordered structure — not a
detector of an engine."""),

code("""# --- One rotation cycle = the population morphing a -> b -> -a -> -b -> a -----
n_neurons = 20
neuron_axis = np.arange(n_neurons)
pattern_a = np.exp(-(neuron_axis - 4) ** 2 / (2 * 2.0 ** 2))    # bump around neuron 4
pattern_b = np.exp(-(neuron_axis - 14) ** 2 / (2 * 2.0 ** 2))   # bump around neuron 14

phases_deg = [0, 45, 90, 135, 180, 225, 270, 315]   # 8 snapshots around one cycle
fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharey=True)
for i in range(8):
    phase = np.radians(phases_deg[i])               # degrees -> radians
    state = pattern_a * np.cos(phase) + pattern_b * np.sin(phase)  # X(t) at this moment
    ax = axes[i // 4, i % 4]                        # panel i -> its (row, column)
    ax.plot(neuron_axis, state)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('phase %d deg' % phases_deg[i], fontsize=9)
    ax.set_xlabel('neuron #')
axes[0, 0].set_ylabel('activity')
axes[1, 0].set_ylabel('activity')
fig.suptitle('The state X(t) = a*cos(wt) + b*sin(wt), frozen at 8 moments: '
             'pattern a morphs into b, then -a, then -b')
fig.tight_layout()
plt.show()"""),

code("""# --- Same rotation, seen by single neurons: one wave, many phases -------------
t1 = np.arange(0, 1.0, dt)                # one full 1 Hz cycle
rates = np.zeros((n_neurons, len(t1)))
for i in range(n_neurons):                # neuron i = its own mixture of a and b
    rates[i, :] = pattern_a[i] * np.cos(2 * np.pi * t1) + pattern_b[i] * np.sin(2 * np.pi * t1)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))

# panel 1: a few single-neuron time courses (stacked vertically for visibility)
example_neurons = [2, 6, 10, 14]
for k in range(len(example_neurons)):
    neuron = example_neurons[k]
    axes[0].plot(t1, rates[neuron, :] + 2.5 * k, label='neuron %d' % neuron)
axes[0].set_title('single neurons: same frequency,\\neach peaking at its own time')
axes[0].set_xlabel('time (s)')
axes[0].set_ylabel('rate (+ offset per neuron)')
axes[0].legend(fontsize=7)

# panel 2: all neurons, sorted by peak time -> the "traveling wave"
peak_bin = np.argmax(rates, axis=1)       # bin where each neuron peaks
order = np.argsort(peak_bin)              # neuron order sorted by peak time
im = axes[1].imshow(rates[order, :], aspect='auto', cmap='viridis',
                    extent=[t1[0], t1[-1], n_neurons, 0])
axes[1].set_title('neurons sorted by peak time\\n-> a "traveling wave" appears')
axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('neuron (sorted)')
fig.colorbar(im, ax=axes[1], label='rate')

# panel 3: the SAME data in state space -> the loop
a_hat = pattern_a / np.linalg.norm(pattern_a)
b_hat = pattern_b / np.linalg.norm(pattern_b)
coord1 = a_hat @ rates                    # how much of pattern a is present, over time
coord2 = b_hat @ rates                    # how much of pattern b
axes[2].plot(coord1, coord2)
axes[2].plot(coord1[0], coord2[0], 'o', color='tab:green')
axes[2].set_title('the same data in state space:\\na loop')
axes[2].set_xlabel('amount of pattern a')
axes[2].set_ylabel('amount of pattern b')
axes[2].set_aspect('equal')
fig.tight_layout()
plt.show()
print('Sequence (panel 2) and rotation (panel 3) are two views of ONE dataset.')"""),

md("""## 6. The catch: jPCA *always* finds rotations

Notice what the procedure does: it fits the best linear flow, keeps only the
rotational part, and then finds the planes where that rotation looks strongest.
jPCA does not test "is this data rotational?" — it **searches for the most
rotational description** of the data. It will always return planes, frequencies,
and loops.

So the loops' existence is never the evidence. The evidence is:

- **R²** — how much of the observed motion the rotation actually explains
  (notebooks 00–02 print this; on the real maze data it is 0.146 in the full
  window),
- **surrogates** — does shuffled / phase-scrambled data rotate just as much?
- **meaning** — do the planes and frequencies correspond to anything physical?

**Recap of the whole method, in five lines:**

1. stack all conditions' trajectories into one cloud of states,
2. PCA down to a few latent dimensions,
3. regress dX/dt = M·X — ordinary least squares,
4. keep only the skew-symmetric part of M — the pure rotation,
5. read planes and frequencies off its imaginary eigenvalue pairs, and project
   the data into the planes → the loop figure.

**Where next:** notebook 01 runs this pipeline on synthetic reaching data,
notebook 00 on the real Churchland 2012 data, and notebook 02 shows why pure
sequences pass the same test — the Lebedev/Kuzmina challenge."""),
]

nb3 = nbf.v4.new_notebook(cells=nb3_cells)
nb3.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python',
                              'name': 'python3'}
nbf.write(nb3, '03_jpca_explained.ipynb')

print('notebooks written')
