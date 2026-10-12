"""Visual check: PCA on X.T (samples = time points) vs PCA on X (samples = neurons).

X is (N neurons, C*T time points), the stacked preprocessed real data.
  A = PCA(6).fit_transform(X.T).T   -> scores,     (6, C*T)   <- what jpca() uses
  B = PCA(6).fit(X).components_     -> components, (6, C*T)
Same shape. Row k of A is row k of B times the k-th singular value (up to
sign); what is left over comes from the two calls centering different axes.

Run:  python demo_pca_two_orientations.py   -> pca_two_orientations.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from jpca_core import preprocess

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e6e5e1'

# --- data: same X as the step-by-step cell in notebook 00 -------------------
Data = loadmat('data/exampleData.mat', squeeze_me=True)['Data']
rates = np.stack(Data['A']).transpose(2, 0, 1)            # (N, C, T)
N, C, T = rates.shape
X = preprocess(rates, bin_ms=10, smooth_sd_ms=0).reshape(N, C * T)

# --- the two PCAs -----------------------------------------------------------
pca_A = PCA(n_components=6)
A = pca_A.fit_transform(X.T).T                            # scores      (6, C*T)
pca_B = PCA(n_components=6).fit(X)
B = pca_B.components_.copy()                              # components  (6, C*T)
s = pca_A.singular_values_
r = np.array([np.corrcoef(A[k], B[k])[0, 1] for k in range(6)])
B *= np.sign(r)[:, None]                                  # PC sign is arbitrary: align it

# --- figure -----------------------------------------------------------------
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False})
n_show = 4 * T                                            # first 4 conditions
x = np.arange(n_show)
fig, axes = plt.subplots(3, 3, figsize=(13, 9.2), facecolor='#fcfcfb')

for row, k in enumerate([0, 1]):
    a, b, both = axes[row]
    a.plot(x, A[k, :n_show], color=BLUE, lw=2)
    a.set_title(f'A: PC{k + 1} scores from PCA on X.T\nrow length = {np.linalg.norm(A[k]):.1f}',
                loc='left', color=INK)
    b.plot(x, B[k, :n_show], color=ORANGE, lw=2)
    b.set_title(f'B: component {k + 1} from PCA on X\nrow length = {np.linalg.norm(B[k]):.1f}',
                loc='left', color=INK)
    both.plot(x, A[k, :n_show], color=BLUE, lw=4.5, alpha=0.55, label='A')
    both.plot(x, s[k] * B[k, :n_show], color=ORANGE, lw=1.4,
              label=f'B x singular value ({s[k]:.1f})')
    both.set_title(f'overlaid after rescaling B\ncorrelation over all {C * T} samples = {abs(r[k]):.4f}',
                   loc='left', color=INK)
    both.legend(frameon=False, loc='lower left')
    for ax in (a, b, both):
        ax.set_facecolor('#fcfcfb')
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        for edge in range(T, n_show, T):                  # condition boundaries
            ax.axvline(edge, color=GRID, lw=1, zorder=0)
        ax.set_xlabel('sample (first 4 conditions, end to end)')

# every sample: A against B falls on a line of slope = singular value
ax = axes[2, 0]
ax.scatter(B[0], A[0], s=3, color=BLUE, alpha=0.35, linewidths=0)
lim = np.abs(B[0]).max()
ax.plot([-lim, lim], [-s[0] * lim, s[0] * lim], color=INK, lw=1,
        label=f'slope = singular value = {s[0]:.1f}')
ax.set_xlabel('B: component 1 value'); ax.set_ylabel('A: PC1 score')
ax.set_title(f'PC1, all {C * T} samples:\nA is proportional to B', loc='left', color=INK)
ax.legend(frameon=False, loc='upper left')

# singular values: the two PCAs find (nearly) the same spectrum
ax = axes[2, 1]
pcs = np.arange(1, 7)
ax.plot(pcs, s, 'o', ms=11, color=BLUE, alpha=0.55, label='A: PCA on X.T')
ax.plot(pcs, pca_B.singular_values_, 'o', ms=4.5, color=ORANGE, label='B: PCA on X')
ax.set_ylim(0, s[0] * 1.15); ax.set_xticks(pcs)
ax.set_xlabel('PC'); ax.set_ylabel('singular value')
ax.set_title('same singular values\n(= the scale factors above)', loc='left', color=INK)
ax.legend(frameon=False, loc='lower left')

# how close is the match, PC by PC
ax = axes[2, 2]
ax.bar(pcs, np.abs(r), width=0.55, color=BLUE)
for p, v in zip(pcs, np.abs(r)):
    ax.text(p, v + 0.012, f'{v:.3f}', ha='center', color=INK, fontsize=8)
ax.set_ylim(0, 1.12); ax.set_xticks(pcs)
ax.set_xlabel('PC'); ax.set_ylabel('|correlation| between A and B rows')
ax.set_title('match is not exact: A centers each neuron,\nB centers each time point',
             loc='left', color=INK)
for ax in axes[2]:
    ax.set_facecolor('#fcfcfb')

fig.suptitle('PCA on X.T (scores) vs PCA on X (components): same patterns, different scale',
             x=0.01, ha='left', fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig('pca_two_orientations.png', dpi=130, facecolor=fig.get_facecolor())
print('saved pca_two_orientations.png')
print('singular values:', np.round(s, 1))
print('|correlation| A vs B per PC:', np.round(np.abs(r), 4))
