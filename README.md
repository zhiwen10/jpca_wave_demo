# jpca_wave_demo

Executable notebooks demonstrating (0) the jPCA pipeline on the real
Churchland et al. 2012 reaching data, (1) why pure sequential / peak-time-wave
structure produces the same rotations without any dynamical machinery
(Lebedev et al. 2019; Kuzmina, Kriukov & Lebedev 2024) — including the
physical-space (standing vs traveling wave) test those papers could not do —
and (2) a from-scratch explainer of the jPCA method itself, best read first.

## Files

- `00_real_data_churchland2012.ipynb` — the real `exampleData.mat` from the
  official jPCA distribution (maze task: 108 conditions, 218 units — matches the
  paper's N-array / monkey N panel), explored nwb4edu-style and run
  through our own pipeline. Includes a before/after view of the two active
  preprocessing steps (soft-normalization, condition-mean subtraction),
  the peak-time-sorted raster on real
  data (the Kuzmina construction made visible), the jPCA run itself (with the
  steps explained in comments), the real rotation figure,
  a side-by-side PCA-vs-jPCA projection of the same trajectories
  (variance axes vs dynamics axes), and a refit on the paper's 200 ms
  analysis window showing why our default figure looks messier than Fig. 3.
  Matches the reference conventions of the original toolbox (and the
  bantin/jPCA port): constrained skew-symmetric fit (`refine_skew=True`),
  in-plane axes aligned so preparatory states lie along jPC1
  (`align_axes=True`), and color = sorted starting jPC1 coordinate
  (preparatory state), via `start_ordered_colors`.
- `01_lebedev_sequences_waves.ipynb` — pure feedforward sequences rotate
  under the same pipeline; the Kuzmina spacing sweep (synchrony and
  decorrelation both kill rotation); traveling vs standing waves in physical
  space (ellipticity, phase-vs-position fingerprint); preprocessing fragility.
- `02_jpca_explained.ipynb` — a from-scratch explainer of the method itself
  (best read first, no data needed): state space, the least-squares fit of
  dX/dt = M·X, why skew-symmetric = pure rotation, how the eigenvalue pairs of
  M become rotation planes and the loop figure (built by hand in 3D and 6D),
  and why a rotation = neurons firing in a phase-shifted sequence.
- `jpca_core.py` — implementation: preprocessing, jPCA fit, plane projection,
  rotation metrics, synthetic data generators. The fit follows the original
  2012 toolbox (and the bantin port) by default: constrained skew-symmetric
  least squares (`refine_skew`) + preparatory-axis alignment (`align_axes`).
- `build_notebooks.py` — regenerates the three .ipynb files (nbformat).
- `data/exampleData.mat` — the real Churchland 2012 data (mirror of the
  Churchland lab jPCA distribution, via the nwb4edu repo).

## Setup (students)

You need [conda](https://docs.conda.io/en/latest/miniconda.html)
(Miniconda or Anaconda). Then, from this folder:

```bash
conda env create -f environment.yml   # one time only; creates the "jpca" env
conda activate jpca
jupyter lab                           # or: jupyter notebook
```

Then open the notebooks in order: `02_jpca_explained.ipynb` first (the
method, from scratch, no data needed), then `00`, then `01`. Run each
notebook top to bottom; everything is self-contained and the real dataset
(`data/exampleData.mat`) is included.

To remove the environment later: `conda deactivate; conda env remove -n jpca`.

## Run (maintainers)

Regenerate and re-execute the notebooks from `build_notebooks.py`:

```bash
python build_notebooks.py
for nb in 00_real_data_churchland2012 01_lebedev_sequences_waves 02_jpca_explained; do
  python -m jupyter nbconvert --to notebook --execute --inplace $nb.ipynb
done
```

## Headline results (executed outputs)

- Notebook 0 (real data): rotations appear in the top jPC planes (39% + 32%
  of variance, ~1.2 and ~0.5 Hz) — visibly the classic nested loops, but
  with weaker fits than the textbook figure suggests; the peak-time-sorted
  raster already contains the "wave" before any jPCA; with the paper's
  200 ms window + axis alignment + constrained skew fit, the figure
  reproduces the bantin/jPCA reference starburst (preparatory states compact
  along jPC1, trajectories sweeping out).
- Notebook 1: pure sequences give whole-space rotation R² ≈ 0.67 with zero
  dynamics in the generator; rotation vs peak-spacing b peaks at intermediate
  b and collapses at b=0 (standing/synchronous) and b→∞ (decorrelated);
  traveling wave → ellipticity ≈ 0.83 with phase linearly ordered along
  position (r = −1.0); standing wave → ellipticity ≈ 0.03, no relation.

## The 8-target center-out datasets (verified public, spike data, no NWB needed)

The Churchland `exampleData.mat` is the *maze* task (108 conditions), not the
8-target center-out design. For true 8-direction center-out data with code
context (all catalogued in Kuzmina et al. 2024's dataset table):

- **Suresh et al. 2020** (eLife 9:e58848; Hatsopoulos/Bensmaia labs) — M1 + S1
  spiking, isolated grasping AND 8-target center-out reaching. Dryad:
  doi:10.5061/dryad.xsj3tx9cm. Bonus context: this is the dataset family in
  which Kuzmina et al. found rotations *absent*.
- **Kalidindi et al. 2021** (eLife 10:e67256; Scott lab) — motor/somatosensory,
  delayed center-out reaching + posture perturbation; Dryad + GitHub. The
  feedback-controller interpretation of rotations.
- **Gallego et al. 2022** (M1/PMd/area 2 LFP, delay center-out) — Dryad.
- All five rotation datasets (incl. Churchland 2012 and Mante 2013 PFC) are
  batch-downloadable via `prepare_data.sh` in
  github.com/NevVerVer/neural-dynamics-gyration (their own analysis code and
  tutorials included).

## References

- Churchland et al. 2012, *Nature* — original jPCA + maze-task data
  (official MATLAB package + exampleData.mat: churchland.zuckermaninstitute.
  columbia.edu/content/code; see also their caveat recommending the HDR
  method, Lara et al. 2018, when interpretability matters).
- bantin/jPCA (GitHub) — standard Python port of the MATLAB codepack, with
  `load_churchland_data`.
- nwb4edu Lesson 5 notebook — teaching walkthrough on the same data.
- Lebedev et al. 2019, *eNeuro*; Kuzmina, Kriukov & Lebedev 2024,
  *Sci. Rep.* 14:3566 (+ code: NevVerVer/neural-dynamics-gyration);
  Michaels et al. 2016, *PLoS Comput. Biol.*; Sussillo et al. 2015,
  *Nat. Neurosci.*
