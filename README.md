# Dream2Image: EEG before awakening (ES606 Computational Neuroscience)

Analysis of [Dream2Image](https://huggingface.co/datasets/opsecsystems/Dream2Image) (Bellec, 2025): EEG recorded
in the minutes before a person is woken, paired with the dream report they gave on waking.

## Question

A dream report is given at the moment of waking. Does the EEG change in the last minute before that moment, and
does the final 15 s carry more information about the content of the report than earlier windows?

- **H1:** within an awakening, posterior delta power and the aperiodic exponent fall from E2 to E4, as the brain
  moves toward waking.
- **H2:** report content is decodable from E4 (the last 15 s) but not from earlier epochs.

The **aperiodic exponent** is the slope of the 1/f background of the power spectrum on log-log axes, after
oscillatory peaks are removed (Donoghue et al., 2020). A steeper (larger) exponent goes with deeper sleep and
lower arousal, a flatter one with waking (Lendner et al., 2020). H1 therefore predicts a flattening. Posterior
channels follow the pre-awakening "hot zone" of Siclari et al. (2017).

## Data

- 129 recordings; 122 kept from 37 participants (2 rows have 6 channels, 5 have windows that are not the tail of
  `eeg_120s`). One dream report per recording. The `image` column is never used: the images were generated from
  the transcripts and carry no brain information.
- Two recording sites, named after their title families: `case` (24 rows, full 120 s, broadband) and `subject`
  (105 rows, Cz dead). Their spectra separate completely above 45 Hz, so all analyses stay
  at 0.5-30 Hz and use the 16 channels present in both.
- `eeg_120s` is z-scored per channel, and the 15/30/60 s windows are exact copies of its tail. Absolute power
  cannot be recovered, so only change within an awakening is tested.
- Awakening types among kept recordings: sleep onset (SO) 70, morning 12, REM 10, NREM 8, untagged 22 (all
  `case` site). SO awakenings exist only in the `subject` site.
- For scale: the DREAM database (Wong et al., 2025) pools 2643 awakenings from 505 participants. Extending this
  analysis to it is future work.

## Epochs

The released windows (last 15, 30, 60 and 120 s) are nested, so they cannot be compared directly. Each one is
cut into a bin, and the 15 s closest to awakening inside each bin is kept, giving four equal-length epochs:

| Epoch | Bin | Epoch kept | Available for |
|---|---|---|---|
| E1 | [-120, -60) s | [-75, -60) s | `case` site only (22 awakenings): exploratory |
| E2 | [-60, -30) s | [-45, -30) s | all |
| E3 | [-30, -15) s | [-30, -15) s | all |
| E4 | [-15, 0) s | [-15, 0) s | all |

All tests use E2-E4.

## Methods

1. **Preprocessing (NB2).** 0.5-30 Hz FIR filter, broken channels interpolated from neighbours, average reference,
   noisy 2-s sub-windows rejected with one robust threshold per recording, so E4 (which carries pre-waking
   movement) is judged by the same rule as E2.
2. **Features (NB3).** Welch spectrum from clean 2-s segments, log band power (delta 1-4, theta 4-8, alpha 8-12,
   sigma 12-16, beta 16-30 Hz) and the aperiodic exponent from specparam fitted at 1-30 Hz.
3. **H1 (NB4).** Linear mixed model per measure: time (epoch centre, per 30 s) + site + artifact rate, random
   intercepts for participant and for awakening within participant, Holm over the two primary tests.
   Checks added later: a random time slope per participant (Barr et al., 2013) and a model-free sign-flip test,
   where one slope per participant is computed and its sign flipped at random to build the null.
4. **H2 (NB5).** Each report becomes a sentence embedding (SBERT `all-mpnet-base-v2`; Reimers & Gurevych, 2019).
   Ridge regression predicts it from 7 spectral measures x 16 channels, leave-one-participant-out. The score is
   **rank accuracy**: for a held-out awakening, the share of other reports that its true report beats in
   similarity to the prediction (0.5 = chance). It is tested against two permutation nulls: a **free** shuffle of
   reports across all awakenings, and a shuffle only **within** the same site x awakening type, which EEG beats
   only if it carries content beyond the category. E4 vs E2 is tested with a sign-flip test and with TOST
   equivalence, bound +/- 0.05 (Lakens, 2017).
5. **Controls (NB6).** H1 without the 4 recordings whose scale is set by an artifact, and within one site;
   artifact rate over time; decoding from artifact rate alone.
6. **Benchmarks.** NB7 (H2): a Riemannian tangent-space decoder on channel covariances (Barachant et al., 2012),
   with everything else as in NB5. NB8 (H1): the exponent re-estimated by IRASA (Wen & Liu, 2016) and by
   specparam refitted on the same 1.5-20 Hz range, with the NB4 tests unchanged.

The type split, the confound control, the random-slope checks and both benchmarks were added after the first
results; this is marked in the notebooks and in `config.yaml`.

## Results

**H1 is not supported.** Pooled over 118 awakenings (37 participants), the exponent gets steeper toward waking,
the opposite of H1, and the effect is fragile. It is carried by sleep-onset awakenings, which are still
descending into sleep when they are interrupted.

| | Result | Table / figure |
|---|---|---|
| H1 | Exponent steeper toward awakening (+0.043 per 30 s, Holm p = .017); delta n.s. (p = .065) | `h1_primary.csv`, `h1_trend` |
| H1 by awakening type (post hoc) | Confined to SO awakenings (n = 68: exponent +0.096, p < .0001; alpha and beta fall, delta rises); other awakenings flat | `h1_posthoc_awakening_type.csv` |
| H1 sensitivity | With a random slope the pooled effect is not significant (p = .075; sign-flip p = .26). It mixes SO steepening (p <= .002 in every check) with flattening in the `case` site (mixed models p = .023, sign-flip p = .19) and no change in `subject` non-SO awakenings | `h1_sensitivity.csv` |
| H1 controls | Holds without flagged recordings and within one site. Artifact rate is flat pooled (p = .82) and in the `subject` site (p = .33), with a non-significant rise in the `case` site (p = .079; E4 26% rejected vs about 15% earlier) | `controls.csv` |
| H1 benchmark (NB8) | IRASA and the 1.5-20 Hz specparam refit agree with NB4 (r = .87-.93). SO steepening holds with every estimator (IRASA: random slope p = .0002, sign-flip p = .002); the pooled effect stays fragile (sign-flip p = .18-.22). Same 262 epochs (107 awakenings) for all estimators | `h1_benchmark.csv`, `h1_benchmark_agreement.csv`, `benchmark_h1` |
| Report richness (exploratory) | Does not change the slope (p = .52 and .53) | `rq3_richness.csv` |

**H2 is not supported.** Over 101 awakenings (33 participants), EEG decodes reports above the free null, equally in
every epoch, but no better than site and awakening type alone.

| | Result | Table / figure |
|---|---|---|
| H2 | Rank accuracy about 0.59 in E2, E3 and E4 (Holm p = .003); no E4 advantage (E4 - E2 = 0.004, TOST p < .001) | `h2_decoding_by_epoch.csv`, `h2_e4_vs_e2.csv` |
| H2 confound control | Site and awakening-type labels alone score 0.65, above EEG; against the within-category null EEG is at chance (p = .65-.81) | `h2_confound_control.csv`, `h2_decoding` |
| H2 benchmark (NB7) | Riemannian decoder 0.53-0.54, not above the free null (Holm p = .28) and at chance within category (p = .92-.96) | `benchmark_h2.csv`, `benchmark_h2` |

Tables are in `results/tables/`, figures (PDF and PNG) in `results/figures/`, executed notebooks with all outputs in
`results/notebooks/`, run manifests (runtime, versions, seed, config and output hashes) in `results/logs/`.

## Limitations

- Small sample: 122 awakenings from 37 people, and small groups once split by awakening type.
- z-scored data, 16 channels, 0.5-30 Hz only: no absolute power and no gamma.
- Site is confounded with hardware, with E1 availability and with awakening type (SO exists in one site only).
- The analyses that explain the null results (type split, category null, benchmarks) were not pre-specified.

## Run order

| Notebook | What it does | Where |
|---|---|---|
| `00_setup` | Hardware check, extras install, gated-access check, version log | Colab |
| `01_data` | Download, per-row cache, structural checks, exclusions, writes `derived/step0.json` | Colab |
| `02_preprocess` | Epochs E1-E4, filtering, interpolation, artifact rejection, writes `derived/clean/` | Colab |
| `03_features` | Band power and aperiodic exponent per epoch and channel | Colab |
| `04_h1` | Mixed models for H1, report richness, split by awakening type, slope sensitivity | Colab or local |
| `05_h2` | Report embeddings, decoding, permutation nulls, E4 vs E2, confound control | Colab (embeddings), then Colab or local |
| `06_controls` | Robustness controls; zips `results/` | Colab or local |
| `07_benchmark` | Riemannian decoder against the same two nulls as NB5 | Colab or local |
| `08_benchmark_h1` | Exponent by IRASA and by specparam on 1.5-20 Hz, with the NB4 tests | Colab or local |

Each notebook reads only what earlier ones wrote, so the chain can be resumed at any step. Every parameter is in
`config.yaml`.

On Colab: for NB0-NB1 only, add `HF_TOKEN` (Hugging Face read token) to Colab Secrets. Open
`notebooks/run_pipeline.ipynb`, set `NOTEBOOKS`, Run all. Data and outputs go to `MyDrive/ES606_Dream2Image/`.

Locally: NB4-NB8 skip the raw EEG (NB7 and NB8 read the cleaned epochs in `derived/clean/`) but need, from the
Drive folder: `derived/`, the metadata files `raw/s*.json` (report word counts) and the `results/tables/` written
by NB1-NB2.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-local.txt
D2I_DATA=/path/to/ES606_Dream2Image .venv/bin/python -c "from d2i import runner; runner.run(['04_h1.ipynb'])"
```

The runner pins BLAS to one thread: the matrices are small and multi-threaded BLAS is much slower here.

Full run from scratch on a laptop (download included, about 10 minutes; needs a Hugging Face login with access to
the dataset):

```bash
python3 -m venv .venv-full && .venv-full/bin/pip install -r requirements-local.txt -r requirements-colab.txt torch transformers pyarrow huggingface_hub
D2I_DATA=/path/to/empty_folder .venv-full/bin/python -c "from d2i import runner; runner.run(['00_setup.ipynb', '01_data.ipynb', '02_preprocess.ipynb', '03_features.ipynb', '04_h1.ipynb', '05_h2.ipynb', '06_controls.ipynb', '07_benchmark.ipynb', '08_benchmark_h1.ipynb'])"
.venv-full/bin/python tools/compare_results.py /path/to/empty_folder/results/tables
```

Rerun checks: fresh clone, empty data folder, NB0-NB8 on macOS with Python 3.12.7, at `eb44907` (2026-11-07) and
again at `04991f9` (2026-11-12). All nine notebooks ran without errors both times. All 26 tables have the same rows,
columns and text; values differ by at most 0.007, except one `case` site random-slope model for alpha in
`h1_sensitivity.csv` (p .887 vs .836). These come from permutation p-values moving by one permutation and from small
mixed models converging slightly differently on another platform. Every number quoted above is unchanged at the
precision shown. Per-table differences for the first rerun: `results/logs/rerun_comparison.csv`.

## Layout

```
config.yaml            every parameter, one place
d2i/                   shared code imported by the notebooks
notebooks/             built from tools/build_notebooks.py (edit cells there, then rebuild)
results/               tables, figures, executed notebooks, logs
tools/                 notebook builder, rerun comparison
```

## References

- Bellec, Y. (2025). Dream2Image: An open multimodal EEG dataset for decoding and visualizing dreams with artificial intelligence. arXiv:2510.06252.
- Siclari, F. et al. (2017). The neural correlates of dreaming. *Nature Neuroscience* 20, 872-878.
- Donoghue, T. et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience* 23, 1655-1665.
- Lendner, J. D. et al. (2020). An electrophysiological marker of arousal level in humans. *eLife* 9, e55092.
- Wong, W. et al. (2025). A dream EEG and mentation database. *Nature Communications* 16, 7495.

Methods:

- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP 2019*, 3982-3992.
- Lakens, D. (2017). Equivalence tests: a practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science* 8, 355-362.
- Barr, D. J. et al. (2013). Random effects structure for confirmatory hypothesis testing: keep it maximal. *Journal of Memory and Language* 68, 255-278.
- Barachant, A. et al. (2012). Multiclass brain-computer interface classification by Riemannian geometry. *IEEE Transactions on Biomedical Engineering* 59, 920-928.
- Wen, H. & Liu, Z. (2016). Separating fractal and oscillatory components in the power spectrum of neurophysiological signal. *Brain Topography* 29, 13-26.
