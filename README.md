# Dream2Image: (ES606) Computational Neuroscience project

Analysis of [Dream2Image](https://huggingface.co/datasets/opsecsystems/Dream2Image) (Bellec, 2025):
EEG recorded before awakening, paired with dream reports.

**Question.** How does pre-awakening EEG change over the last ~75 s before waking, and does the final 15 s
carry more dream-content information than earlier windows?

- **H1:** posterior delta power and the aperiodic exponent fall from E2 to E4.
- **H2:** dream-report content is decodable from E4 but not from earlier epochs.

Epochs are the last 15 s of each bin before awakening: E1 [-75, -60) s, E2 [-45, -30) s, E3 [-30, -15) s,
E4 [-15, 0) s, all cut from `eeg_120s`.

## Results

| | Result | Table / figure |
|---|---|---|
| H1 | Not supported. Exponent becomes steeper toward awakening (Holm p = .017); delta n.s. (p = .065) | `h1_primary.csv`, `h1_trend` |
| H1, by awakening type (post hoc) | The change is confined to sleep-onset awakenings (n = 68; exponent p < .0001, alpha and beta fall, delta rises); other awakenings flat | `h1_posthoc_awakening_type.csv` |
| H1 sensitivity (added after review) | With a random time slope per participant the pooled exponent effect is not significant (p = .075; sign-flip over participants p = .26). It mixes sleep-onset steepening (robust in every check, p ≤ .002) with flattening in the `case` site (mixed models p = .023, but sign-flip p = .19) and no change in `subject` non-SO awakenings | `h1_sensitivity.csv` |
| H1 controls | Holds without flagged samples and within one site. Artifact rate: flat pooled (p = .82) and in the `subject` site (p = .33), a non-significant rise toward awakening in the `case` site (p = .079; E4 26% rejected vs about 15% earlier) | `controls.csv` |
| Report richness (exploratory) | Does not moderate the slope (p ≈ .52) | `rq3_richness.csv` |
| H2 | Not supported. Decoding above a free-shuffle null in every epoch (Holm p = .003) with no E4 advantage (E4 - E2 = 0.004, equivalence p < .001) | `h2_decoding_by_epoch.csv`, `h2_e4_vs_e2.csv` |
| H2 confound control (post hoc) | Site and awakening-type labels alone decode better (0.65) than EEG (0.59); against a within-category null, EEG is at chance (p = .65-.81) | `h2_confound_control.csv`, `h2_decoding` |
| H2 benchmark (added after H2) | A Riemannian tangent-space decoder (channel covariances) does not beat the free-shuffle null (0.53-0.54, Holm p = .28) and is at chance within category (p = .92-.96): neither EEG representation decodes content beyond site and awakening type | `benchmark_h2.csv`, `benchmark_h2` |

Tables are in `results/tables/`, figures (PDF and PNG) in `results/figures/`, executed notebooks with all
outputs in `results/notebooks/`, run manifests in `results/logs/`.

## Data facts established in 01_data

- 129 rows; 122 retained from 37 participants (2 rows have 6 channels, 5 have windows that are not the tail of `eeg_120s`).
- `eeg_120s` is z-scored per channel, and the 15/30/60 s windows are exact copies of its tail: absolute power is not recoverable.
- Two recording sites (title families `case` / `subject`) separate completely above 45 Hz; analyses stay below 30 Hz.
- Cz is dead in every `subject` recording; 16 primary channels.
- Awakening types among retained samples: sleep onset 70, morning 12, REM 10, NREM 8, untagged (`case` site) 22.
- The `image` column is never loaded: images were generated from the transcripts and carry no brain information.

## Run order

| Notebook | What it does | Where |
|---|---|---|
| `00_setup` | Hardware check, extras install, gated-access check, version log | Colab |
| `01_data` | Download, per-row cache, structural checks, exclusions → `derived/step0.json` | Colab |
| `02_preprocess` | Epochs E1-E4, 0.5-30 Hz filter, interpolation, artifact rejection → `derived/clean/` | Colab |
| `03_features` | Band power and specparam aperiodic exponent per epoch × channel | Colab |
| `04_h1` | Mixed models for H1, richness moderation, post hoc split by awakening type, slope sensitivity | Colab or local |
| `05_h2` | Report embeddings (SBERT), leave-one-participant-out ridge decoding, permutation nulls, confound control | Colab (embeddings), then Colab or local |
| `06_controls` | Robustness controls; zips `results/` | Colab or local |
| `07_benchmark` | Riemannian tangent-space decoder against the same two nulls as NB5 | Colab or local |

**Colab.** For NB0-NB1 only, add to Colab Secrets: `HF_TOKEN` (Hugging Face read token). Open `notebooks/run_pipeline.ipynb`,
set `NOTEBOOKS`, Run all. Data and outputs go to `MyDrive/ES606_Dream2Image/`.

**Local.** NB4-NB7 skip the raw EEG (NB7 reads the cleaned epochs in `derived/clean/`) but need, from the Drive folder: `derived/`, the metadata files
`raw/s*.json` (report word counts) and the `results/tables/` written by NB1-NB2.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-local.txt
D2I_DATA=/path/to/ES606_Dream2Image .venv/bin/python -c "from d2i import runner; runner.run(['04_h1.ipynb'])"
```

The runner pins BLAS to one thread: the matrices are small and multi-threaded BLAS is much slower here.

## Layout

```
config.yaml            every parameter, one place
d2i/                   shared code imported by the notebooks
notebooks/             built from tools/build_notebooks.py (edit cells there, then rebuild)
results/               tables, figures, executed notebooks, logs
```

## References

- Bellec, Y. (2025). Dream2Image: an open multimodal EEG dataset for decoding and visualizing dreams with AI. arXiv:2510.06252.
- Siclari, F. et al. (2017). The neural correlates of dreaming. *Nature Neuroscience* 20, 872-878.
- Donoghue, T. et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience* 23, 1655-1665.
- Wong, W. et al. (2025). A dream EEG and mentation database. *Nature Communications* 16, 7495.

Methods used in the checks:

- Barr, D. J. et al. (2013). Random effects structure for confirmatory hypothesis testing: keep it maximal. *Journal of Memory and Language* 68, 255-278.
- Barachant, A. et al. (2012). Multiclass brain-computer interface classification by Riemannian geometry. *IEEE Transactions on Biomedical Engineering* 59, 920-928.
