"""Write notebooks/*.ipynb from the cell lists below (keeps the bootstrap cell identical everywhere).

Edit cells here, then: python tools/build_notebooks.py
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

BOOTSTRAP = '''# Bootstrap: same cell in every notebook. On Colab: mount Drive, get the repo, install extras.
# The repo is private: add a read-only GitHub token to Colab Secrets as GH_TOKEN.
import os, sys, subprocess, base64
REPO_URL = "https://github.com/sanjanasinghx/dream2image-es606.git"
if "google.colab" in sys.modules and not os.environ.get("D2I_HEADLESS"):  # headless = run by d2i.runner
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    REPO = "/content/dream2image-es606"
    try:
        auth = base64.b64encode(f"x-access-token:{userdata.get('GH_TOKEN')}".encode()).decode()
        git = ["git", "-c", f"http.extraHeader=Authorization: Basic {auth}"]  # token never written to disk
    except Exception:
        git = ["git"]  # no secret: works once the repo is public
    if not os.path.exists(REPO):
        subprocess.run([*git, "clone", "-q", REPO_URL, REPO], check=True)
    else:
        subprocess.run([*git, "-C", REPO, "pull", "-q"], check=True)
    freeze = lambda: subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    before = freeze()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{REPO}/requirements-colab.txt"], check=True)
    if freeze() != before:  # e.g. numpy upgraded under an already-loaded copy -> broken imports
        print("Packages changed: restarting the runtime. When it reconnects, Run all again.")
        os.kill(os.getpid(), 9)
else:
    REPO = os.path.abspath("..")
sys.path.insert(0, REPO)
from d2i import env
cfg = env.load_config()
rng = env.seed_everything(cfg["seed"])
print("data root:", env.data_root())'''


def md(s): return ("markdown", s)
def code(s): return ("code", s)


NB00 = [
    md("""# 00 · Setup and runtime check

Run once per new Colab runtime type, or whenever something breaks.
Checks the hardware, installs extras, confirms gated Hugging Face access, and logs exact package versions to Drive.

**Before running:** in Colab, open the key icon (Secrets) and add, with notebook access allowed:
- `HF_TOKEN`: a *read* token from huggingface.co/settings/tokens
- `GH_TOKEN`: a fine-grained GitHub token (github.com/settings/personal-access-tokens), repository access =
  only `dream2image-es606`, permission Contents = Read-only. Not needed once the repo is public. You must have accepted the dataset's access conditions on its Hugging Face page."""),
    code(BOOTSTRAP),
    code('''# Hardware
import psutil, shutil
print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB, CPUs: {os.cpu_count()}")
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True) if shutil.which("nvidia-smi") else None
print("GPU:", gpu.stdout.strip() if gpu and gpu.returncode == 0 else "none (fine for NB1-NB3)")
print(f"Drive free: {shutil.disk_usage(env.data_root()).free / 1e9:.1f} GB")'''),
    code('''# Gated dataset access
from huggingface_hub import HfApi
api = HfApi()
api.auth_check(cfg["dataset"]["hf_id"], repo_type="dataset", token=env.hf_token())
files = [f for f in api.list_repo_files(cfg["dataset"]["hf_id"], repo_type="dataset", token=env.hf_token())]
print("access OK; files:", files)'''),
    code('''# Log versions
import importlib, json, platform
pkgs = ["numpy", "scipy", "pandas", "sklearn", "mne", "yasa", "specparam", "torch",
        "transformers", "pyarrow", "huggingface_hub", "matplotlib"]
versions = {"python": platform.python_version()}
for p in pkgs:
    try:
        versions[p] = importlib.import_module(p).__version__
    except Exception as e:
        versions[p] = f"MISSING ({type(e).__name__})"
print(json.dumps(versions, indent=1))
out = env.path(cfg, "results") / "logs" / "environment.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"versions": versions, "gpu": gpu.stdout.strip() if gpu else None}, indent=2))
print("wrote", out)'''),
]

NB01 = [
    md("""# 01 · Data: download, cache, structural checks

Downloads the three parquet shards (about 1.15 GB) to Drive once, caches each row as arrays, then re-runs the
structural checks from the October pipeline. Every number that later notebooks rely on (retained samples,
primary channels, flagged samples) is written to `derived/step0.json`.

Runtime: CPU, High-RAM is not needed. First run about 10 minutes (mostly the download); reruns skip what is cached."""),
    code(BOOTSTRAP),
    code('''import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from d2i import data, checks
tables = env.path(cfg, "results") / "tables"; tables.mkdir(exist_ok=True)
figs = env.path(cfg, "results") / "figures"; figs.mkdir(exist_ok=True)
load = lambda sid, col=None: data.load_eeg(cfg, sid, col)'''),
    md("## 1. Download and cache (resumable)"),
    code('''shards = data.download(cfg)
print([s.name for s in shards])
n = data.cache_rows(cfg, shards)
metas = data.load_meta(cfg)
assert len(metas) == cfg["dataset"]["expected_n_rows"], f"{len(metas)} rows cached"'''),
    md("## 2. Is every EEG column z-scored on its own?\nIf yes, absolute power is gone: only spectral *shape* can be compared across samples."),
    code('''scaling = checks.scaling_table(cfg, metas, load)
scaling.to_csv(tables / "scaling_check.csv", index=False)
live = scaling[scaling.sd > cfg["dataset"]["dead_sd_floor"]]
print(live.groupby("source")[["is_z"]].mean().rename(columns={"is_z": "frac_live_channels_z_scored"}))'''),
    md("## 3. Are the 15/30/60 s windows the tail of the 120 s recording?"),
    code('''nest = checks.nesting_table(metas)
nest.to_csv(tables / "nesting_check.csv", index=False)
counts = nest.groupby(["window_s", "verdict"]).size().unstack(fill_value=0)
print(counts)
mis_frac = (nest.verdict == "misaligned").mean()
assert mis_frac <= cfg["checks"]["misaligned_stop_fraction"], f"{mis_frac:.1%} misaligned: stop and inspect"'''),
    md("## 4. Participants, sites and exclusions"),
    code('''parts = checks.participants_table(metas, data.parse_title)
parts.to_csv(tables / "participants.csv", index=False)
excl = checks.exclusions_table(cfg, metas, nest, parts)
excl.to_csv(tables / "sample_exclusions.csv", index=False)
ret = excl[excl.retained]
print(f"participants: {parts.participant.nunique()} (paper: {cfg['participant']['expected_n']})")
print(f"retained {len(ret)}/{len(excl)} samples from {ret.participant.nunique()} participants")
print(ret.groupby("site").agg(samples=("sample_id", "size"), participants=("participant", "nunique")))
print("awakening tags (subject site):", ret.tag_class.fillna("none").value_counts().to_dict())
print("excluded:\\n", excl.loc[~excl.retained, ["sample_id", "title", "reasons"]].to_string(index=False))
assert parts.participant.nunique() == cfg["participant"]["expected_n"]'''),
    md("## 5. Primary channel set\nChannels live in at least 95% of retained samples (Cz is dead in every `subject`-site recording)."),
    code('''avail = checks.channel_availability(cfg, scaling, excl)
avail.to_csv(tables / "channel_availability.csv", index=False)
primary = avail.loc[avail.primary, "channel"].tolist()
print(avail.to_string(index=False))
print("primary:", primary)'''),
    md("## 6. Is the undocumented channel order spatially plausible?"),
    code('''order_res, corr = checks.channel_order_check(cfg, metas, load, rng)
corr.to_csv(tables / "channel_corr_mean.csv")
print(order_res)'''),
    md("## 7. Site differences in the spectrum\nIf the sites separate completely above 45 Hz, anything above ~30 Hz is a site marker, not brain signal."),
    code('''hf, f, spectra = checks.site_spectra(cfg, metas, parts, load)
hf.to_csv(tables / "site_hf_fraction.csv", index=False)
by = hf.groupby("site")["log10_frac_45_100"].agg(["count", "min", "median", "max"])
print(by)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
colors = {"case": "#c0504d", "subject": "#4a6fa5"}
m = f <= 100
for s, c in colors.items():
    sel = (hf.site == s).values
    ax[0].hist(hf.log10_frac_45_100[sel], bins=25, alpha=.7, color=c, label=f"{s} (n={sel.sum()})")
    ax[1].plot(f[m], spectra[sel][:, m].T, color=c, alpha=.08, lw=.6)
    ax[1].plot(f[m], spectra[sel][:, m].mean(0), color=c, lw=2, label=s)
ax[0].set(xlabel="log10 P(45-100 Hz) / P(1-45 Hz)", ylabel="samples", title="High-frequency power by site")
ax[1].set(xlabel="Hz", ylabel="log10 PSD (z-units²/Hz)", title="Channel-mean spectra, eeg_120s")
for a in ax: a.legend()
fig.tight_layout(); fig.savefig(figs / "site_spectra.pdf"); fig.savefig(figs / "site_spectra.png", dpi=150)'''),
    md("## 8. Per-epoch scale (sensitivity flags)"),
    code('''so = checks.scaling_outliers(cfg, excl, load)
so.to_csv(tables / "scaling_outliers.csv", index=False)
flagged = so.loc[so.retained & so.scaling_flag, "sample_id"].tolist()
print("flagged for sensitivity analysis:", flagged)
print("samples with full 120 s (E1 available):", int((so.retained & (so.len_s >= 120)).sum()))'''),
    md("## 9. Save the decisions later notebooks use"),
    code('''step0 = {
    "n_rows": len(metas),
    "retained": ret.sample_id.tolist(),
    "n_participants_retained": int(ret.participant.nunique()),
    "primary_channels": primary,
    "scaling_flagged": flagged,
    "nesting_counts": {str(k): v for k, v in counts.to_dict().items()},
    "channel_order": order_res,
    "site_hf": by.round(3).to_dict(orient="index"),
}
out = env.path(cfg, "derived") / "step0.json"
out.write_text(json.dumps(step0, indent=2, default=str))
env.write_manifest(cfg, "01_data", [out, *sorted(tables.glob("*.csv"))])
print(json.dumps({k: v for k, v in step0.items() if k != "retained"}, indent=1, default=str))'''),
]


NB02 = [
    md("""# 02 · Preprocessing

Cuts E1-E4 (the last 15 s of each bin before awakening) from `eeg_120s` and cleans them with the October
pipeline: 0.5-30 Hz FIR, bad-channel interpolation, average re-reference, and 2-s sub-window rejection with one
robust threshold per sample, so E4 (which carries pre-waking movement) is judged by the same rule as E2.

Output: `derived/clean/<sample>.npz` (signal + rejected-sample mask per epoch), `results/tables/reject_rates.csv`.
Runtime: CPU, a few minutes. Resumable."""),
    code(BOOTSTRAP),
    code('''import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from joblib import Parallel, delayed
from d2i import data, preprocess
step0 = json.loads((env.path(cfg, "derived") / "step0.json").read_text())
channels = [c for c in step0["primary_channels"] if c not in cfg["preprocess"]["drop_channels"]]
out_dir = env.path(cfg, "derived") / "clean"; out_dir.mkdir(exist_ok=True)
tables = env.path(cfg, "results") / "tables"
figs = env.path(cfg, "results") / "figures"
print(len(step0["retained"]), "samples,", len(channels), "channels")'''),
    code('''def run(sid):
    arrays, recs, chans = preprocess.process_sample(cfg, sid, data.load_eeg(cfg, sid), channels)
    np.savez_compressed(out_dir / f"{sid}.npz", channels=np.array(channels), **arrays)
    pd.DataFrame(recs).to_csv(out_dir / f"{sid}_epochs.csv", index=False)
    pd.DataFrame(chans).to_csv(out_dir / f"{sid}_channels.csv", index=False)
    return sid

todo = [s for s in step0["retained"] if not (out_dir / f"{s}_channels.csv").exists()]
print(f"{len(todo)} samples to process")
Parallel(n_jobs=-1, verbose=5)(delayed(run)(s) for s in todo)'''),
    code('''excl = pd.read_csv(tables / "sample_exclusions.csv")
rej = pd.concat([pd.read_csv(out_dir / f"{s}_epochs.csv") for s in step0["retained"]])
rej = rej.merge(excl[["sample_id", "participant", "site", "tag_class"]], on="sample_id")
chq = pd.concat([pd.read_csv(out_dir / f"{s}_channels.csv") for s in step0["retained"]])
rej.to_csv(tables / "reject_rates.csv", index=False)
chq.to_csv(tables / "channel_quality.csv", index=False)
summary = rej.groupby("epoch").agg(n=("reject_rate", "size"), reject_mean=("reject_rate", "mean"),
                                   frac_any_reject=("reject_rate", lambda s: (s > 0).mean()),
                                   interp_mean=("n_interp", "mean"), unusable=("usable", lambda s: (~s).sum()))
summary.round(3).to_csv(tables / "reject_summary.csv")
print(summary.round(3))
print("sample-level bad channels:", chq.bad_reason.value_counts().to_dict())'''),
    code('''fig, ax = plt.subplots(figsize=(6, 4))
order = ["E1", "E2", "E3", "E4"]
ax.boxplot([rej.loc[rej.epoch == e, "reject_rate"] for e in order], tick_labels=order, showfliers=False)
for i, e in enumerate(order, 1):
    y = rej.loc[rej.epoch == e, "reject_rate"]
    ax.scatter(np.random.default_rng(0).normal(i, .06, len(y)), y, s=6, alpha=.4, color="#4a6fa5")
ax.set(ylabel="fraction of epoch rejected", title="Artifact rejection by epoch (E4 = last 15 s)")
fig.tight_layout(); fig.savefig(figs / "reject_by_epoch.pdf"); fig.savefig(figs / "reject_by_epoch.png", dpi=150)
env.write_manifest(cfg, "02_preprocess", [tables / "reject_rates.csv", tables / "channel_quality.csv",
                                          tables / "reject_summary.csv"],
                   notes={"n_clean_files": len(list(out_dir.glob("s*.npz"))), "channels": channels})'''),
]

NB03 = [
    md("""# 03 · Spectral features per epoch

For every sample × epoch × channel: the power spectrum from clean 4-s segments, log band power
(delta, theta, alpha, sigma, beta), and the aperiodic exponent and offset from specparam
(Donoghue et al. 2020), fitted 1-30 Hz. The posterior average (O1, O2, Oz, P3, P4, Pz) follows the
pre-awakening "hot zone" of Siclari et al. (2017).

Output: `derived/features_long.csv`, `derived/features_roi.csv`, `derived/spectra.npz`.
Runtime: CPU, a few minutes. No statistics here: tests come in NB4."""),
    code(BOOTSTRAP),
    code('''import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from d2i import features
fc, fs = cfg["features"], cfg["dataset"]["sfreq"]
step0 = json.loads((env.path(cfg, "derived") / "step0.json").read_text())
clean = env.path(cfg, "derived") / "clean"
tables = env.path(cfg, "results") / "tables"
figs = env.path(cfg, "results") / "figures"
rej = pd.read_csv(tables / "reject_rates.csv")'''),
    md("## 1. Spectra and band power"),
    code('''rows, spectra = [], []
for _, r in rej.iterrows():
    z = np.load(clean / f"{r.sample_id}.npz")
    chans = list(z["channels"])
    f, psd, n_seg = features.clean_psd(z[r.epoch], z[f"{r.epoch}_bad"], fs, fc["welch_win_s"], fc["welch_overlap_s"])
    ok = r.usable and n_seg >= fc["min_clean_segments"]
    bp = features.band_power(f, psd, fc["bands"]) if ok else {}
    for i, c in enumerate(chans):
        rows.append({**r[["sample_id", "participant", "site", "tag_class", "epoch", "center_s",
                          "reject_rate"]].to_dict(),
                     "channel": c, "n_clean_seg": n_seg, "ok": ok,
                     **{k: (v[i] if ok else np.nan) for k, v in bp.items()}})
        spectra.append(psd[i] if ok else np.full(len(f), np.nan))
feat = pd.DataFrame(rows)
fmask = (f >= 0.5) & (f <= 30)
S = np.array(spectra)
print(feat.groupby("epoch").ok.agg(["size", "mean"]).rename(columns={"size": "rows", "mean": "frac_ok"}))'''),
    md("## 2. Aperiodic exponent (specparam)"),
    code('''good = feat.ok.values
ap = features.fit_aperiodic(f, S[good], fc)
for k, v in ap.items():
    feat.loc[good, k] = v
print(f"fits: {good.sum()}, median R² = {np.median(ap['fit_r2']):.3f}, "
      f"R² < {fc['min_r2']}: {(ap['fit_r2'] < fc['min_r2']).mean():.1%}, median peaks = {np.median(ap['n_peaks']):.0f}")
print(feat.groupby("epoch")[["ap_exponent", "ap_offset"]].median().round(3))'''),
    md("## 3. Save, plus the posterior ROI average"),
    code('''der = env.path(cfg, "derived")
feat.to_csv(der / "features_long.csv", index=False)
np.savez_compressed(der / "spectra.npz", f=f, psd=S, sample_id=feat.sample_id.values,
                    epoch=feat.epoch.values, channel=feat.channel.values)
num = [c for c in feat.columns if c.startswith(("logpow_", "ap_"))] + ["fit_r2"]
keys = ["sample_id", "participant", "site", "tag_class", "epoch", "center_s", "reject_rate"]
roi = (feat[feat.channel.isin(fc["posterior_roi"]) & feat.ok]
       .groupby(keys, dropna=False)[num].mean().reset_index())
roi.to_csv(der / "features_roi.csv", index=False)
print(roi.groupby("epoch")[["logpow_delta", "logpow_beta", "ap_exponent"]].agg(["count", "mean"]).round(3))'''),
    md("## 4. Descriptive look (no inference)"),
    code('''order = ["E1", "E2", "E3", "E4"]
cols = dict(zip(order, ["#bbbbbb", "#9ecae1", "#4292c6", "#08306b"]))
post = feat.channel.isin(fc["posterior_roi"]).values & feat.ok.values
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
# E1 exists only in the 'case' site (different hardware), so the spectrum panel uses those 22 samples
# for every epoch: a within-recording comparison rather than a site comparison.
case = (feat.site == "case").values
for e in order:
    m = post & case & (feat.epoch == e).values
    if m.any():
        ax[0].plot(f[fmask], np.log10(np.nanmean(S[m][:, fmask], 0)), color=cols[e], lw=2, label=e)
ax[0].set(xlabel="Hz", ylabel="log10 power (z-units²/Hz)",
          title=f"Posterior spectrum, 'case' site only (n={feat[case].sample_id.nunique()})"); ax[0].legend()
for a, col, lab in [(ax[1], "logpow_delta", "log10 delta power (1-4 Hz)"), (ax[2], "ap_exponent", "aperiodic exponent")]:
    d = roi[roi.epoch.isin(["E2", "E3", "E4"])].pivot_table(index="sample_id", columns="epoch", values=col)
    a.plot(range(3), d[["E2", "E3", "E4"]].T, color="#4a6fa5", alpha=.15)
    a.plot(range(3), d[["E2", "E3", "E4"]].mean(), color="k", lw=2.5, marker="o")
    a.set(xticks=range(3), xticklabels=["E2\\n(-45..-30 s)", "E3\\n(-30..-15 s)", "E4\\n(-15..0 s)"],
          ylabel=lab, title=f"Posterior {lab.split(' (')[0]}, one line per awakening")
fig.tight_layout(); fig.savefig(figs / "features_by_epoch.pdf"); fig.savefig(figs / "features_by_epoch.png", dpi=150)
env.write_manifest(cfg, "03_features", [der / "features_long.csv", der / "features_roi.csv"],
                   notes={"n_rows": len(feat), "frac_ok": float(feat.ok.mean()),
                          "median_r2": float(np.median(ap["fit_r2"]))})'''),
]


NB04 = [
    md("""# 04 · H1: does the EEG change as awakening approaches?

**H1:** within participants, posterior delta power and the aperiodic exponent fall from E2 to E4.
Linear mixed model per measure over E2-E4: time (epoch centre, per 30 s) + site + artifact rate,
random intercepts for participant and for awakening within participant. Holm across the two primary tests.

**RQ3 (exploratory):** does the slope depend on report richness (log word count of the transcription)?

Everything below the primary section is exploratory and labelled so."""),
    code(BOOTSTRAP),
    code('''import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from d2i import data, stats
hc = cfg["h1"]
der, tables, figs = env.path(cfg, "derived"), env.path(cfg, "results") / "tables", env.path(cfg, "results") / "figures"
words = {m["sample_id"]: len((m.get("transcription") or "").split()) for m in data.load_meta(cfg)}
def prep(df):
    d = df[df.epoch.isin(hc["epochs"])].copy()
    d["time30"] = d.center_s / 30.0
    d["richness_z"] = np.log(d.sample_id.map(words))
    d["richness_z"] = (d.richness_z - d.richness_z.mean()) / d.richness_z.std()
    return d
roi = prep(pd.read_csv(der / "features_roi.csv"))
long = prep(pd.read_csv(der / "features_long.csv"))
long = long[long.ok]
print(f"ROI rows {len(roi)}, awakenings {roi.sample_id.nunique()}, participants {roi.participant.nunique()}")'''),
    md("## 1. Primary test (confirmatory)"),
    code('''rows = []
for y in hc["primary"]:
    rows.append(stats.fit_trend(roi, y, hc["covariates"]))
    rows.append(stats.fit_trend(roi, y, [c for c in hc["covariates"] if c != "reject_rate"]))
prim = pd.DataFrame(rows)
main = prim.covariates == "+".join(hc["covariates"])
prim.loc[main, "p_holm"] = stats.holm(prim.loc[main, "p"])
prim.to_csv(tables / "h1_primary.csv", index=False)
print(prim[["y", "covariates", "estimate", "ci_low", "ci_high", "std_estimate", "p", "p_holm",
            "n_awakenings", "n_participants", "converged"]].round(4).to_string(index=False))'''),
    md("## 2. RQ3: does the slope depend on report richness? (exploratory)"),
    code('''rq3 = pd.DataFrame([stats.fit_trend(roi, y, hc["covariates"], interaction="richness_z") for y in hc["primary"]])
rq3.to_csv(tables / "rq3_richness.csv", index=False)
print("word count: median", int(np.median(list(words.values()))), "range", min(words.values()), "-", max(words.values()))
print(rq3[["y", "term", "estimate", "ci_low", "ci_high", "p", "converged"]].round(4).to_string(index=False))'''),
    md("## 3. All bands, posterior ROI, and the channel map (exploratory, FDR)"),
    code('''measures = [c for c in roi.columns if c.startswith("logpow_")] + ["ap_exponent", "ap_offset"]
expl = pd.DataFrame([stats.fit_trend(roi, y, hc["covariates"]) for y in measures])
expl["p_fdr"] = stats.fdr_bh(expl.p)
expl.to_csv(tables / "h1_exploratory_roi.csv", index=False)
print(expl[["y", "std_estimate", "ci_low", "ci_high", "p", "p_fdr"]].round(4).to_string(index=False))

chan = []
for ch, g in long.groupby("channel"):
    for y in ["logpow_delta", "logpow_alpha", "logpow_beta", "ap_exponent"]:
        chan.append({"channel": ch, **stats.fit_trend(g, y, hc["covariates"])})
chan = pd.DataFrame(chan)
chan["p_fdr"] = stats.fdr_bh(chan.p)
chan.to_csv(tables / "h1_exploratory_channels.csv", index=False)
print("channel x measure tests surviving FDR:", int((chan.p_fdr < hc["alpha"]).sum()), "of", len(chan))'''),
    md("## 4. Figure"),
    code('''fig, ax = plt.subplots(1, 3, figsize=(15, 4))
labels = {"logpow_delta": "posterior log10 delta power", "ap_exponent": "posterior aperiodic exponent"}
for a, y in zip(ax[:2], hc["primary"]):
    d = roi.dropna(subset=[y]).copy()
    d["centered"] = d[y] - d.groupby("sample_id")[y].transform("mean")  # within-awakening change
    m = d.groupby("center_s").centered.agg(["mean", "sem"])
    a.errorbar(m.index, m["mean"], yerr=1.96 * m["sem"], marker="o", color="k", capsize=4)
    a.axhline(0, color="grey", lw=.8, ls=":")
    r = prim[(prim.y == y) & main].iloc[0]
    a.set(xlabel="epoch centre (s before awakening)", ylabel=f"{labels[y]}\\n(within-awakening, mean ± 95% CI)",
          title=f"slope {r.estimate:+.3f} per 30 s, p_holm = {r.p_holm:.3g}")
piv = chan.pivot(index="channel", columns="y", values="std_estimate")
im = ax[2].imshow(piv.values, cmap="RdBu_r", vmin=-.3, vmax=.3, aspect="auto")
sig = chan.pivot(index="channel", columns="y", values="p_fdr").values < hc["alpha"]
for i, j in zip(*np.where(sig)):
    ax[2].text(j, i, "*", ha="center", va="center")
ax[2].set(xticks=range(piv.shape[1]), xticklabels=piv.columns, yticks=range(piv.shape[0]), yticklabels=piv.index,
          title="Exploratory: standardised slope per 30 s (* FDR < .05)")
plt.colorbar(im, ax=ax[2])
fig.tight_layout(); fig.savefig(figs / "h1_trend.pdf"); fig.savefig(figs / "h1_trend.png", dpi=150)
env.write_manifest(cfg, "04_h1", [tables / f for f in ["h1_primary.csv", "rq3_richness.csv",
                                                       "h1_exploratory_roi.csv", "h1_exploratory_channels.csv"]])'''),
    md("""## 5. Post hoc: split by awakening type (exploratory)

**Added after seeing section 1**, so this is an explanation to be tested elsewhere, not a confirmatory result.
Sleep-onset (SO) awakenings wake people shortly after they fall asleep, so their last 45 s may track the
descent into sleep rather than anything that precedes waking. SO awakenings exist only in the `subject` site,
so site is dropped as a covariate within SO."""),
    code('''roi["awakening_type"] = np.where(roi.tag_class == "SO", "sleep onset (SO)", "other")
split = []
for t, g in roi.groupby("awakening_type"):
    cov = [c for c in hc["covariates"] if c != "site" or g.site.nunique() > 1]
    for y in ["logpow_delta", "logpow_alpha", "logpow_beta", "ap_exponent"]:
        split.append({"awakening_type": t, **stats.fit_trend(g, y, cov)})
split = pd.DataFrame(split)
split["p_fdr"] = stats.fdr_bh(split.p)
split.to_csv(tables / "h1_posthoc_awakening_type.csv", index=False)
print(split[["awakening_type", "y", "std_estimate", "ci_low", "ci_high", "p", "p_fdr", "n_awakenings",
             "n_participants"]].round(4).to_string(index=False))

fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
for a, y in zip(ax, ["logpow_delta", "logpow_alpha", "logpow_beta", "ap_exponent"]):
    for t, c in [("sleep onset (SO)", "#c0504d"), ("other", "#4a6fa5")]:
        d = roi[(roi.awakening_type == t)].dropna(subset=[y]).copy()
        d["centered"] = d[y] - d.groupby("sample_id")[y].transform("mean")
        m = d.groupby("center_s").centered.agg(["mean", "sem"])
        a.errorbar(m.index, m["mean"], yerr=1.96 * m["sem"], marker="o", color=c, capsize=3,
                   label=f"{t} (n={d.sample_id.nunique()})")
    a.axhline(0, color="grey", lw=.8, ls=":")
    a.set(xlabel="epoch centre (s before awakening)", title=f"posterior {y}", ylabel="within-awakening change")
ax[0].legend(fontsize=8)
fig.suptitle("Post hoc (exploratory): pre-awakening change by awakening type")
fig.tight_layout(); fig.savefig(figs / "h1_posthoc_awakening_type.pdf"); fig.savefig(figs / "h1_posthoc_awakening_type.png", dpi=150)'''),
]

NB05 = [
    md("""# 05 · H2: is report content more decodable from the last 15 s?

**H2:** dream-report content is decodable above chance from E4 (last 15 s) but not from earlier epochs.

- **Target:** sentence embedding of each transcription (SBERT `all-mpnet-base-v2`, Reimers & Gurevych 2019; software).
- **Features:** per epoch, the 7 spectral measures from NB3 × 16 channels.
- **Model:** ridge regression, leave-one-participant-out (no participant is ever in train and test).
- **Score:** for each held-out awakening, the percentile rank of its true report among all reports by cosine
  similarity to the prediction. Nominal chance is 0.5, but ridge predictions lie in the span of the training
  reports, which therefore look slightly closer to every prediction than the held-out one does: true chance sits
  a little below 0.5 (about 0.47 with random embeddings). Every comparison is therefore against an **empirical
  chance level**, the same score under the permutation null.
- **Null:** 1000 runs with reports shuffled across awakenings; per-awakening null scores are kept, so any subset
  (e.g. an awakening type) is tested against its own null. **E4 vs E2:** sign-flip test over participants,
  plus an equivalence test (TOST, ±0.05) so a flat result can be stated rather than inferred from p > .05.

Sample: awakenings with usable features in all of E2, E3, E4. E1 (22 'case' samples only) is left out.
Runtime: CPU is enough; ~10-20 min for the permutations."""),
    code(BOOTSTRAP),
    code('''import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from d2i import data, decode, stats
dc = cfg["h2"]
der, tables, figs = env.path(cfg, "derived"), env.path(cfg, "results") / "tables", env.path(cfg, "results") / "figures"
long = pd.read_csv(der / "features_long.csv")
X, meta = decode.feature_matrices(long, dc["epochs"], dc["features"])
print({e: x.shape for e, x in X.items()}, "| participants:", meta.participant.nunique())'''),
    md("## 1. Report embeddings"),
    code('''texts = {m["sample_id"]: m["transcription"] for m in data.load_meta(cfg)}
emb_path = der / "report_embeddings.npz"
if emb_path.exists():
    z = np.load(emb_path, allow_pickle=True); E = dict(zip(z["sample_id"], z["emb"]))
else:
    E = decode.embed(texts, dc["text_model"])
    np.savez(emb_path, sample_id=np.array(list(E)), emb=np.array(list(E.values())))
Y = np.array([E[s] for s in meta.sample_id])
groups = meta.participant.values
print("embeddings", Y.shape)'''),
    md("## 2. Decoding per epoch, with the permutation null"),
    code('''rng = np.random.default_rng(cfg["seed"])
perms = [rng.permutation(len(Y)) for _ in range(dc["n_perm"])]
res, scores, nulls = [], {}, {}
for e in dc["epochs"]:
    scores[e] = decode.loso_rank(X[e], Y, groups, dc["alphas"])
    nulls[e] = np.array([decode.loso_rank(X[e], Y[p], groups, dc["alphas"]) for p in perms])  # (n_perm, n)
    res.append(decode.vs_null(scores[e], nulls[e], {"epoch": e}))
    print(res[-1])
np.savez_compressed(der / "h2_null_scores.npz", **{f"{e}_obs": scores[e] for e in scores}, **nulls)
dec = pd.DataFrame(res)
dec["p_holm"] = stats.holm(dec.p_perm)
dec.to_csv(tables / "h2_decoding_by_epoch.csv", index=False)
pd.DataFrame({"sample_id": meta.sample_id, "participant": groups, **scores}).to_csv(tables / "h2_rank_scores.csv", index=False)
print(dec.round(4).to_string(index=False))'''),
    md("## 3. E4 vs E2: difference test and equivalence test"),
    code('''cmp = decode.paired_epoch_test(scores["E4"], scores["E2"], groups, dc["n_perm"], rng, dc["tost_bound"])
pd.DataFrame([cmp]).to_csv(tables / "h2_e4_vs_e2.csv", index=False)
print(json.dumps(cmp, indent=1, default=float))'''),
    md("## 4. Post hoc: by awakening type (exploratory)"),
    code('''meta["awakening_type"] = np.where(meta.tag_class == "SO", "sleep onset (SO)", "other")
sub = pd.DataFrame([decode.vs_null(scores[e][m], nulls[e][:, m], {"awakening_type": t, "epoch": e})
                    for t in meta.awakening_type.unique() for e in dc["epochs"]
                    for m in [(meta.awakening_type == t).values]])
sub["p_fdr"] = stats.fdr_bh(sub.p_perm)
sub.to_csv(tables / "h2_posthoc_awakening_type.csv", index=False)
print(sub.round(4).to_string(index=False))'''),
    md("## 5. Figure"),
    code('''fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(dec))
ax.bar(x, dec.above_chance, color="#4a6fa5", width=.6, label="observed - empirical chance")
ax.scatter(x, dec.null_p95 - dec.null_mean, marker="_", s=600, color="#c0504d", label="null 95th percentile")
ax.axhline(0, color="k", lw=.8, ls=":")
ax.set(xticks=x, xticklabels=[f"{e}\\np_holm={p:.3f}" for e, p in zip(dec.epoch, dec.p_holm)],
       ylabel="rank accuracy above empirical chance",
       title="EEG -> report embedding, leave-one-participant-out")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(figs / "h2_decoding.pdf"); fig.savefig(figs / "h2_decoding.png", dpi=150)
env.write_manifest(cfg, "05_h2", [tables / f for f in ["h2_decoding_by_epoch.csv", "h2_rank_scores.csv",
                                                       "h2_e4_vs_e2.csv", "h2_posthoc_awakening_type.csv"]])'''),
]


def write(name, cells):
    nb = {"cells": [], "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}},
        "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(cells):
        cell = {"cell_type": kind, "id": f"{name[:2]}-{i:02d}", "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        nb["cells"].append(cell)
    (NB_DIR / name).write_text(json.dumps(nb, indent=1) + "\n")


if __name__ == "__main__":
    write("00_setup.ipynb", NB00)
    write("01_data.ipynb", NB01)
    write("02_preprocess.ipynb", NB02)
    write("03_features.ipynb", NB03)
    write("04_h1.ipynb", NB04)
    write("05_h2.ipynb", NB05)
    print("wrote", sorted(p.name for p in NB_DIR.glob("*.ipynb")))
