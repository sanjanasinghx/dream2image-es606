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
if "google.colab" in sys.modules:
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{REPO}/requirements-colab.txt"], check=True)
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
    print("wrote", sorted(p.name for p in NB_DIR.glob("*.ipynb")))
