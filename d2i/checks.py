"""Structural checks on the cached data (ported from the Oct 2026 Step 0).

Each function returns a DataFrame or dict; the notebook decides what to save.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- nesting
def nesting_test(total: np.ndarray, win: np.ndarray, n_expected: int, chk: dict) -> dict:
    """Compare eeg_Ns against the last N samples of eeg_total (float64)."""
    lt, lw = total.shape[1], win.shape[1]
    n = min(lt, lw)
    res = {"len_total": lt, "len_window": lw, "n_expected": n_expected,
           "nch_total": total.shape[0], "nch_window": win.shape[0]}
    if total.shape[0] != win.shape[0]:
        res["verdict"] = "channel-mismatch"
        return res
    x, y = total[:, -n:], win[:, -n:]
    res["exact_equal"] = bool(np.array_equal(x, y))
    res["allclose"] = bool(np.allclose(x, y, atol=chk["identical_atol"], rtol=0))
    live = np.where((x.std(1) > 0) & (y.std(1) > 0))[0]  # dead channels have no r
    rs = [np.corrcoef(x[c], y[c])[0, 1] for c in live]
    res["r_min"] = float(np.min(rs)) if rs else np.nan
    if res["allclose"]:
        v = "identical"
    elif res["r_min"] >= chk["affine_min_r"]:
        v = "affine-rescaled"
    else:
        v = "misaligned"
    if not (lw == n_expected and lt >= n_expected):
        res["verdict_on_overlap"] = v
        v = "length-mismatch"
    res["verdict"] = v
    if res.get("verdict_on_overlap", v) == "misaligned":
        res.update(lag_search(total, win))
    return res


def lag_search(total: np.ndarray, win: np.ndarray) -> dict:
    """Where does the window best match inside eeg_total (channel-mean normalised xcorr)?"""
    from scipy.signal import fftconvolve
    lw, lt = win.shape[1], total.shape[1]
    if lw > lt:
        return {"best_lag_start": None, "best_lag_r": None}
    acc = np.zeros(lt - lw + 1)
    for c in range(win.shape[0]):
        t = (win[c] - win[c].mean()) / (win[c].std() + 1e-12)
        s = total[c]
        num = fftconvolve(s, t[::-1], mode="valid")
        cs, cs2 = np.cumsum(np.r_[0, s]), np.cumsum(np.r_[0, s ** 2])
        m = (cs[lw:] - cs[:-lw]) / lw
        sd = np.sqrt(np.maximum((cs2[lw:] - cs2[:-lw]) / lw - m ** 2, 1e-12))
        acc += num / (lw * sd)
    acc /= win.shape[0]
    k = int(np.argmax(acc))
    return {"best_lag_start": k, "best_lag_end_offset": lt - (k + lw), "best_lag_r": float(acc[k])}


def nesting_table(metas: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"sample_id": m["sample_id"], "window_s": int(s), **r}
                         for m in metas for s, r in m["nesting"].items()])


# --------------------------------------------------------------------------- scaling
def scaling_table(cfg: dict, metas: list[dict], load) -> pd.DataFrame:
    """Per sample x column x channel mean/SD: is every eeg_* column z-scored on its own?"""
    ds, chk = cfg["dataset"], cfg["checks"]
    cols = [ds["total_col"], *ds["window_cols"].values()]
    rows = []
    for m in metas:
        for c in cols:
            if not m["shapes"].get(c):
                continue
            x = load(m["sample_id"], c)
            labels = ds["channels"] if x.shape[0] == len(ds["channels"]) else [f"ch{k}" for k in range(x.shape[0])]
            for ci, ch in enumerate(labels):
                rows.append({"sample_id": m["sample_id"], "source": c, "channel": ch,
                             "mean": float(x[ci].mean()), "sd": float(x[ci].std()),
                             "max_abs": float(np.abs(x[ci]).max())})
    df = pd.DataFrame(rows)
    df["is_z"] = (df["mean"].abs() < chk["zscore_mean_tol"]) & ((df["sd"] - 1).abs() < chk["zscore_sd_tol"])
    return df


# --------------------------------------------------------------------------- participants / exclusions
def participants_table(metas: list[dict], parse_title) -> pd.DataFrame:
    df = pd.DataFrame([{"sample_id": m["sample_id"], "title": m["title"], **parse_title(m["title"])}
                       for m in metas])
    df["participant"] = df["site"] + "_" + df["subject_code"].map(lambda c: f"{c:03d}")
    return df


def exclusions_table(cfg: dict, metas: list[dict], nest: pd.DataFrame, parts: pd.DataFrame) -> pd.DataFrame:
    tc, need = cfg["dataset"]["total_col"], cfg["exclusions"]["drop_channel_count_not"]
    mis = set(nest.loc[nest.verdict == "misaligned", "sample_id"])
    rows = []
    for m in metas:
        reasons = []
        if cfg["exclusions"]["drop_nesting_misaligned"] and m["sample_id"] in mis:
            reasons.append("nesting_misaligned")
        if m["shapes"][tc]["n_channels"] != need:
            reasons.append(f"n_channels={m['shapes'][tc]['n_channels']}")
        rows.append({"sample_id": m["sample_id"], "retained": not reasons, "reasons": ";".join(reasons)})
    return pd.DataFrame(rows).merge(parts[["sample_id", "title", "participant", "site", "tag_class"]],
                                    on="sample_id")


def channel_availability(cfg: dict, scaling: pd.DataFrame, excl: pd.DataFrame) -> pd.DataFrame:
    """Primary channel set = live (SD > floor) in >= primary_channel_min_frac of retained samples."""
    keep = set(excl.loc[excl.retained, "sample_id"])
    chans, floor = cfg["dataset"]["channels"], cfg["dataset"]["dead_sd_floor"]
    s = scaling[(scaling.source == cfg["dataset"]["total_col"]) & scaling.sample_id.isin(keep)]
    av = (s.assign(live=s.sd > floor).groupby("channel")["live"]
          .agg(n_live="sum", n="size").reindex(chans).reset_index())
    av["frac_live"] = av.n_live / av.n
    av["primary"] = av.frac_live >= cfg["checks"]["primary_channel_min_frac"]
    return av


# --------------------------------------------------------------------------- channel order
def channel_order_check(cfg: dict, metas: list[dict], load, rng) -> tuple[dict, pd.DataFrame]:
    """Is the undocumented channel order spatially plausible? Mean inter-channel correlation should
    fall with scalp distance; compared against random relabellings of the channels."""
    import mne
    from scipy.stats import spearmanr
    chans, tc = cfg["dataset"]["channels"], cfg["dataset"]["total_col"]
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    xyz = np.array([pos[c] for c in chans])
    D = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    Zs = []
    for m in metas:
        if not m["shapes"][tc] or m["shapes"][tc]["n_channels"] != len(chans):
            continue
        x = load(m["sample_id"], tc)
        live = x.std(1) > 0
        Z = np.full((len(chans), len(chans)), np.nan)
        Z[np.ix_(live, live)] = np.arctanh(np.clip(np.corrcoef(x[live]), -0.999999, 0.999999))
        Zs.append(Z)
    C = np.tanh(np.nanmean(Zs, 0))
    iu = np.triu_indices(len(chans), 1)
    rho = spearmanr(C[iu], -D[iu]).statistic
    null = np.array([spearmanr(C[iu], -D[np.ix_(p, p)][iu]).statistic
                     for p in (rng.permutation(len(chans)) for _ in range(cfg["checks"]["n_perm_channel_order"]))])
    res = {"spearman_corr_vs_neg_distance": float(rho),
           "perm_p": float((np.sum(null >= rho) + 1) / (len(null) + 1)),
           "null_p95": float(np.percentile(null, 95)), "n_samples_used": len(Zs)}
    return res, pd.DataFrame(C, index=chans, columns=chans)


# --------------------------------------------------------------------------- site spectra
def site_spectra(cfg: dict, metas: list[dict], parts: pd.DataFrame, load) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Channel-mean Welch spectrum of eeg_total per sample, and the 45-100 Hz power fraction.
    The two sites should separate completely above 45 Hz (different hardware filters)."""
    from scipy.signal import welch
    fs, tc = cfg["dataset"]["sfreq"], cfg["dataset"]["total_col"]
    nper = int(cfg["checks"]["psd_win_s"] * fs)
    site = parts.set_index("sample_id")["site"]
    rows, spectra = [], []
    for m in metas:
        x = load(m["sample_id"], tc)
        x = x[x.std(1) > 0]
        f, P = welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1)
        band, hf = (f >= 1) & (f <= 45), (f > 45) & (f <= 100)
        rows.append({"sample_id": m["sample_id"], "site": site[m["sample_id"]],
                     "log10_frac_45_100": float(np.log10(np.median(P[:, hf].sum(1) / P[:, band].sum(1))))})
        spectra.append(np.log10(P.mean(0)))
    return pd.DataFrame(rows), f, np.array(spectra)


# --------------------------------------------------------------------------- per-epoch scale
def scaling_outliers(cfg: dict, excl: pd.DataFrame, load) -> pd.DataFrame:
    """Median channel SD inside each epoch, in eeg_total z-units. A primary epoch far below 1 means
    the sample's z-scale was set by an artifact elsewhere; such samples are kept but flagged."""
    fs, floor, L = cfg["dataset"]["sfreq"], cfg["dataset"]["dead_sd_floor"], cfg["epochs"]["epoch_len_s"]
    eps = {k: cfg["epochs"][k] for k in ["E1", "E2", "E3", "E4"]}
    rows = []
    for _, r in excl.iterrows():
        x = load(r.sample_id, cfg["dataset"]["total_col"])
        x = x[x.std(1) > floor]
        n = x.shape[1]
        row = {"sample_id": r.sample_id, "retained": r.retained, "site": r.site,
               "len_s": n / fs, "max_abs_z": float(np.abs(x).max())}
        for k, (_, b) in eps.items():
            i0, i1 = n + int((b - L) * fs), n + int(b * fs)
            row[f"{k}_sd_median"] = float(np.median(x[:, i0:i1].std(1))) if i0 >= 0 else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    prim = [f"{k}_sd_median" for k in cfg["epochs"]["primary_epochs"]]
    df["scaling_flag"] = (df[prim] < cfg["checks"]["scaling_flag_sd"]).any(axis=1)
    return df
