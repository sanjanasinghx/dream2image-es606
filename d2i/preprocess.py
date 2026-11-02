"""Epoching and cleaning (ported from the Oct 2026 Steps 1-2; same algorithm, npz output).

Every epoch is the last `epoch_len_s` of its bin, sliced from the one eeg_120s array, so all four
epochs of a sample share a single z-scale. Units stay z-units throughout (never volts).
"""
from __future__ import annotations

import warnings

import numpy as np

MAD_K = 1.4826
EPOCHS = ["E1", "E2", "E3", "E4"]


def epoch_bounds(n_total: int, fs: float, bins: dict, epoch_len_s: float) -> dict:
    """Sample indices of each epoch; None if the recording does not reach back far enough."""
    n_ep = int(round(epoch_len_s * fs))
    out = {}
    for name, (_, b1) in bins.items():
        stop = n_total + int(round(b1 * fs))
        out[name] = (stop - n_ep, stop) if stop - n_ep >= 0 else None
    return out


def robust_scale(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=1, keepdims=True)
    return MAD_K * np.median(np.abs(x - med), axis=1)


def subwindow_starts(n: int, fs: float, win_s: float) -> tuple[np.ndarray, int]:
    """Evenly spaced win_s windows covering all n samples (last one ends at n)."""
    w = int(round(win_s * fs))
    k = int(np.ceil(n / w))
    return np.round(np.linspace(0, n - w, k)).astype(int), w


def subwindow_failures(x, scale, starts, w, pc) -> np.ndarray:
    """(n_win, n_ch) bool: peak-to-peak, flat or gradient failure, in robust units of `scale`."""
    s = np.where(scale > 0, scale, np.nan)[None, :]
    seg = np.stack([x[:, a:a + w] for a in starts])
    ptp = np.nan_to_num(np.ptp(seg, axis=2) / s) > pc["ptp_threshold"]
    flat = np.nan_to_num(seg.std(axis=2) / s, nan=0.0) < pc["flat_threshold"]
    grad = np.nan_to_num(np.abs(np.diff(seg, axis=2)).max(axis=2) / s) > pc["gradient_threshold"]
    return ptp | flat | grad


def neighbor_r(x: np.ndarray, pos: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean Pearson r of each channel with its k nearest (3-D) neighbours."""
    D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(x)
    return np.array([np.nanmean(C[i, np.argsort(D[i])[:k]]) for i in range(len(x))])


def _raw(x, chans, fs, montage):
    import mne
    # copy: MNE filters/re-references in place and would otherwise alter the caller's array
    raw = mne.io.RawArray(np.array(x, np.float64), mne.create_info(chans, fs, "eeg", verbose=False),
                          verbose=False)
    raw.set_montage(montage, verbose=False)
    return raw


def _filter(raw, pc):
    lo, hi = pc["band"]
    return raw.filter(lo, hi, method="fir", phase="zero", fir_design="firwin",
                      pad=pc["filter_pad"], verbose=False)


def _interpolate(raw, bads):
    if bads:
        raw.info["bads"] = list(bads)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)
    return raw


def _reference(raw, on: bool):
    if on:
        raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def process_sample(cfg: dict, sid: str, total: np.ndarray, channels: list[str]) -> tuple[dict, list, list]:
    """Clean every available epoch of one sample.

    Returns (arrays for npz, one record per epoch, one record per channel).
    Arrays: <E> (n_ch, n) cleaned signal, <E>_bad (n,) bool rejected-sample mask.
    """
    import mne
    mne.set_log_level("ERROR")
    ds, pc, ep = cfg["dataset"], cfg["preprocess"], cfg["epochs"]
    fs, floor = ds["sfreq"], ds["dead_sd_floor"]
    keep = [ds["channels"].index(c) for c in channels]
    total = total[keep]
    montage = mne.channels.make_standard_montage("standard_1020")
    pos = np.array([montage.get_positions()["ch_pos"][c] for c in channels])

    # sample level: bad channels on the filtered full recording, then the two robust scales
    full = _filter(_raw(total, channels, fs, montage), pc)
    nbr = neighbor_r(full.get_data(), pos)
    dead = [c for c, sd in zip(channels, total.std(1)) if sd <= floor]
    low_r = [c for c, r in zip(channels, nbr) if r < pc["bad_neighbor_r_min"] and c not in dead]
    sample_bads = sorted(set(dead) | set(low_r))
    _interpolate(full, sample_bads)
    scale_pre = robust_scale(full.get_data())
    _reference(full, pc["average_reference"])
    scale_post = robust_scale(full.get_data())

    arrays, recs = {}, []
    bins = {k: ep[k] for k in EPOCHS}
    for name, b in epoch_bounds(total.shape[1], fs, bins, ep["epoch_len_s"]).items():
        if b is None:
            continue
        raw = _filter(_raw(total[:, b[0]:b[1]], channels, fs, montage), pc)
        n = raw.n_times
        starts, w = subwindow_starts(n, fs, pc["reject_subwin_s"])
        frac = subwindow_failures(raw.get_data(), scale_pre, starts, w, pc).mean(axis=0)
        ep_bads = [c for c, fr in zip(channels, frac) if fr > pc["channel_epoch_bad_frac"] and c not in sample_bads]
        all_bads = sorted(set(sample_bads) | set(ep_bads))
        _interpolate(raw, all_bads)
        _reference(raw, pc["average_reference"])
        x = raw.get_data()
        bad_win = subwindow_failures(x, scale_post, starts, w, pc).any(axis=1)
        mask = np.zeros(n, bool)
        for a, bw in zip(starts, bad_win):
            if bw:
                mask[a:a + w] = True
        arrays[name], arrays[f"{name}_bad"] = x.astype(np.float32), mask
        recs.append({"sample_id": sid, "epoch": name, "center_s": bins[name][1] - ep["epoch_len_s"] / 2,
                     "reject_rate": float(mask.mean()), "n_bad_subwin": int(bad_win.sum()),
                     "n_interp": len(all_bads), "epoch_bad_channels": ";".join(ep_bads),
                     "usable": len(all_bads) <= pc["max_interp_frac"] * len(channels)})
    chan_recs = [{"sample_id": sid, "channel": c, "neighbor_r": float(nbr[i]),
                  "scale_pre_ref": float(scale_pre[i]), "scale_post_ref": float(scale_post[i]),
                  "bad_reason": "dead" if c in dead else ("low_neighbor_r" if c in low_r else "")}
                 for i, c in enumerate(channels)]
    return arrays, recs, chan_recs
