"""Per-epoch spectral features.

Power spectrum: Welch-style average of Hann-windowed 4-s segments (2-s step) that do not touch a
rejected sample. Band power is log10 of the integrated spectrum. The aperiodic exponent and offset
come from specparam (Donoghue et al. 2020), fitted 1-30 Hz with peaks modelled separately.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import periodogram


def clean_psd(x: np.ndarray, bad: np.ndarray, fs: float, win_s: float, step_s: float):
    """(freqs, psd (n_ch, n_freq), n_clean_segments) from segments with no rejected sample."""
    w, step = int(round(win_s * fs)), int(round(step_s * fs))
    freqs = np.fft.rfftfreq(w, 1 / fs)
    starts = [a for a in range(0, x.shape[1] - w + 1, step) if not bad[a:a + w].any()]
    if not starts:
        return freqs, None, 0
    ps = [periodogram(x[:, a:a + w], fs=fs, window="hann", detrend="constant", axis=-1)[1] for a in starts]
    return freqs, np.mean(ps, axis=0), len(starts)


def band_power(f: np.ndarray, psd: np.ndarray, bands: dict) -> dict:
    out = {}
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f < hi)
        out[f"logpow_{name}"] = np.log10(np.trapezoid(psd[:, m], f[m], axis=-1))
    return out


def fit_aperiodic(f: np.ndarray, spectra: np.ndarray, fcfg: dict) -> dict:
    """specparam on every spectrum (rows). Returns exponent, offset, r2, n_peaks arrays."""
    from specparam import SpectralGroupModel
    sp = dict(fcfg["specparam"])
    sp.pop("aperiodic_mode", None)  # 'fixed' is the specparam default (no knee)
    g = SpectralGroupModel(aperiodic_mode=fcfg["specparam"]["aperiodic_mode"], verbose=False, **sp)
    g.fit(f, spectra, list(fcfg["specparam_range"]))
    ap = g.results.get_params("aperiodic")
    r2 = np.asarray(g.results.get_metrics("gof_rsquared"), float)
    peaks = np.atleast_2d(g.results.get_params("periodic"))
    found = peaks[np.isfinite(peaks[:, 0]), -1].astype(int) if peaks.size else np.array([], int)
    n_peaks = np.bincount(found, minlength=len(spectra))
    ok = r2 >= fcfg["min_r2"]
    return {"ap_offset": np.where(ok, ap[:, 0], np.nan), "ap_exponent": np.where(ok, ap[:, 1], np.nan),
            "fit_r2": r2, "n_peaks": n_peaks}
