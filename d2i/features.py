"""Per-epoch spectral features.

Power spectrum: Welch-style average of Hann-windowed 2-s segments (1-s step) that do not touch a
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


def irasa(x: np.ndarray, bad: np.ndarray, fs: float, win_s: float, step_s: float, hset, pad_s: float):
    """Fractal (aperiodic) spectrum by IRASA (Wen & Liu 2016), from segments with no rejected sample.

    For each resampling factor h the epoch is resampled by h and by 1/h and the result is read at the
    relabelled rates h*fs and fs/h, with the same segment length in samples as the plain spectrum: bin k
    then holds power at h*f_k and f_k/h. Their geometric mean cancels oscillatory peaks and keeps a power
    law; the median over h is the fractal spectrum on the plain frequency grid (as in YASA's irasa).
    Rejected samples are padded by pad_s (resampling filters spread an artifact) and mapped to each
    resampled signal. Returns (freqs, fractal psd (n_ch, n_freq) or None, fewest clean segments over h)."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    w, step = int(round(win_s * fs)), int(round(step_s * fs))
    pad = int(round(pad_s * fs))
    badp = np.convolve(bad.astype(float), np.ones(2 * pad + 1), mode="same") > 0 if pad else bad.astype(bool)

    def segs_psd(y, b, rate):
        starts = [a for a in range(0, y.shape[1] - w + 1, step) if not b[a:a + w].any()]
        if not starts:
            return None, 0
        ps = [periodogram(y[:, a:a + w], fs=rate, window="hann", detrend="constant", axis=-1)[1] for a in starts]
        return np.mean(ps, axis=0), len(starts)

    freqs = np.fft.rfftfreq(w, 1 / fs)
    out, n_min = [], np.inf
    for h in hset:
        up, down = Fraction(str(round(h, 4))).as_integer_ratio()
        geo = []
        for u, d, rate in [(up, down, fs * h), (down, up, fs / h)]:
            y = resample_poly(x, u, d, axis=-1)
            idx = np.minimum((np.arange(y.shape[1]) * d) // u, len(badp) - 1)
            p, n = segs_psd(y, badp[idx], rate)
            if p is None:
                return freqs, None, 0
            geo.append(p)
            n_min = min(n_min, n)
        out.append(np.sqrt(geo[0] * geo[1]))
    return freqs, np.median(out, axis=0), int(n_min)


def fit_powerlaw(f: np.ndarray, psd: np.ndarray, frange) -> dict:
    """Least-squares line in log10-log10 over frange: exponent = -slope, offset = intercept, and R^2."""
    m = (f >= frange[0]) & (f <= frange[1])
    lx, ly = np.log10(f[m]), np.log10(psd[:, m])
    A = np.c_[np.ones_like(lx), lx]
    coef, *_ = np.linalg.lstsq(A, ly.T, rcond=None)
    resid = ly - (A @ coef).T
    r2 = 1 - (resid ** 2).sum(1) / ((ly - ly.mean(1, keepdims=True)) ** 2).sum(1)
    return {"ap_offset": coef[0], "ap_exponent": -coef[1], "fit_r2": r2}
