"""Sleep-EDF Expanded, Sleep Cassette: features around scored sleep onset, with the Dream2Image pipeline.

Same filter, artifact rejection and spectral features as NB2-NB3 (d2i.preprocess, d2i.features), applied
to one posterior bipolar channel. Each 15-s epoch is filtered on its own, as in NB2, and rejected against
one robust scale per recording (from the filtered segment around onset).
"""
from __future__ import annotations

import numpy as np

from . import features, preprocess

STAGES = {"Sleep stage W": "W", "Sleep stage 1": "1", "Sleep stage 2": "2", "Sleep stage 3": "3",
          "Sleep stage 4": "4", "Sleep stage R": "R", "Sleep stage ?": "?", "Movement time": "M"}
SLEEP = {"1", "2", "3", "4", "R"}


def fetch(subjects, recording: int, path=None, retries: int = 4) -> tuple[list[tuple[int, str, str]], list[dict]]:
    """(subject, psg, hypnogram) for every subject that has this night, plus a record of the subjects
    that are absent from the study or could not be downloaded. PhysioNet times out now and then, so each
    subject is retried; a subject is only dropped after `retries` failed attempts, and that is logged."""
    import time
    from mne.datasets.sleep_physionet.age import fetch_data
    out, failed = [], []
    for s in subjects:
        for attempt in range(1, retries + 1):
            try:
                files = fetch_data([s], recording=[recording], path=path, on_missing="ignore", verbose=False)
                break
            except ValueError:  # subject or night not in the cassette study
                files, err = [], "not in study"
                break
            except Exception as e:  # network: wait and retry
                files, err = None, type(e).__name__
                time.sleep(10 * attempt)
        if not files:
            failed.append({"subject": s, "included": False,
                           "reason": "not in study" if files == [] else f"download failed ({err})"})
            continue
        out += [(s, str(psg), str(hyp)) for psg, hyp in files]
    return out, failed


def stage_labels(annotations, n_epochs: int, epoch_s: float) -> np.ndarray:
    """One stage label per 30-s epoch from the hypnogram annotations ('' where unscored)."""
    lab = np.full(n_epochs, "", dtype=object)
    for a in annotations:
        st = STAGES.get(a["description"])
        if st is None:
            continue
        i0 = int(round(a["onset"] / epoch_s))
        lab[i0:i0 + int(round(a["duration"] / epoch_s))] = st
    return lab


def onset_epoch(lab: np.ndarray, min_run: int) -> int | None:
    """First epoch scored 1 or 2 that starts >= min_run consecutive sleep epochs."""
    for i in range(len(lab) - min_run + 1):
        if lab[i] in {"1", "2"} and all(x in SLEEP for x in lab[i:i + min_run]):
            return i
    return None


def process(psg: str, hyp: str, cfg: dict) -> tuple[list[dict], np.ndarray | None, np.ndarray, dict]:
    """Epoch rows (no specparam yet), their spectra, the frequency axis, and a recording record."""
    import mne
    sc, pc, fc = cfg["sleepedf"], cfg["preprocess"], cfg["features"]
    raw = mne.io.read_raw_edf(psg, include=[sc["channel"]], preload=True, verbose=False)
    fs, ep_len = raw.info["sfreq"], cfg["epochs"]["epoch_len_s"]
    es = sc["stage_epoch_s"]
    lab = stage_labels(mne.read_annotations(hyp), int(raw.n_times / fs // es), es)
    rec = {"n_scored_epochs": int((lab != "").sum())}
    i0 = onset_epoch(lab, sc["onset_min_run"])
    if i0 is None:
        return [], None, np.array([]), {**rec, "included": False, "reason": "no sleep onset"}
    t0 = i0 * es
    rec["onset_s"] = t0
    # the wake control window must be scored wake in every 30-s epoch it touches
    w0 = t0 + sc["windows"]["wake"]
    touched = range(int(w0 // es), int(np.ceil((w0 + 3 * ep_len) / es)))
    if w0 < 0 or any(lab[k] != "W" for k in touched):
        return [], None, np.array([]), {**rec, "included": False, "reason": "control window not all wake"}
    a, b = (int(round((t0 + s) * fs)) for s in sc["segment_s"])
    if a < 0 or b > raw.n_times:
        return [], None, np.array([]), {**rec, "included": False, "reason": "segment outside recording"}
    x_all = raw.get_data()[:, :] * 1e6  # volts -> microvolts: numerics only, every measure is scale-free
    info = mne.create_info([sc["channel"]], fs, "eeg", verbose=False)
    seg = preprocess._filter(mne.io.RawArray(x_all[:, a:b], info, verbose=False), pc).get_data()
    scale = preprocess.robust_scale(seg)

    rows, spectra, f = [], [], np.array([])
    n_ep = int(round(ep_len * fs))
    for window, start in sc["windows"].items():
        for k in range(3):
            s0 = int(round((t0 + start + k * ep_len) * fs))
            x = preprocess._filter(mne.io.RawArray(x_all[:, s0:s0 + n_ep], info, verbose=False), pc).get_data()
            starts, w = preprocess.subwindow_starts(n_ep, fs, pc["reject_subwin_s"])
            bad_win = preprocess.subwindow_failures(x, scale, starts, w, pc).any(axis=1)
            mask = np.zeros(n_ep, bool)
            for st, bw in zip(starts, bad_win):
                if bw:
                    mask[st:st + w] = True
            f, psd, n_seg = features.clean_psd(x, mask, fs, fc["welch_win_s"], fc["welch_overlap_s"])
            ok = n_seg >= fc["min_clean_segments"]
            bp = features.band_power(f, psd, fc["bands"]) if ok else {}
            rows.append({"window": window, "epoch": f"S{k + 1}", "center_s": k * ep_len + ep_len / 2,
                         "reject_rate": float(mask.mean()), "n_clean_seg": n_seg, "ok": ok,
                         **{kk: v[0] for kk, v in bp.items()}})
            spectra.append(psd[0] if ok else np.full(len(f), np.nan))
    return rows, np.array(spectra), f, {**rec, "included": True, "reason": ""}
