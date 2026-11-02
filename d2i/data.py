"""Download Dream2Image once, cache each row as float32 arrays + JSON metadata.

The nesting test (are eeg_15s/30s/60s the tail of eeg_120s?) runs in float64 while the row is
in memory, because the float32 cache cannot support an exact-identity check.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from .checks import nesting_test
from .env import hf_token, path


def download(cfg: dict) -> list[Path]:
    """Fetch only the parquet shards (images included in the files, never decoded)."""
    from huggingface_hub import snapshot_download
    local = snapshot_download(cfg["dataset"]["hf_id"], repo_type="dataset",
                              allow_patterns=[cfg["dataset"]["parquet_glob"], "README.md"],
                              local_dir=path(cfg, "hf"), token=hf_token())
    shards = sorted(Path(local).glob(cfg["dataset"]["parquet_glob"]))
    if not shards:
        raise FileNotFoundError("no parquet shards downloaded; is the gated access accepted?")
    return shards


def arrow_2d(cell) -> np.ndarray | None:
    """One list<list<double>> cell -> float64 (n_outer, n_inner)."""
    if not cell.is_valid:
        return None
    rows = cell.values
    lens = rows.value_lengths().to_numpy()
    if len(set(lens.tolist())) != 1:
        raise ValueError(f"ragged inner lengths {set(lens.tolist())}")
    vals = rows.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
    return vals.reshape(len(lens), int(lens[0]))


def orient(x: np.ndarray) -> tuple[np.ndarray, str]:
    """Channels = the shorter axis (channel count is not assumed to be 17)."""
    if x.shape[0] <= x.shape[1]:
        return x, "channels_x_time"
    return x.T, "time_x_channels"


def cache_rows(cfg: dict, shards: list[Path], force: bool = False, log=print) -> int:
    """Stream shards one row at a time into raw/s###.{npz,json}. Resumable after a disconnect."""
    import pyarrow.parquet as pq

    ds = cfg["dataset"]
    raw, fs = path(cfg, "raw"), ds["sfreq"]
    eeg_cols = [ds["total_col"], *ds["window_cols"].values()]
    cols = [ds["name_col"], *ds["text_cols"], *eeg_cols]
    i, t0 = -1, time.time()
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=1, columns=cols):
            i += 1
            sid = f"s{i:03d}"
            jpath, npath = raw / f"{sid}.json", raw / f"{sid}.npz"
            if not force and jpath.exists() and npath.exists():
                continue
            meta = {"sample_id": sid, "row_index": i, "shard": shard.name,
                    "title": batch.column(ds["name_col"])[0].as_py(),
                    **{c: batch.column(c)[0].as_py() for c in ds["text_cols"]}}
            arrays, shapes = {}, {}
            for c in eeg_cols:
                x = arrow_2d(batch.column(c)[0])
                if x is None:
                    shapes[c] = None
                    continue
                raw_shape = list(x.shape)
                x, orientation = orient(x)
                shapes[c] = {"raw_shape": raw_shape, "orientation": orientation,
                             "n_channels": int(x.shape[0]), "n_samples": int(x.shape[1]),
                             "seconds": x.shape[1] / fs, "finite_all": bool(np.isfinite(x).all())}
                arrays[c] = x
            meta["shapes"] = shapes
            meta["nesting"] = {
                str(secs): nesting_test(arrays[ds["total_col"]], arrays[wc],
                                        int(round(int(secs) * fs)), cfg["checks"])
                for secs, wc in ds["window_cols"].items()
                if ds["total_col"] in arrays and wc in arrays}
            np.savez(npath, **{c: a.astype(np.float32) for c, a in arrays.items()})
            jpath.write_text(json.dumps(meta, indent=2, default=float))
            if (i + 1) % 10 == 0:
                log(f"[{i + 1}] {sid} {meta['title']!r}  {time.time() - t0:.0f}s")
    n = i + 1
    log(f"cached {n} rows in {time.time() - t0:.0f}s")
    return n


def load_meta(cfg: dict) -> list[dict]:
    raw = path(cfg, "raw")
    return [json.loads(f.read_text()) for f in sorted(raw.glob("s[0-9][0-9][0-9].json"))]


def load_eeg(cfg: dict, sample_id: str, col: str | None = None) -> np.ndarray:
    col = col or cfg["dataset"]["total_col"]
    return np.load(path(cfg, "raw") / f"{sample_id}.npz")[col].astype(np.float64)


# --------------------------------------------------------------------------- titles
TITLE_CASE = re.compile(r"^case(\d+)_sub(\d+)$", re.I)
TITLE_SUBJECT = re.compile(r"^subject(\d+)_([A-Za-z]+?)(\d*)(?:\.edf)?(ok)?$", re.I)


def parse_title(title: str) -> dict:
    """Title -> site / subject code / awakening tag. Unparseable titles raise; nothing is guessed.

    Site = naming family. 'case' rows come from one source study, 'subject' rows from the other;
    subject codes overlap across families (e.g. 107), so participant = <site>_<code>.
    """
    m = TITLE_CASE.fullmatch(title)
    if m:
        return {"site": "case", "subject_code": int(m.group(2)), "case_no": int(m.group(1)),
                "tag": None, "tag_class": None}
    m = TITLE_SUBJECT.fullmatch(title)
    if m:
        return {"site": "subject", "subject_code": int(m.group(1)), "case_no": None,
                "tag": f"{m.group(2)}{m.group(3)}", "tag_class": m.group(2)}
    raise ValueError(f"unparseable title {title!r}")
