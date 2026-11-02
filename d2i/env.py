"""Runtime environment: Colab vs local, data root, config, seeding, manifests."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "config.yaml"
DRIVE_ROOT = Path("/content/drive/MyDrive/ES606_Dream2Image")


def in_colab() -> bool:
    return "google.colab" in sys.modules


def data_root() -> Path:
    """Drive folder on Colab; $D2I_DATA (or ./_data) locally."""
    if in_colab():
        root = DRIVE_ROOT
    else:
        root = Path(os.environ.get("D2I_DATA", REPO / "_data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def path(cfg: dict, key: str) -> Path:
    d = data_root() / cfg["paths"][key]
    d.mkdir(parents=True, exist_ok=True)
    return d


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


def hf_token() -> str | None:
    """HF token from Colab Secrets (key HF_TOKEN) or the environment; None -> cached login."""
    if in_colab():
        try:
            from google.colab import userdata
            return userdata.get("HF_TOKEN")
        except Exception:
            return None
    return os.environ.get("HF_TOKEN")


def _sha(p: Path) -> str | None:
    p = Path(p)
    if not p.is_file() or p.stat().st_size > 50_000_000:
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def write_manifest(cfg: dict, step: str, outputs, notes: dict | None = None) -> Path:
    """Record what a notebook produced, with which config, on which runtime."""
    out = {
        "step": step,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "runtime": "colab" if in_colab() else "local",
        "python": platform.python_version(),
        "seed": cfg["seed"],
        "config_sha": _sha(CONFIG_PATH),
        "outputs": [{"path": str(Path(o).relative_to(data_root())), "sha256_16": _sha(o)}
                    for o in outputs],
        "notes": notes or {},
    }
    m = path(cfg, "results") / "logs" / f"manifest_{step}.json"
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps(out, indent=2, default=str))
    return m
