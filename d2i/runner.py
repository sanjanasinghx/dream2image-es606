"""Execute pipeline notebooks headless inside an already-set-up runtime.

From any Colab notebook whose bootstrap cell has run:
    from d2i import runner; runner.run(["02_preprocess.ipynb", "03_features.ipynb"])
One Drive mount and one install serve every notebook. The executed copies, outputs included,
land in <data root>/results/notebooks/ as the record of what ran.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from .env import REPO, data_root


def scrub(path) -> None:
    """Replace machine-specific paths in an executed notebook so it can be published."""
    from pathlib import Path
    p = Path(path)
    if p.exists():
        text = p.read_text()
        for local, label in [(str(data_root()), "<data root>"), (str(REPO), "<repo>"), (str(Path.home()), "~")]:
            text = text.replace(local, label)
        p.write_text(text)


def run(notebooks: list[str], timeout_s: int = 3600) -> None:
    out = data_root() / "results" / "notebooks"
    out.mkdir(parents=True, exist_ok=True)
    # the child kernel skips the Colab bootstrap (Drive is already mounted, repo already pulled)
    env = dict(os.environ, D2I_DATA=str(data_root()), D2I_HEADLESS="1")
    # One BLAS thread: the matrices here are small, and multi-threaded BLAS on them thrashes (~80x slower on
    # Colab's 2 vCPUs, every core pegged on a laptop). Single-threaded is both faster and cooler.
    env.update({k: "1" for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]})
    for nb in notebooks:
        t0 = time.time()
        # this interpreter's nbconvert and kernel, not whichever `jupyter` is first on PATH
        r = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                            str(REPO / "notebooks" / nb), "--output-dir", str(out),
                            f"--ExecutePreprocessor.timeout={timeout_s}"],
                           cwd=REPO / "notebooks", env=env, capture_output=True, text=True)
        scrub(out / nb)
        status = "OK" if r.returncode == 0 else "FAILED"
        print(f"{nb}: {status} in {time.time() - t0:.0f}s -> {out / nb}")
        if r.returncode != 0:
            print(r.stderr[-3000:])
            break
