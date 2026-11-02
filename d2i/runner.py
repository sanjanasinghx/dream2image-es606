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


def run(notebooks: list[str], timeout_s: int = 3600) -> None:
    out = data_root() / "results" / "notebooks"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, D2I_DATA=str(data_root()))  # the child kernel is not Colab: point it at Drive
    for nb in notebooks:
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                            str(REPO / "notebooks" / nb), "--output-dir", str(out),
                            f"--ExecutePreprocessor.timeout={timeout_s}"],
                           cwd=REPO / "notebooks", env=env, capture_output=True, text=True)
        status = "OK" if r.returncode == 0 else "FAILED"
        print(f"{nb}: {status} in {time.time() - t0:.0f}s -> {out / nb}")
        if r.returncode != 0:
            print(r.stderr[-3000:])
            break
