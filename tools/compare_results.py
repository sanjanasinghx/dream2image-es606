"""Compare the tables of a rerun against the committed results/tables.

    python tools/compare_results.py <rerun data root>/results/tables [out.csv]

Prints, per table, the largest absolute difference over numeric columns and whether the
text columns agree (missing values count as equal).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

committed = Path(__file__).resolve().parents[1] / "results" / "tables"
rerun = Path(sys.argv[1])
rows = []
for f in sorted(committed.glob("*.csv")):
    g = rerun / f.name
    if not g.exists():
        rows.append({"table": f.name, "status": "missing in rerun"})
        continue
    x, y = pd.read_csv(f), pd.read_csv(g)
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        rows.append({"table": f.name, "status": f"shape {x.shape} vs {y.shape}"})
        continue
    num = x.select_dtypes("number").columns
    txt = [c for c in x.columns if c not in num]
    d = np.abs(x[num].to_numpy(float) - y[num].to_numpy(float))
    max_diff = float(np.nanmax(d)) if d.size and not np.isnan(d).all() else 0.0
    same_nan = (x[num].isna().to_numpy() == y[num].isna().to_numpy()).all()
    same_txt = all(x[c].fillna("").astype(str).equals(y[c].fillna("").astype(str)) for c in txt)
    rows.append({"table": f.name, "rows": len(x), "max_abs_diff": max_diff,
                 "nan_pattern_same": bool(same_nan), "text_same": bool(same_txt), "status": "compared"})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
if len(sys.argv) > 2:
    out.to_csv(sys.argv[2], index=False)
