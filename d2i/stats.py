"""Linear mixed models for the within-participant change toward awakening (H1, RQ3)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def fit_trend(df: pd.DataFrame, y: str, covariates: list[str], interaction: str | None = None) -> dict:
    """y ~ time30 [+ time30:interaction + interaction] + covariates, random intercepts for participant
    and for awakening (sample) nested in participant. time30 = epoch centre / 30 s, so the slope is the
    change per 30 s closer to awakening. Returns the time (or interaction) term with a 95% CI."""
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[y, *covariates] + ([interaction] if interaction else [])).copy()
    rhs = ["time30", *covariates]
    if interaction:
        rhs += [interaction, f"time30:{interaction}"]
    model = smf.mixedlm(f"{y} ~ {' + '.join(rhs)}", d, groups="participant", re_formula="1",
                        vc_formula={"awakening": "0 + C(sample_id)"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(reml=True, method=["lbfgs", "powell"])
    term = f"time30:{interaction}" if interaction else "time30"
    lo, hi = fit.conf_int().loc[term]
    sd = d[y].std()
    return {"y": y, "term": term, "estimate": fit.params[term], "ci_low": lo, "ci_high": hi,
            "p": fit.pvalues[term], "std_estimate": fit.params[term] / sd,
            "n_obs": len(d), "n_awakenings": d.sample_id.nunique(), "n_participants": d.participant.nunique(),
            "covariates": "+".join(covariates) or "none", "converged": bool(fit.converged)}


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def fdr_bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out
