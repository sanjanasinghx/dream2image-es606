"""H2: decode report content from per-epoch spectral features, leave-one-participant-out."""
from __future__ import annotations

import numpy as np
import pandas as pd


def feature_matrices(long: pd.DataFrame, epochs: list[str], feats: list[str]) -> tuple[dict, pd.DataFrame]:
    """One (n_awakenings, n_channels * n_features) matrix per epoch, over the awakenings that have
    complete, finite features in every listed epoch (same rows in every matrix)."""
    d = long[long.ok & long.epoch.isin(epochs)]
    wide = d.pivot_table(index=["sample_id", "epoch"], columns="channel", values=feats)
    wide.columns = [f"{f}@{c}" for f, c in wide.columns]
    wide = wide.dropna()
    complete = wide.reset_index().groupby("sample_id").epoch.nunique()
    keep = sorted(complete[complete == len(epochs)].index)
    X = {e: wide.xs(e, level="epoch").loc[keep].to_numpy() for e in epochs}
    meta = (long.drop_duplicates("sample_id").set_index("sample_id")
            .loc[keep, ["participant", "site", "tag_class"]].reset_index())
    return X, meta


def embed(texts: dict, model_name: str) -> dict:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    ids = list(texts)
    vecs = model.encode([texts[i] for i in ids], batch_size=16, show_progress_bar=False)
    return dict(zip(ids, vecs))


def loso_rank(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, alphas) -> np.ndarray:
    """Percentile rank of each held-out awakening's true report among all reports (chance 0.5).
    Features are standardised and the embeddings centred with training-fold statistics only."""
    from sklearn.linear_model import RidgeCV
    scores = np.empty(len(Y))
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        ym = Y[tr].mean(0)
        pred = RidgeCV(alphas=alphas).fit((X[tr] - mu) / sd, Y[tr] - ym).predict((X[te] - mu) / sd)
        C = Y - ym
        sim = (pred / np.linalg.norm(pred, axis=1, keepdims=True)) @ (C / np.linalg.norm(C, axis=1, keepdims=True)).T
        for row, i in zip(sim, np.flatnonzero(te)):
            others = np.delete(row, i)
            scores[i] = np.mean(others < row[i])
    return scores


def vs_null(obs: np.ndarray, null: np.ndarray, label: dict) -> dict:
    """Mean rank accuracy of a set of awakenings against the same awakenings under the permutation null."""
    dist = null.mean(1)
    return {**label, "rank_acc": obs.mean(), "null_mean": dist.mean(), "above_chance": obs.mean() - dist.mean(),
            "null_p95": np.percentile(dist, 95), "p_perm": (np.sum(dist >= obs.mean()) + 1) / (len(dist) + 1),
            "n": len(obs)}


def paired_epoch_test(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n_perm: int, rng, bound: float) -> dict:
    """Mean of per-participant (a - b): sign-flip permutation p (two-sided) and TOST equivalence p
    against +/- bound (Lakens 2017), both with participants as the unit."""
    from scipy import stats as st
    diff = pd.Series(a - b).groupby(groups).mean().to_numpy()
    obs = diff.mean()
    flips = rng.choice([-1, 1], size=(n_perm, len(diff)))
    null = (flips * diff).mean(1)
    lo, hi = st.t.interval(0.95, len(diff) - 1, loc=obs, scale=st.sem(diff))
    p_lower = st.ttest_1samp(diff, -bound, alternative="greater").pvalue
    p_upper = st.ttest_1samp(diff, bound, alternative="less").pvalue
    return {"mean_diff": obs, "ci_low": lo, "ci_high": hi,
            "p_signflip": (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1),
            "tost_bound": bound, "p_tost": max(p_lower, p_upper), "n_participants": len(diff)}
