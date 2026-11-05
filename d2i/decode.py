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


def make_folds(X: np.ndarray, groups: np.ndarray, fit_transform=None) -> list[dict]:
    """Leave-one-participant-out folds with the SVD of the standardised training features.
    X is fixed across permutations (only the reports are shuffled), so this is computed once.
    fit_transform(train_mask) -> feature matrix for all rows, learned on the training rows only
    (e.g. the Riemannian tangent space); it replaces X fold by fold."""
    folds = []
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        if fit_transform is not None:
            X = fit_transform(tr)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        U, s, Vt = np.linalg.svd((X[tr] - mu) / sd, full_matrices=False)
        folds.append({"te": np.flatnonzero(te), "tr": tr, "U": U, "s": s,
                      "XteV": ((X[te] - mu) / sd) @ Vt.T})
    return folds


def loso_rank(folds: list[dict], Y: np.ndarray, alphas) -> np.ndarray:
    """Percentile rank of each held-out awakening's true report among all reports, by cosine similarity
    to the ridge prediction. Ridge alpha per fold by generalised cross-validation on the training fold
    (efficient leave-one-out, alpha shared across embedding dimensions); features standardised and
    embeddings centred with training-fold statistics only."""
    scores = np.empty(len(Y))
    for f in folds:
        ym = Y[f["tr"]].mean(0)
        Yc = Y[f["tr"]] - ym
        U = f["U"]
        UtY = U.T @ Yc
        # Residual for a given alpha = R + U diag(1 - shrink) UtY, with R = Yc - U UtY fixed across alphas.
        # Its row norms need only small (n x k) and (k x k) products, not a 768-wide product per alpha.
        R = Yc - U @ UtY
        rr, M, G = (R ** 2).sum(1), R @ UtY.T, UtY @ UtY.T
        s2, best = f["s"] ** 2, None
        for a in alphas:
            shrink = s2 / (s2 + a)
            UD = U * (1 - shrink)
            res2 = rr + 2 * (UD * M).sum(1) + ((UD @ G) * UD).sum(1)
            err = np.sum(res2 / (1 - (U ** 2) @ shrink) ** 2) / Yc.size
            if best is None or err < best[0]:
                best = (err, a)
        a = best[1]
        pred = f["XteV"] @ ((f["s"] / (s2 + a))[:, None] * UtY)
        C = Y - ym
        sim = (pred / np.linalg.norm(pred, axis=1, keepdims=True)) @ (C / np.linalg.norm(C, axis=1, keepdims=True)).T
        true = sim[np.arange(len(f["te"])), f["te"]]
        scores[f["te"]] = (sim < true[:, None]).sum(1) / (len(Y) - 1)  # the true report is never < itself
    return scores


def vs_null(obs: np.ndarray, null: np.ndarray, label: dict) -> dict:
    """Mean rank accuracy of a set of awakenings against the same awakenings under the permutation null."""
    dist = null.mean(1)
    return {**label, "rank_acc": obs.mean(), "null_mean": dist.mean(), "above_chance": obs.mean() - dist.mean(),
            "null_p95": np.percentile(dist, 95), "p_perm": (np.sum(dist >= obs.mean()) + 1) / (len(dist) + 1),
            "n": len(obs)}


def paired_epoch_test(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n_perm: int, rng, bound: float) -> dict:
    """Mean of per-participant (a - b): sign-flip permutation p (two-sided) and TOST equivalence p
    against +/- bound (Lakens 2017), both with participants as the unit. The CI is 90%, the interval
    that matches TOST at alpha = .05."""
    from scipy import stats as st
    diff = pd.Series(a - b).groupby(groups).mean().to_numpy()
    obs = diff.mean()
    flips = rng.choice([-1, 1], size=(n_perm, len(diff)))
    null = (flips * diff).mean(1)
    lo, hi = st.t.interval(0.90, len(diff) - 1, loc=obs, scale=st.sem(diff))
    p_lower = st.ttest_1samp(diff, -bound, alternative="greater").pvalue
    p_upper = st.ttest_1samp(diff, bound, alternative="less").pvalue
    return {"mean_diff": obs, "ci_low": lo, "ci_high": hi,
            "p_signflip": (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1),
            "tost_bound": bound, "p_tost": max(p_lower, p_upper), "n_participants": len(diff)}


def tangent_space(covs: np.ndarray):
    """Fold-wise Riemannian tangent-space features (Barachant et al. 2012): the reference point
    (Riemannian mean) is estimated on the training rows only, then every covariance is projected."""
    from pyriemann.tangentspace import TangentSpace

    def fit_transform(tr: np.ndarray) -> np.ndarray:
        return TangentSpace(metric="riemann").fit(covs[tr]).transform(covs)
    return fit_transform


def clean_covariance(x: np.ndarray, bad: np.ndarray, estimator: str = "oas") -> np.ndarray:
    """Channel covariance of one epoch from its non-rejected samples. Shrinkage (OAS) keeps it
    positive definite: the average reference and interpolation leave the raw covariance rank-deficient."""
    from pyriemann.utils.covariance import covariances
    return covariances(x[None, :, ~bad], estimator=estimator)[0]
