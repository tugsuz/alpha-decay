"""
Event-study estimators shared by 02_decay_panel.py and 08_stage1_eventtime.py.

Both scripts face the same design: units (signals, or characteristics) treated at
staggered dates (publication years), outcomes observed before and after, and a two-way
fixed effects regression as the first pass. This module holds

    two_way_fe        OLS with unit and time effects, two-way clustered standard errors
    sun_abraham       the interaction-weighted estimator of Sun and Abraham (2021)
    iw_aggregate      the step that turns cohort-by-bin coefficients into one per bin

--------------------------------------------------------------------------------------
WHY A SECOND ESTIMATOR

    y_it = a_i + d_t + b post_it + e_it

With staggered adoption, b is a weighted average of every two-group comparison in the
panel (Goodman-Bacon 2021), including the ones that use already-treated units as
controls for later-treated ones. When the effect changes with time since treatment,
and decay that goes on for years is such an effect, those comparisons subtract the
early units' deepening effect from the late units' estimate. Binned event-time dummies
reduce the problem and do not remove it.

Sun and Abraham (2021) fit one set of event-time dummies per cohort,

    y_it = a_i + d_t + sum_g sum_l delta_{g,l} 1[G_i = g] 1[t - G_i in bin l] + e_it,

with a reference bin left out and a control cohort whose dummies are left out. Each
delta_{g,l} is then a clean comparison: cohort g in bin l against units that are not
yet treated or in the control cohort, in the same calendar periods. The bin
coefficient is the average of the delta_{g,l} over cohorts, weighted by each cohort's
share of the observations in that bin, and its standard error follows by the delta
method on the clustered covariance.

The control cohort is the never-treated units when there are any. In both panels here
every unit is eventually published, so the control cohort is the last-treated one:
units published in or after a cutoff year, and the sample stops the year the cutoff
cohort is treated. That trades window length for control size, and the callers report
more than one cutoff so the reader sees the trade.

Units treated before the sample starts have no untreated period and are dropped from
the estimation; so are treated cohorts with no observation in the reference bin, since
their dummies would be absorbed by the unit effects. Both counts are returned.

Fixed effects are absorbed by alternating demeaning, so the regression that is solved
has only the event-time columns; with one row per unit-period the two-way clustered
covariance of Cameron, Gelbach and Miller (2011) is V_unit + V_time - V_robust. That
matrix is not positive semi-definite in general; when it is not, the subtracted term is
dropped, which counts the within-cell variance twice and is conservative, and the
output says so. (Setting negative eigenvalues to zero, the other common fix, gave
different standard errors for the same coefficient depending on how many nuisance
parameters the matrix carried, which is why it is not used.)
--------------------------------------------------------------------------------------
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---- fixed effects and clustering ----------------------------------------------------

def demean_two_way(A: np.ndarray, g1: np.ndarray, g2: np.ndarray,
                   tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    """Remove group means along g1 and g2 from every column of A by alternating
    projections (Guimaraes and Portugal 2010). Exact for one grouping; converges to the
    two-way within transformation for two."""
    A = np.array(A, dtype=float, copy=True)
    n1, n2 = g1.max() + 1, g2.max() + 1
    c1 = np.bincount(g1, minlength=n1).astype(float)
    c2 = np.bincount(g2, minlength=n2).astype(float)
    for _ in range(max_iter):
        before = A.copy()
        for g, n, c in ((g1, n1, c1), (g2, n2, c2)):
            for j in range(A.shape[1]):
                m = np.bincount(g, weights=A[:, j], minlength=n) / c
                A[:, j] -= m[g]
        if np.abs(A - before).max() < tol:
            break
    return A


def two_way_cluster_cov(Xd: np.ndarray, u: np.ndarray, c_unit: np.ndarray,
                        c_time: np.ndarray, k_total: int) -> Tuple[np.ndarray, int]:
    """Cameron-Gelbach-Miller covariance for the coefficients on Xd (already within-
    transformed), residuals u. k_total counts the absorbed effects in the degrees of
    freedom correction. The two-way matrix V_unit + V_time - V_both is not positive
    semi-definite in general; when it is not, the third term is dropped, which counts
    the within-cell variance twice and so is conservative. Returns the matrix and a flag
    saying whether that happened."""
    n = len(u)
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    S = Xd * u[:, None]

    def meat(codes):
        G = codes.max() + 1
        sums = np.zeros((G, S.shape[1]))
        np.add.at(sums, codes, S)
        return (G / (G - 1)) * ((n - 1) / (n - k_total)) * (sums.T @ sums)

    c_both = pd.factorize(pd.Series(c_unit).astype(str) + "|" + pd.Series(c_time).astype(str))[0]
    M_two = meat(c_unit) + meat(c_time)
    V_two = XtX_inv @ M_two @ XtX_inv
    V = V_two - XtX_inv @ meat(c_both) @ XtX_inv
    w = np.linalg.eigvalsh((V + V.T) / 2)
    conservative = int(w.min() < -1e-12 * max(abs(w).max(), 1e-300))
    if conservative:
        V = V_two
    return V, conservative


def two_way_fe(df: pd.DataFrame, yvar: str, xvars: Sequence[str],
               unit: str = "characteristic", time: str = "year") -> pd.DataFrame:
    """OLS of yvar on xvars with unit and time effects; two-way clustered standard
    errors. One row per regressor."""
    d = df.dropna(subset=[yvar] + list(xvars))
    cu = pd.factorize(d[unit])[0]
    ct = pd.factorize(d[time])[0]
    A = demean_two_way(np.column_stack([d[yvar].to_numpy(float), d[list(xvars)].to_numpy(float)]), cu, ct)
    yd, Xd = A[:, 0], A[:, 1:]
    assert np.isfinite(A).all(), "non-finite values in the regression inputs"
    beta = np.linalg.pinv(Xd.T @ Xd) @ (Xd.T @ yd)
    u = yd - Xd @ beta
    k_total = Xd.shape[1] + cu.max() + ct.max() + 1
    V, conservative = two_way_cluster_cov(Xd, u, cu, ct, k_total)
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return pd.DataFrame({"term": list(xvars), "coef": beta, "se": se,
                         "t": beta / np.where(se > 0, se, np.nan), "n": len(d),
                         "clusters_unit": cu.max() + 1, "clusters_time": ct.max() + 1,
                         "conservative_cov": conservative})


# ---- Sun and Abraham ----------------------------------------------------------------

def iw_aggregate(coef, V, names, counts):
    """Aggregate cohort-by-bin coefficients into one coefficient per bin.

    coef: (k,) cohort x bin coefficients
    V: (k, k) clustered covariance, same order as coef
    names: k (cohort, bin) pairs, same order as coef
    counts: {(cohort, bin): number of observations in that cell}

    For bin l, w_g = counts[(g, l)] / sum_g counts[(g, l)], the bin
    coefficient is w'coef_l and its standard error is sqrt(w' V_ll w).
    Bins keep the order in which they first appear in names.
    """
    coef = np.asarray(coef, dtype=float)
    V = np.asarray(V, dtype=float)
    k = len(coef)
    if V.shape != (k, k) or len(names) != k:
        raise ValueError("coef, V and names must have matching sizes")

    bins = list(dict.fromkeys(b for _, b in names))
    rows = []
    for b in bins:
        idx = [i for i, (_, bb) in enumerate(names) if bb == b]
        n = np.array([counts.get(names[i], 0) for i in idx], dtype=float)
        keep = n > 0
        idx = [i for i, ok in zip(idx, keep) if ok]
        n = n[keep]
        if n.sum() == 0:
            rows.append((b, np.nan, np.nan, np.nan, 0, 0))
            continue
        w = n / n.sum()
        est = float(w @ coef[idx])
        var = float(w @ V[np.ix_(idx, idx)] @ w)
        se = np.sqrt(var) if var >= 0 else np.nan
        t = est / se if se and se > 0 else np.nan
        rows.append((b, est, se, t, len(idx), int(n.sum())))
    out = pd.DataFrame(rows, columns=["bin", "coef", "se", "t", "n_cohorts", "n_obs"])
    return out.set_index("bin")


def assign_bins(e: np.ndarray, bins: Sequence[Tuple[int, int, str]]) -> np.ndarray:
    out = np.array([None] * len(e), dtype=object)
    for lo, hi, name in bins:
        out[(e >= lo) & (e <= hi)] = name
    return out


def sun_abraham(df: pd.DataFrame, outcome: str, bins: Sequence[Tuple[int, int, str]],
                reference: str, control_min_cohort: int,
                unit: str = "unit", time: str = "time", year: str = "year",
                cohort: str = "cohort", pooled: bool = False) -> Dict:
    """Interaction-weighted event study.

    df      one row per unit and period, with the unit id, the period (used for the
            time effects and the time clusters), the calendar year of the period, the
            unit's treatment year (cohort) and the outcome
    bins    (low, high, name) in years relative to treatment; together they must cover
            every relative time that occurs
    reference           the bin whose dummies are left out
    control_min_cohort  units with cohort >= this are the control cohort; periods with
                        year >= this are dropped, so no control unit is ever treated
                        inside the sample
    pooled  True fits one set of bin dummies for all treated cohorts on the same
            sample and controls; this is the two-way fixed effects event study for a
            like-for-like comparison, and it is not the Sun-Abraham estimator

    Returns a dict: 'bins' (one row per bin: coef, se, t, n_cohorts, n_obs), 'post'
    (the average over every bin at or after treatment, cross-bin covariances included),
    'pre_wald' (joint test that the pre-treatment bins are zero), 'cells' (the
    cohort-by-bin estimates) and the sample counts.
    """
    d = df.dropna(subset=[outcome]).copy()
    d = d[d[year] < control_min_cohort]
    d["_e"] = d[year].astype(int) - d[cohort].astype(int)
    d["_bin"] = assign_bins(d["_e"].to_numpy(), bins)
    assert d["_bin"].notna().all(), "bins must cover every relative time in the sample"
    d["_treated"] = d[cohort] < control_min_cohort

    # units with no untreated period, and treated cohorts without a reference observation
    first_e = d[d["_treated"]].groupby(unit)["_e"].min()
    always = set(first_e[first_e >= 0].index)
    has_ref = d[d["_treated"] & (d["_bin"] == reference)].groupby(cohort).size()
    no_ref = set(d.loc[d["_treated"], cohort].unique()) - set(has_ref.index)
    d = d[~d[unit].isin(always) & ~(d["_treated"] & d[cohort].isin(no_ref))]

    key = d[cohort].where(d["_treated"], -1)
    if pooled:
        key = key.where(~d["_treated"], 0)
    cells = (d[d["_treated"] & (d["_bin"] != reference)]
             .groupby([key[d["_treated"] & (d["_bin"] != reference)], "_bin"]).size())
    names = [(int(g), str(b)) for (g, b) in cells.index]
    counts = {k: int(v) for k, v in cells.items()}
    D = np.zeros((len(d), len(names)))
    kb = list(zip(key.to_numpy(), d["_bin"].to_numpy()))
    pos = {nm: j for j, nm in enumerate(names)}
    treated = d["_treated"].to_numpy()
    for i, (g, b) in enumerate(kb):
        if treated[i]:
            j = pos.get((int(g), str(b)))
            if j is not None:
                D[i, j] = 1.0

    cu = pd.factorize(d[unit])[0]
    ct = pd.factorize(d[time])[0]
    A = demean_two_way(np.column_stack([d[outcome].to_numpy(float), D]), cu, ct)
    yd, Dd = A[:, 0], A[:, 1:]
    beta = np.linalg.pinv(Dd.T @ Dd) @ (Dd.T @ yd)
    u = yd - Dd @ beta
    k_total = Dd.shape[1] + cu.max() + ct.max() + 1
    V, conservative = two_way_cluster_cov(Dd, u, cu, ct, k_total)

    table = iw_aggregate(beta, V, names, counts)
    order = [name for _, _, name in bins if name in table.index]
    table = table.loc[order]

    # one post number: weights over every post cell, full covariance
    post_names = [nm for nm in names if any(lo >= 0 and nm[1] == b for lo, _, b in bins)]
    idx = [names.index(nm) for nm in post_names]
    w = np.array([counts[nm] for nm in post_names], float)
    w = w / w.sum()
    post_coef = float(w @ beta[idx])
    post_se = float(np.sqrt(max(w @ V[np.ix_(idx, idx)] @ w, 0)))
    post = pd.Series({"coef": post_coef, "se": post_se,
                      "t": post_coef / post_se if post_se > 0 else np.nan, "n_obs": int(sum(counts[nm] for nm in post_names))})

    # joint test that the pre-treatment bins (other than the reference) are zero
    pre_bins = [b for lo, hi, b in bins if hi < 0 and b != reference and b in table.index]
    W = np.zeros((len(names), len(pre_bins)))
    for j, b in enumerate(pre_bins):
        cell_idx = [i for i, nm in enumerate(names) if nm[1] == b]
        tot = sum(counts[names[i]] for i in cell_idx)
        for i in cell_idx:
            W[i, j] = counts[names[i]] / tot
    b_pre = W.T @ beta
    V_pre = W.T @ V @ W
    stat = float(b_pre @ np.linalg.pinv(V_pre) @ b_pre) if len(pre_bins) else np.nan
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(stat, len(pre_bins))) if len(pre_bins) else np.nan
    except ImportError:                      # scipy absent: report the statistic only
        p = np.nan
    pre_wald = {"stat": stat, "df": len(pre_bins), "p": p}

    cell_table = pd.DataFrame({"cohort": [g for g, _ in names], "bin": [b for _, b in names],
                               "coef": beta, "se": np.sqrt(np.maximum(np.diag(V), 0)),
                               "n_obs": [counts[nm] for nm in names]})
    treated_units = d.loc[d["_treated"], unit].nunique()
    control_units = d.loc[~d["_treated"], unit].nunique()
    return {"bins": table, "post": post, "pre_wald": pre_wald, "cells": cell_table,
            "n_obs": len(d), "n_treated_units": int(treated_units), "n_control_units": int(control_units),
            "n_always_treated_dropped": len(always), "n_cohorts_no_reference": len(no_ref),
            "n_cohorts": int(cells.index.get_level_values(0).nunique()),
            "period_first": d[time].min(), "period_last": d[time].max(), "conservative_cov": conservative,
            "control_min_cohort": control_min_cohort, "pooled": pooled}
