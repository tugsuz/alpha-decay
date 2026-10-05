"""
STEP 8 -- what the model uses, in event time around each characteristic's publication.

    python 08_stage1_eventtime.py
    python 08_stage1_eventtime.py --synthetic     # on the output of 07 --synthetic

Reads   output/stage1_importance.csv   (07_stage1_models.py) and jkp_characteristics.csv,
        output/stage1_r2.csv and output/stage1_portfolio_summary.csv for the figures.
Writes  output/stage1_eventtime.csv      the fixed-effects estimates
        output/fig07_eventtime.png        importance against years since publication
        output/fig07_r2_by_year.png       public and full R2 by test year

--------------------------------------------------------------------------------------
Each row of stage1_importance.csv is one characteristic's permutation importance in one
test year for the full-information model: the rise in test mean squared error when that
column is shuffled. Put it in event time, e = test year minus the characteristic's
publication year, and ask whether the model leans less on a characteristic once it is
public:

    imp_{c,y} = a_c + d_y + b * post_{c,y} + e_{c,y},      post = 1[y >= pub_year_c]

Characteristic fixed effects a_c absorb how useful a characteristic is on average; year
fixed effects d_y absorb anything that moves all importances in a year, including the
calendar-time decline the paper worries about. b is identified from the characteristics
whose publication falls inside the test window. Standard errors are clustered two ways,
by characteristic and by year. A second specification replaces post with event-time
bins, the five years before publication being the reference.

Importance is used two ways: in units of mean squared error (scaled by 1e4), and as a
share of the year's total positive importance, which removes the level differences
between years.

A caveat that belongs with the table. Publication dates are staggered, so this is a
two-way fixed effects event study, and when the effect differs across publication
cohorts the binned coefficients mix clean comparisons with comparisons that use
already-published characteristics as controls (Goodman-Bacon 2021; Sun and Abraham
2021). The table here is the first pass. A Sun-Abraham interaction-weighted version, or
a stacked regression with one clean window per cohort, is the planned robustness check
once the first results are in.
--------------------------------------------------------------------------------------
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "output"
CHARS_FILE = HERE / "jkp_characteristics.csv"

BINS = [(-100, -6, "-10 to -6"), (-5, -1, "-5 to -1"), (0, 4, "0 to 4"), (5, 9, "5 to 9"), (10, 100, "10+")]
REFERENCE = "-5 to -1"


def two_way_fe(df, yvar, xvars, unit="characteristic", time="year"):
    """OLS with unit and time dummies; covariance clustered by unit and by time
    (Cameron, Gelbach and Miller 2011: V_unit + V_time - V_both)."""
    d = df.dropna(subset=[yvar] + xvars).copy()
    X = [d[xvars].to_numpy(float)]
    for key in (unit, time):
        dummies = pd.get_dummies(d[key], drop_first=(key == time)).to_numpy(float)
        X.append(dummies)
    X = np.hstack(X)
    y = d[yvar].to_numpy(float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    n, k = X.shape

    def cluster_V(codes):
        G = codes.max() + 1
        meat = np.zeros((k, k))
        S = X * u[:, None]
        for g in range(G):
            s = S[codes == g].sum(axis=0)
            meat += np.outer(s, s)
        return (G / (G - 1)) * ((n - 1) / (n - k)) * (XtX_inv @ meat @ XtX_inv)

    cu = pd.factorize(d[unit])[0]
    ct = pd.factorize(d[time])[0]
    cb = pd.factorize(d[unit].astype(str) + "|" + d[time].astype(str))[0]
    V = cluster_V(cu) + cluster_V(ct) - cluster_V(cb)
    se = np.sqrt(np.maximum(np.diag(V)[:len(xvars)], 0))
    return pd.DataFrame({"term": xvars, "coef": beta[:len(xvars)], "se": se,
                         "t": beta[:len(xvars)] / np.where(se > 0, se, np.nan),
                         "n": n, "clusters_unit": cu.max() + 1, "clusters_time": ct.max() + 1})


def main():
    global OUT
    if "--synthetic" in sys.argv[1:]:        # the folder a 07 --synthetic run writes to
        OUT = OUT / "synthetic"
    imp_file = OUT / "stage1_importance.csv"
    if not imp_file.exists():
        sys.exit("output/stage1_importance.csv is missing; run 07_stage1_models.py first")
    imp = pd.read_csv(imp_file)
    chars = pd.read_csv(CHARS_FILE).set_index("characteristic")
    imp["pub_year"] = imp["characteristic"].map(chars["pub_year"]).astype(int)
    imp["event_time"] = imp["year"] - imp["pub_year"]
    imp["post"] = (imp["event_time"] >= 0).astype(float)
    imp["imp_bp"] = imp["mse_increase"] * 1e4
    pos = imp["mse_increase"].clip(lower=0)
    imp["share"] = 100 * pos / pos.groupby([imp["model"], imp["year"]]).transform("sum")
    for lo, hi, name in BINS:
        imp[f"bin_{name}"] = ((imp["event_time"] >= lo) & (imp["event_time"] <= hi)).astype(float)
    bin_cols = [f"bin_{name}" for _, _, name in BINS if name != REFERENCE]

    rows = []
    for model, g in imp.groupby("model"):
        if g["post"].nunique() < 2:
            print(f"{model}: no variation in post within the window, skipped")
            continue
        for yvar in ("imp_bp", "share"):
            r = two_way_fe(g, yvar, ["post"]); r["model"] = model; r["outcome"] = yvar; r["spec"] = "post"
            rows.append(r)
            if g[bin_cols].sum().min() > 0:
                r = two_way_fe(g, yvar, bin_cols); r["model"] = model; r["outcome"] = yvar; r["spec"] = "bins"
                rows.append(r)
    if not rows:
        sys.exit("nothing to estimate")
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(OUT / "stage1_eventtime.csv", index=False)
    print(res[["model", "outcome", "spec", "term", "coef", "se", "t", "n"]].round(4).to_string(index=False))

    # figure: binned event-time coefficients, share outcome, one line per model
    fig, ax = plt.subplots(figsize=(7, 4))
    b = res[(res["spec"] == "bins") & (res["outcome"] == "share")]
    names = [n for _, _, n in BINS]
    xpos = {n: i for i, n in enumerate(names)}
    for k, (model, g) in enumerate(b.groupby("model")):
        g = g.set_index("term")
        xs = [xpos[REFERENCE]] + [xpos[t.replace("bin_", "")] for t in g.index]
        ys = [0.0] + g["coef"].tolist()
        es = [0.0] + (1.96 * g["se"]).tolist()
        order = np.argsort(xs)
        ax.errorbar(np.array(xs)[order] + 0.08 * k, np.array(ys)[order], yerr=np.array(es)[order],
                    marker="o", capsize=3, lw=1.3, label=model)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(xpos["-5 to -1"] + 0.5, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_xlabel("years since publication (reference: -5 to -1)")
    ax.set_ylabel("change in importance share, percentage points")
    ax.set_title("What the full model uses, around each characteristic's publication", loc="left")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig07_eventtime.png", dpi=160)

    # figure: R2 by year, public and full, for the models in the importance file
    r2_file = OUT / "stage1_r2.csv"
    if r2_file.exists():
        r2 = pd.read_csv(r2_file)
        r2 = r2[r2["year"].astype(str) != "all"].copy()
        r2["year"] = r2["year"].astype(int)
        fig, ax = plt.subplots(figsize=(7, 4))
        for model, ls in [("huber", "--"), ("gbrt", "-"), ("nn3", ":")]:
            for s, color in [("public", "#777777"), ("full", "#1f77b4")]:
                g = r2[(r2["model"] == model) & (r2["set"] == s)].sort_values("year")
                if len(g):
                    ax.plot(g["year"], 100 * g["r2"], ls=ls, color=color, lw=1.4, label=f"{model}, {s}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("out-of-sample R², %"); ax.set_xlabel("test year")
        ax.set_title("Public and full information sets, by test year", loc="left")
        ax.legend(frameon=False, fontsize=8, ncol=2); ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(OUT / "fig07_r2_by_year.png", dpi=160)
    print(f"wrote stage1_eventtime.csv and fig07_*.png to {OUT}")


if __name__ == "__main__":
    main()
