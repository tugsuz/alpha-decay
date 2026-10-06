"""
STEP 8 -- what the model uses, in event time around each characteristic's publication.

    python 08_stage1_eventtime.py
    python 08_stage1_eventtime.py --synthetic     # on the output of 07 --synthetic

Reads   output/stage1_importance.csv   (07_stage1_models.py) and jkp_characteristics.csv,
        output/stage1_r2.csv and output/stage1_portfolio_summary.csv for the figures.
Writes  output/stage1_eventtime.csv      the fixed-effects estimates
        output/stage1_sunabraham.csv     the same outcomes with the Sun-Abraham estimator, and
                                         the two-way FE bins and post dummy on its sample
        output/fig07_sunabraham.png       Huber importance share: two-way FE bins against Sun-Abraham
        output/fig07_eventtime.png        importance against years since publication
        output/fig07_decomposition.png    Huber model: importance share, |slope|, the characteristic's own IC
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
by characteristic and by year (Cameron, Gelbach and Miller 2011); when that two-way
matrix is not positive semi-definite the within-cell term is not subtracted, which is
conservative, and the output flags it. A second specification replaces post with event-time bins, the five years
before publication being the reference. Both estimators live in event_study.py.

Importance is used two ways: in units of mean squared error (scaled by 1e4), and as a
share of the year's total positive importance, which removes the level differences
between years.

Permutation importance in a linear model is about 2 beta Cov(y - rest, x) on the test rows:
it rises when the model leans harder on the characteristic (beta) and when the
characteristic still covaries with the return out of sample. The two are separated with
the same design on two more outcomes, read from the csvs 07 writes beside the forecasts:
the Huber regression's slope on each characteristic (log |beta|) and the characteristic's
own monthly Spearman correlation with the realised return over the test year.

A caveat that belongs with the table. Publication dates are staggered, so this is a
two-way fixed effects event study, and when the effect differs across publication
cohorts the binned coefficients mix clean comparisons with comparisons that use
already-published characteristics as controls (Goodman-Bacon 2021; Sun and Abraham
2021). stage1_sunabraham.csv re-estimates every outcome with the Sun-Abraham
interaction-weighted estimator: one set of bin dummies per publication cohort, the
last-published characteristics as the control cohort (publication year at or after a
cutoff; the sample stops at the cutoff year), and bin coefficients averaged over cohorts
with each cohort's share of the bin. Characteristics published before the first test
year have no untreated period and are dropped from that estimation. Two cutoffs are
reported, since a later cutoff keeps more years and fewer controls.
--------------------------------------------------------------------------------------
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from event_study import two_way_fe, sun_abraham

HERE = Path(__file__).parent
OUT = HERE / "output"
CHARS_FILE = HERE / "jkp_characteristics.csv"

BINS = [(-100, -6, "-10 to -6"), (-5, -1, "-5 to -1"), (0, 4, "0 to 4"), (5, 9, "5 to 9"), (10, 100, "10+")]
REFERENCE = "-5 to -1"
SA_CUTOFFS = [2016, 2018]        # control cohort: published in or after; the sample stops there


def main():
    global OUT
    if "--synthetic" in sys.argv[1:]:        # the folder a 07 --synthetic run writes to
        OUT = OUT / "synthetic"
    imp_file = OUT / "stage1_importance.csv"
    if not imp_file.exists():
        sys.exit("output/stage1_importance.csv is missing; run 07_stage1_models.py first")
    imp = pd.read_csv(imp_file)
    manifest = OUT / "stage1_run.json"
    if not manifest.exists():
        sys.exit(f"{manifest} is missing; run 07_stage1_models.py first")
    run_id = json.loads(manifest.read_text())["run_id"]
    keep = (imp["run_id"] == run_id) if "run_id" in imp else pd.Series(False, index=imp.index)
    if (~keep).any():
        print(f"dropping {(~keep).sum():,} importance rows from other runs; keeping run {run_id}")
        imp = imp[keep]
    if imp.empty:
        sys.exit("no importance rows from the current run")
    r2_file = OUT / "stage1_r2.csv"
    if r2_file.exists():                    # the forecast years of the run that built the tables
        yrs = pd.read_csv(r2_file)["year"]
        run_years = sorted(set(int(v) for v in yrs[yrs.astype(str) != "all"]))
        stale = ~imp["year"].isin(run_years)
        if stale.any():
            print(f"dropping {stale.sum():,} importance rows from years without forecasts in this run: "
                  f"{sorted(set(imp.loc[stale, 'year']))}")
            imp = imp[~stale]
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
    # the decomposition: the Huber slope on each characteristic (how much the model leans on
    # it) and the characteristic's own monthly Spearman with the realised return (whether it
    # still works), each in the same event-time design
    extra = []
    coef_file, uic_file = OUT / "stage1_huber_coef.csv", OUT / "stage1_char_ic.csv"
    if coef_file.exists():
        cf = pd.read_csv(coef_file)
        cf = cf[(cf["run_id"] == run_id) & (cf["set"] == "full") & cf["year"].isin(imp["year"].unique())].copy()
        cf["log_abs_beta"] = np.log(np.abs(cf["beta"]).clip(lower=1e-8))
        cf["abs_beta_bp"] = np.abs(cf["beta"]) * 1e4
        extra.append(("huber_beta", cf, ["log_abs_beta", "abs_beta_bp"]))
    if uic_file.exists():
        uf = pd.read_csv(uic_file)
        uf = uf[(uf["run_id"] == run_id) & uf["year"].isin(imp["year"].unique())].copy()
        uf["char_ic_x100"] = 100 * uf["ic_mean"]
        uf["abs_char_ic_x100"] = 100 * uf["ic_mean"].abs()
        extra.append(("char_ic", uf, ["char_ic_x100", "abs_char_ic_x100"]))
    for label, df, yvars in extra:
        df["pub_year"] = df["characteristic"].map(chars["pub_year"]).astype(int)
        df["event_time"] = df["year"] - df["pub_year"]
        df["post"] = (df["event_time"] >= 0).astype(float)
        for lo, hi, name in BINS:
            df[f"bin_{name}"] = ((df["event_time"] >= lo) & (df["event_time"] <= hi)).astype(float)
        if df["post"].nunique() < 2:
            continue
        for yvar in yvars:
            r = two_way_fe(df, yvar, ["post"]); r["model"] = label; r["outcome"] = yvar; r["spec"] = "post"
            rows.append(r)
            if df[bin_cols].sum().min() > 0:
                r = two_way_fe(df, yvar, bin_cols); r["model"] = label; r["outcome"] = yvar; r["spec"] = "bins"
                rows.append(r)

    if not rows:
        sys.exit("nothing to estimate")
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(OUT / "stage1_eventtime.csv", index=False)
    print(res[["model", "outcome", "spec", "term", "coef", "se", "t", "n"]].round(4).to_string(index=False))

    # the same outcomes with the Sun-Abraham estimator, two control cutoffs, and the
    # pooled two-way FE bins on the same sample for a like-for-like comparison
    sa_rows = []
    panels_sa = [(model, g, ["imp_bp", "share"]) for model, g in imp.groupby("model")] + \
                [(label, df, yvars) for label, df, yvars in extra]
    for cutoff in SA_CUTOFFS:
        for label, df, yvars in panels_sa:
            for yvar in yvars:
                for pooled in (False, True):
                    try:
                        r = sun_abraham(df.rename(columns={"characteristic": "unit", "pub_year": "cohort"})
                                        .assign(time=lambda x: x["year"]),
                                        yvar, BINS, REFERENCE, cutoff, pooled=pooled)
                    except (AssertionError, ValueError, IndexError) as e:
                        print(f"Sun-Abraham skipped for {label} {yvar} cutoff {cutoff}: {e}")
                        continue
                    meta = {"model": label, "outcome": yvar, "control_cutoff": cutoff,
                            "estimator": "twfe_pooled" if pooled else "sun_abraham",
                            "n_obs": r["n_obs"], "n_treated_units": r["n_treated_units"],
                            "n_control_units": r["n_control_units"], "n_cohorts": r["n_cohorts"],
                            "n_always_treated_dropped": r["n_always_treated_dropped"],
                            "period_first": r["period_first"], "period_last": r["period_last"],
                            "conservative_cov": r["conservative_cov"]}
                    for b, row in r["bins"].iterrows():
                        sa_rows.append(dict(meta, term=b, coef=row["coef"], se=row["se"], t=row["t"],
                                            n_cohorts_in_bin=row["n_cohorts"], n_obs_in_bin=row["n_obs"]))
                    sa_rows.append(dict(meta, term="post", coef=r["post"]["coef"], se=r["post"]["se"],
                                        t=r["post"]["t"], n_obs_in_bin=r["post"]["n_obs"],
                                        wald_pre_stat=r["pre_wald"]["stat"], wald_pre_df=r["pre_wald"]["df"],
                                        wald_pre_p=r["pre_wald"]["p"]))
                    if pooled:
                        # the first-pass regression, a single post dummy, on the same sample:
                        # test years before the cutoff, without the characteristics that
                        # were already public in the first test year
                        sub = df[df["year"] < cutoff]
                        first_e = sub.groupby("characteristic")["event_time"].min()
                        sub = sub[~sub["characteristic"].isin(first_e[first_e >= 0].index)]
                        st = two_way_fe(sub, yvar, ["post"]).iloc[0]
                        sa_rows.append(dict(meta, estimator="twfe_static", term="post", coef=st["coef"],
                                            se=st["se"], t=st["t"], n_obs_in_bin=int(st["n"]),
                                            conservative_cov=int(st["conservative_cov"])))
    sa = pd.DataFrame(sa_rows)
    sa.to_csv(OUT / "stage1_sunabraham.csv", index=False)
    show = sa[(sa["term"] == "post")][["model", "outcome", "control_cutoff", "estimator", "coef", "se", "t",
                                        "n_treated_units", "n_control_units", "period_last", "wald_pre_p"]]
    print("\nSun-Abraham, the post average over cohorts and bins:")
    print(show.round(4).to_string(index=False))

    # figure: Huber importance share, two-way FE bins against Sun-Abraham, main cutoff
    sub = sa[(sa["model"] == "huber") & (sa["outcome"] == "share") & (sa["control_cutoff"] == SA_CUTOFFS[0])]
    if len(sub):
        names = [n for _, _, n in BINS]
        xpos = {n: i for i, n in enumerate(names)}
        fig, ax = plt.subplots(figsize=(7, 4))
        for k, (est, lab, color) in enumerate([("twfe_pooled", "two-way fixed effects", "#777777"),
                                               ("sun_abraham", "Sun-Abraham", "#1f77b4")]):
            g = sub[(sub["estimator"] == est) & (sub["term"] != "post")].set_index("term")
            xs = [xpos[REFERENCE]] + [xpos[t] for t in g.index]
            ys = [0.0] + g["coef"].tolist()
            es = [0.0] + (1.96 * g["se"]).tolist()
            order = np.argsort(xs)
            ax.errorbar(np.array(xs)[order] + 0.08 * k, np.array(ys)[order], yerr=np.array(es)[order],
                        marker="o", capsize=3, lw=1.3, color=color, label=lab)
        ax.axhline(0, color="black", lw=0.8); ax.axvline(xpos[REFERENCE] + 0.5, color="grey", lw=0.8, ls="--")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
        ax.set_xlabel("years since publication (reference: -5 to -1)")
        ax.set_ylabel("change in importance share, percentage points")
        r0 = sub.iloc[0]
        ax.set_title(f"Huber model, importance share\ncontrol cohort published {SA_CUTOFFS[0]} or later, "
                     f"test years to {int(r0['period_last'])}", loc="left", fontsize=10)
        ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(OUT / "fig07_sunabraham.png", dpi=160)

    # figure: the decomposition for the Huber model, three panels of binned coefficients
    panels = [("huber", "share", "importance share, pp"), ("huber_beta", "log_abs_beta", "log |slope|"),
              ("char_ic", "char_ic_x100", "characteristic's own IC, x100")]
    have = [(m, o, t) for m, o, t in panels if ((res["model"] == m) & (res["outcome"] == o) & (res["spec"] == "bins")).any()]
    if len(have) > 1:
        names = [n for _, _, n in BINS]
        xpos = {n: i for i, n in enumerate(names)}
        fig, axes = plt.subplots(1, len(have), figsize=(3.2 * len(have), 3.6), sharex=True)
        for ax, (m, o, title) in zip(np.atleast_1d(axes), have):
            g = res[(res["model"] == m) & (res["outcome"] == o) & (res["spec"] == "bins")].set_index("term")
            xs = [xpos[REFERENCE]] + [xpos[t.replace("bin_", "")] for t in g.index]
            ys = [0.0] + g["coef"].tolist()
            es = [0.0] + (1.96 * g["se"]).tolist()
            order = np.argsort(xs)
            ax.errorbar(np.array(xs)[order], np.array(ys)[order], yerr=np.array(es)[order],
                        marker="o", capsize=3, lw=1.3, color="#1f77b4")
            ax.axhline(0, color="black", lw=0.8); ax.axvline(xpos["-5 to -1"] + 0.5, color="grey", lw=0.8, ls="--")
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
            ax.set_title(title, loc="left", fontsize=10); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
        axes_list = np.atleast_1d(axes)
        axes_list[0].set_ylabel("change against years -5 to -1")
        fig.suptitle("Huber model: reliance on a characteristic, and the characteristic itself, around publication",
                     x=0.01, ha="left", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / "fig07_decomposition.png", dpi=160)

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
        if "universe" in r2:
            r2 = r2[r2["universe"] == "all"]
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
