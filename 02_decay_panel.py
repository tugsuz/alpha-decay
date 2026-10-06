"""
alpha-decay, step 2 (Result 1): the McLean-Pontiff three-window decay, across ALL signals.

Step 1 looked at one anomaly. This is the headline number: averaged over every usable
predictor in the Chen-Zimmermann universe, how much of the published edge survives
(a) the end of the original paper's sample, and (b) publication itself?

    python 02_decay_panel.py

Needs the same two files as step 1, in data/:
    data/PredictorLSretWide.csv
    data/SignalDoc.csv

Writes:
    output/result1_windows.csv     one row per signal: mean return in each window
    output/result1_summary.txt     the headline table, copy-pasteable
    output/fig02_decay.png
    output/result1_sunabraham.csv  the publication effect with the Sun-Abraham estimator
    output/fig02_sunabraham.png    its event-time bins against the two-way FE bins

--------------------------------------------------------------------------------------
THE THREE WINDOWS, and why there are three and not two

    1. IN-SAMPLE          date <= SampleEndYear        the paper's own sample
    2. POST-SAMPLE        SampleEndYear < date < Year  discovered, not yet public
    3. POST-PUBLICATION   date >= Year                 public, and being traded on

If the edge decays because publication invites arbitrage, window 2 should look like
window 1 and window 3 should fall. If it decays because the original sample was lucky
(overfitting, data mining), window 2 should ALREADY be lower. The gap between the two
declines is the part attributable to publication. That is the whole design.

McLean and Pontiff (2016) report roughly -26% for window 2 and -58% for window 3 over
97 anomalies. This script recomputes it over the current universe and through the end
of the data.

THE STAGGERED-ADOPTION CHECK

Publication arrives at a different date for every signal, so the panel regression is a
two-way fixed effects design with staggered treatment, and its post coefficient mixes
clean comparisons with ones that use already-published signals as controls for later
ones (Goodman-Bacon 2021). The second estimate here is the Sun and Abraham (2021)
interaction-weighted event study from event_study.py: bin dummies per publication
cohort, averaged over cohorts with each cohort's share of the bin. Every signal in this
universe is eventually published, so the control cohort is the last-published group,
signals published in or after a cutoff year, and the sample stops at that year. A later
cutoff keeps more years and fewer controls; the csv carries three cutoffs so the reader
sees the trade. Event time is measured from publication, so the reference bin, the five
years before publication, sits mostly inside the post-sample window (publication comes a
median four years after the sample ends), and the earlier bins show the in-sample level
against it.
--------------------------------------------------------------------------------------
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from event_study import sun_abraham

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "output"

PORTFOLIO_FILE = DATA / "PredictorLSretWide.csv"
SIGNALDOC_FILE = DATA / "SignalDoc.csv"

# Column names verified against the OpenSourceAP/CrossSection SignalDoc.
DOC_NAME_COL = "Acronym"
DOC_YEAR_COL = "Year"
DOC_SAMPEND  = "SampleEndYear"
DOC_CATEGORY = "Cat.Signal"
DOC_QUALITY  = "Signal Rep Quality"

# Keep only signals the authors class as real predictors, not placebos.
KEEP_CATEGORY = "Predictor"
# Minimum months of data required in EVERY window, so a signal cannot enter the
# average on the strength of three observations.
MIN_MONTHS = 24

# Sun-Abraham: bins in years since publication, the control cutoffs (signals published in
# or after the cutoff form the control cohort; the sample ends the December before)
SA_BINS = [(-1000, -11, "-11 or earlier"), (-10, -6, "-10 to -6"), (-5, -1, "-5 to -1"),
           (0, 4, "0 to 4"), (5, 9, "5 to 9"), (10, 14, "10 to 14"), (15, 1000, "15+")]
SA_REFERENCE = "-5 to -1"
SA_CUTOFFS = [2014, 2016, 2011]


def _require(path: Path) -> None:
    if not path.exists():
        sys.exit(
            f"Not found: {path}\n\n"
            "Download both files from https://www.openassetpricing.com (Download tab)\n"
            f"into {DATA}/ and run this again."
        )


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    _require(PORTFOLIO_FILE)
    _require(SIGNALDOC_FILE)

    rets = pd.read_csv(PORTFOLIO_FILE)
    date_col = "date" if "date" in rets.columns else rets.columns[0]
    rets[date_col] = pd.to_datetime(rets[date_col])
    rets = rets.rename(columns={date_col: "date"})

    doc = pd.read_csv(SIGNALDOC_FILE)
    missing = [c for c in (DOC_NAME_COL, DOC_YEAR_COL, DOC_SAMPEND) if c not in doc.columns]
    if missing:
        sys.exit(f"SignalDoc is missing {missing}. Columns present: {list(doc.columns)}")
    return rets, doc


def build_long(rets: pd.DataFrame, doc: pd.DataFrame) -> pd.DataFrame:
    """Wide -> long, one row per (signal, month), with the window label attached."""
    if DOC_CATEGORY in doc.columns:
        doc = doc[doc[DOC_CATEGORY] == KEEP_CATEGORY]
    doc = doc.dropna(subset=[DOC_YEAR_COL, DOC_SAMPEND])
    # A signal is only usable if the paper's sample ended BEFORE it was published;
    # otherwise window 2 does not exist and the design collapses to two periods.
    doc = doc[doc[DOC_SAMPEND] < doc[DOC_YEAR_COL]]

    usable = [c for c in rets.columns if c != "date" and c in set(doc[DOC_NAME_COL])]
    if not usable:
        sys.exit("No overlap between the returns columns and SignalDoc acronyms.")

    long = (rets[["date"] + usable]
            .melt(id_vars="date", var_name="signal", value_name="ret")
            .dropna(subset=["ret"]))

    meta = doc.set_index(DOC_NAME_COL)[[DOC_YEAR_COL, DOC_SAMPEND]]
    long["pub_year"] = long["signal"].map(meta[DOC_YEAR_COL]).astype(int)
    long["samp_end"] = long["signal"].map(meta[DOC_SAMPEND]).astype(int)
    long["year"] = long["date"].dt.year

    long["window"] = np.select(
        [long["year"] <= long["samp_end"],
         (long["year"] > long["samp_end"]) & (long["year"] < long["pub_year"])],
        ["1_in_sample", "2_post_sample"],
        default="3_post_pub",
    )
    return long


def per_signal_table(long: pd.DataFrame) -> pd.DataFrame:
    """Mean monthly return per signal per window, keeping only complete signals."""
    g = (long.groupby(["signal", "window"])["ret"]
              .agg(["mean", "size"])
              .unstack("window"))
    means = g["mean"]
    sizes = g["size"].fillna(0)
    complete = (sizes >= MIN_MONTHS).all(axis=1)
    out = means[complete].copy()
    out.columns = [c.split("_", 1)[1] for c in out.columns]
    return out.dropna()


def panel_regression(long: pd.DataFrame) -> dict:
    """
    r_it = a_i + b1*post_sample_it + b2*post_pub_it + e_it

    Signal fixed effects are absorbed by within-demeaning, so b1 and b2 are identified
    from variation over time WITHIN a signal, not from some signals being stronger than
    others. Standard errors are clustered by DATE, because in any given month every
    long-short portfolio is exposed to the same market.

    Written out in numpy: the cluster-robust sandwich is four lines, and the point of
    the exercise is to know what it does.
    """
    d = long.copy()
    d["post_sample"] = (d["window"] == "2_post_sample").astype(float)
    d["post_pub"] = (d["window"] == "3_post_pub").astype(float)

    cols = ["ret", "post_sample", "post_pub"]
    dm = d[cols] - d.groupby("signal")[cols].transform("mean")

    y = dm["ret"].to_numpy()
    X = dm[["post_sample", "post_pub"]].to_numpy()

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    # cluster-robust (date) sandwich: bread @ meat @ bread
    codes = pd.factorize(d["date"].to_numpy())[0]
    G = codes.max() + 1
    u = X * resid[:, None]
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in range(G):
        ug = u[codes == g].sum(axis=0)
        meat += np.outer(ug, ug)

    n, k = X.shape
    dof = (G / (G - 1)) * ((n - 1) / (n - k))          # standard small-sample correction
    V = dof * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(V))

    return {"post_sample": beta[0], "post_pub": beta[1],
            "se_post_sample": se[0], "se_post_pub": se[1],
            "t_post_sample": beta[0] / se[0], "t_post_pub": beta[1] / se[1],
            "n_obs": n, "n_dates": G}


def sun_abraham_table(long: pd.DataFrame) -> pd.DataFrame:
    """The publication effect on the monthly long-short return, cohort by cohort, for
    each control cutoff, with the pooled two-way FE bins on the same sample beside it."""
    d = long.rename(columns={"signal": "unit", "pub_year": "cohort"}).assign(time=lambda x: x["date"])
    rows = []
    for cutoff in SA_CUTOFFS:
        for pooled in (False, True):
            r = sun_abraham(d, "ret", SA_BINS, SA_REFERENCE, cutoff, pooled=pooled)
            meta = {"control_cutoff": cutoff, "estimator": "twfe_pooled" if pooled else "sun_abraham",
                    "n_obs": r["n_obs"], "n_treated_units": r["n_treated_units"],
                    "n_control_units": r["n_control_units"], "n_cohorts": r["n_cohorts"],
                    "period_first": f"{r['period_first']:%Y-%m}", "period_last": f"{r['period_last']:%Y-%m}",
                    "conservative_cov": r["conservative_cov"]}
            for b, row in r["bins"].iterrows():
                rows.append(dict(meta, term=b, coef=row["coef"], se=row["se"], t=row["t"],
                                 n_cohorts_in_bin=row["n_cohorts"], n_obs_in_bin=row["n_obs"]))
            rows.append(dict(meta, term="post", coef=r["post"]["coef"], se=r["post"]["se"], t=r["post"]["t"],
                             n_obs_in_bin=r["post"]["n_obs"], wald_pre_stat=r["pre_wald"]["stat"],
                             wald_pre_df=r["pre_wald"]["df"], wald_pre_p=r["pre_wald"]["p"]))
            print(f"  cutoff {cutoff} {'two-way FE' if pooled else 'Sun-Abraham'}: post "
                  f"{r['post']['coef']:+.4f} (t = {r['post']['t']:.2f}), {r['n_treated_units']} treated, "
                  f"{r['n_control_units']} control signals, to {r['period_last']:%Y-%m}")
    return pd.DataFrame(rows)


def sun_abraham_figure(sa: pd.DataFrame, path: Path) -> None:
    sub = sa[sa["control_cutoff"] == SA_CUTOFFS[0]]
    names = [n for _, _, n in SA_BINS]
    xpos = {n: i for i, n in enumerate(names)}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, (est, lab, color) in enumerate([("twfe_pooled", "two-way fixed effects", "#777777"),
                                           ("sun_abraham", "Sun-Abraham", "#1f77b4")]):
        g = sub[(sub["estimator"] == est) & (sub["term"] != "post")].set_index("term")
        xs = [xpos[SA_REFERENCE]] + [xpos[t] for t in g.index]
        ys = [0.0] + g["coef"].tolist()
        es = [0.0] + (1.96 * g["se"]).tolist()
        order = np.argsort(xs)
        ax.errorbar(np.array(xs)[order] + 0.08 * k, np.array(ys)[order], yerr=np.array(es)[order],
                    marker="o", capsize=3, lw=1.3, color=color, label=lab)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(xpos[SA_REFERENCE] + 0.5, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
    ax.set_xlabel("years since publication (reference: -5 to -1)")
    ax.set_ylabel("change in long-short return, %/month")
    r0 = sub.iloc[0]
    ax.set_title(f"Published edges in event time: control cohort published {SA_CUTOFFS[0]} or later, "
                 f"sample to {r0['period_last']}", loc="left", fontsize=10)
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=160)


def main() -> None:
    rets, doc = load()
    long = build_long(rets, doc)
    table = per_signal_table(long)

    n = len(table)
    m1, m2, m3 = table["in_sample"].mean(), table["post_sample"].mean(), table["post_pub"].mean()
    d2, d3 = 100 * (m2 - m1) / abs(m1), 100 * (m3 - m1) / abs(m1)

    res = panel_regression(long)
    b1, b2 = res["post_sample"], res["post_pub"]
    t1, t2 = res["t_post_sample"], res["t_post_pub"]

    lines = []
    add = lines.append
    add("RESULT 1 --- three-window decay, Chen-Zimmermann universe")
    add("=" * 72)
    add(f"signals used                      : {n}")
    add(f"months                            : {long['date'].min():%Y-%m} to {long['date'].max():%Y-%m}")
    add(f"signal-month observations         : {len(long):,}")
    add("")
    add("Equal-weighted across signals, mean monthly long-short return (%/month):")
    add(f"  1. in the paper's own sample     : {m1:6.3f}")
    add(f"  2. post-sample, pre-publication  : {m2:6.3f}   ({d2:+.0f}% vs window 1)")
    add(f"  3. post-publication              : {m3:6.3f}   ({d3:+.0f}% vs window 1)")
    add("")
    add("Panel regression, signal fixed effects, SEs clustered by date:")
    add(f"  post-sample   {b1:+.4f} %/month   (t = {t1:5.2f})")
    add(f"  post-publication {b2:+.4f} %/month   (t = {t2:5.2f})")
    add(f"  implied decline vs in-sample mean: {100*b1/abs(m1):+.0f}% and {100*b2/abs(m1):+.0f}%")
    add("")
    add("McLean and Pontiff (2016): about -26% post-sample and -58% post-publication,")
    add("over 97 anomalies. Differences come from universe, sample end, and weighting.")

    OUT.mkdir(exist_ok=True)
    print("Sun-Abraham event study, monthly long-short return:")
    sa = sun_abraham_table(long)
    sa.to_csv(OUT / "result1_sunabraham.csv", index=False)
    sun_abraham_figure(sa, OUT / "fig02_sunabraham.png")
    main_sa = sa[(sa["control_cutoff"] == SA_CUTOFFS[0]) & (sa["estimator"] == "sun_abraham") & (sa["term"] == "post")].iloc[0]
    add("")
    add(f"Sun-Abraham, control cohort published {SA_CUTOFFS[0]} or later, sample to {main_sa['period_last']}:")
    add(f"  post-publication vs the five years before it: {main_sa['coef']:+.4f} %/month (t = {main_sa['t']:.2f}), "
        f"{int(main_sa['n_treated_units'])} treated and {int(main_sa['n_control_units'])} control signals")
    table.to_csv(OUT / "result1_windows.csv")
    (OUT / "result1_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["in-sample", "post-sample\npre-publication", "post-publication"]
    ax.bar(labels, [m1, m2, m3], color=["#333333", "#777777", "#bbbbbb"], width=0.6)
    for i, v in enumerate([m1, m2, m3]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("mean long-short return, %/month")
    ax.set_title(f"Published edges after publication ({n} signals)", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig02_decay.png", dpi=160)
    print(f"\nwrote {OUT/'result1_windows.csv'}, {OUT/'result1_summary.txt'}, {OUT/'fig02_decay.png'}")


if __name__ == "__main__":
    main()
