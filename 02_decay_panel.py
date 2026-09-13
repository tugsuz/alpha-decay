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

    Written out in numpy rather than called from a library: the cluster-robust sandwich
    is four lines, and the point of the exercise is to know what it does.
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
