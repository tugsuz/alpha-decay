"""
STEP 4 -- rebuild two Chen-Zimmermann anomalies from raw CRSP, then check them against
Chen-Zimmermann's own published series.

    source ~/venvs/wrds/bin/activate
    cd ~/code/alpha-decay
    python 04_build_anomalies.py

Needs (already on disk):
    data/crsp_monthly.parquet      from 03_crsp_pull.py
    data/PredictorLSretWide.csv    Chen-Zimmermann long-short returns
    data/SignalDoc.csv             Chen-Zimmermann implementation parameters

Writes:
    output/rebuild_<Signal>.csv    my monthly long-short series next to theirs
    output/rebuild_summary.txt     the verdict, copy-pasteable
    output/fig04_rebuild.png       cumulative long-short, mine vs theirs

==============================================================================================
WHY THIS SCRIPT EXISTS, AND WHY IT IS THE HONEST VERSION OF "I HAVE TOUCHED CRSP"

Anyone can download a file of pre-computed anomaly returns. The claim worth making is that
you can rebuild one from raw stock data and land on the same number. This script does that,
and then it checks itself against the published series rather than asserting a match.

The check is the point. If the correlation is high, the pipeline is validated and Result 3
can be built on it. If it is low, that is a finding about implementation choices, which is
a better interview answer than a clean match.

THE PARAMETERS ARE NOT INVENTED. They are read out of SignalDoc.csv at runtime -- the sign,
the weighting, the quantile cut, the holding period. Chen-Zimmermann implement each signal
"following the original paper", and those choices are recorded in their own documentation.
Guessing them and then comparing would be comparing two different strategies.

FOUR DETAILS THAT ARE EASY TO GET WRONG, ALL OF THEM VISIBLE IN SignalDoc:

  1. Size has LS Quantile = 0.5, not 0.1. Banz (1981) splits the market in half, he does not
     sort deciles. Sorting deciles gives a bigger and wrong number.

  2. Size has Sign = -1. The anomaly is SMALL minus big. Getting the sign backwards produces
     a beautifully significant result with the wrong label on it.

  3. Mom12m skips month t. The definition reads "stock return between months t-12 and t-1",
     and t-1 is where it stops. Forming on t-11..t instead drags short-term reversal into the
     signal and cancels the premium outright: 0.791 %/month becomes -0.009. This one cost a
     debugging cycle, because correlation with the published series stayed at 0.974 the whole
     time. Correlation tells you the timing is right. It does not tell you the signal is.

  4. Portfolio Period is a holding period, with overlapping cohorts: the return earned in
     calendar month c is the average across the portfolios formed at c-1, c-2, ..., c-hold.
     Rebalancing monthly instead is a different strategy with a different turnover and a
     different mean.

WHAT THE RUN ACTUALLY FOUND, WHICH IS NOT WHAT THIS HEADER ORIGINALLY PREDICTED:

  Size reproduces at the documented holding period essentially exactly -- 0.324 against
  0.322 %/month, correlation 0.992 over 1,182 months.

  Mom12m does not. The documented hold=3 gives 0.652 against a published 0.900; plain monthly
  rebalancing gives 0.791 and is closer. So the script reports every candidate convention
  side by side and lets the level choose, rather than asserting that the documented one wins.
  The remaining -0.107 %/month is inside sampling error (t = -1.82) and is carried entirely by
  about sixty momentum-crash months. No screen variants are tried: searching filters until the
  number matches would be curve-fitting the replication itself.
==============================================================================================
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

PANEL = DATA / "crsp_monthly.parquet"
CZ_RET = DATA / "PredictorLSretWide.csv"
CZ_DOC = DATA / "SignalDoc.csv"

SIGNALS = ["Size", "Mom12m"]
MIN_PER_SIDE = 10          # do not let a bucket of 3 stocks into the average


# ----------------------------------------------------------------- signal construction
def add_month_index(df: pd.DataFrame) -> pd.DataFrame:
    df["mi"] = df["mthcaldt"].dt.year * 12 + df["mthcaldt"].dt.month
    return df


def ym(mi) -> tuple[int, int]:
    """Invert mi = year*12 + month back to (year, month).

    Month runs 1..12, so December has mi % 12 == 0 and mi // 12 == year + 1. Decoding
    with (mi // 12, mi % 12 or 12) is therefore right eleven months out of twelve and
    puts every December in the following year -- which is how an earlier version of this
    script reported the overlap window as ending 2025-12 when Chen-Zimmermann ends
    2024-12. divmod on mi - 1 has no such seam.
    """
    y, m = divmod(int(mi) - 1, 12)
    return y, m + 1


def signal_size(df: pd.DataFrame) -> pd.Series:
    """Log market value of equity. SignalDoc: 'Log of monthly market value of equity'.

    Uses mthcap rather than abs(prc)*shrout: CRSP's price is negative when it is a
    bid-ask average rather than a trade, and mthcap already handles the sign and the
    share-adjustment factors.
    """
    cap = pd.to_numeric(df["mthcap"], errors="coerce").astype("float64")
    return np.log(cap.where(cap > 0))


def signal_mom12m(df: pd.DataFrame, skip: int = 1) -> pd.Series:
    """SignalDoc: 'Stock return between months t-12 and t-1'.

    THE SKIP MONTH IS THE WHOLE GAME. Read the window literally: a portfolio formed at
    month t uses returns from t-12 through t-1, which EXCLUDES month t itself. That is
    the classic one-month skip, and the reason for it is short-term reversal: last
    month's return is negatively autocorrelated, largely from bid-ask bounce, so letting
    it into the signal does not merely add noise, it pulls in an effect of the opposite
    sign and cancels the momentum premium.

    First version of this script used months m-11..m, i.e. no skip. Result: the
    one-month-rebalanced strategy earned -0.009 %/month, t = -0.03. The premium was
    entirely gone. Correlation with Chen-Zimmermann stayed at 0.974 because the
    three-month overlapping construction meant two of the three cohorts skipped anyway,
    so the shape survived and only the level was halved. High correlation with a wrong
    mean is what a scaling or definition error looks like; a timing error would have
    shown up as the correlation peaking at a non-zero lag instead.

    skip = 0 reproduces that mistake on purpose, for the diagnostic printed below.

    Built from a cumulative sum of log returns. Two guards:
      - mthret is clipped just above -1, because a -100% month sends log1p to -inf and
        that -inf then poisons every later month of the same permno through the cumsum.
      - the contiguity check: a permno with a gap in its monthly record would otherwise
        silently get a window that spans the gap and is not twelve months long.
    """
    r = pd.to_numeric(df["mthret"], errors="coerce").astype("float64").clip(lower=-0.9999)
    cum = np.log1p(r).groupby(df["permno"]).cumsum()
    hi = cum.groupby(df["permno"]).shift(skip)            # cumulative through m-skip
    lo = cum.groupby(df["permno"]).shift(skip + 12)       # cumulative through m-skip-12
    mi_hi = df["mi"].groupby(df["permno"]).shift(skip)
    mi_lo = df["mi"].groupby(df["permno"]).shift(skip + 12)
    contiguous = ((mi_hi - mi_lo) == 12) & ((df["mi"] - mi_hi) == skip)
    return np.expm1(hi - lo).where(contiguous)


BUILDERS = {"Size": signal_size, "Mom12m": signal_mom12m}


# ----------------------------------------------------------------- portfolio machinery
def assign_buckets(g: pd.DataFrame, q: float, nyse_breaks: bool) -> pd.Series:
    """Cut one formation month's cross-section into quantile buckets.

    q = 0.1 -> deciles, q = 0.5 -> halves. Breakpoints come from the whole cross-section
    unless SignalDoc says NYSE, in which case they come from NYSE names only and are then
    applied to everybody -- which is what 'NYSE breakpoints' means and why it produces
    far fewer stocks in the extreme buckets than an all-stock cut.
    """
    s = g["signal"]
    n = int(round(1 / q))
    ref = s[g["primaryexch"] == "N"] if nyse_breaks else s
    if ref.notna().sum() < n * MIN_PER_SIDE:
        return pd.Series(np.nan, index=g.index)
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, n + 1)))
    if len(edges) < 3:
        return pd.Series(np.nan, index=g.index)
    return pd.Series(
        np.digitize(s, edges[1:-1], right=True).astype(float), index=g.index
    ).where(s.notna())


def long_short_series(df: pd.DataFrame, q: float, weight: str, sign: float,
                      hold: int, nyse_breaks: bool, exch_filter) -> pd.Series:
    """Return the strategy's monthly long-short return, in percent per month.

    One cohort is formed every month. Each cohort is held for `hold` months. The return
    booked in calendar month c is the average over the cohorts still alive in c, which for
    hold = 3 means the cohorts formed at c-1, c-2 and c-3. That averaging is the
    Jegadeesh-Titman overlapping-portfolio convention.
    """
    d = df
    if exch_filter is not None:
        d = d[d["primaryexch"].isin(exch_filter)]
    d = d[d["signal"].notna()].copy()

    n = int(round(1 / q))
    d["bucket"] = (d.groupby("mi", group_keys=False)[["signal", "primaryexch"]]
                     .apply(lambda g: assign_buckets(g, q, nyse_breaks)))
    d = d[d["bucket"].notna()]
    top, bot = float(n - 1), 0.0

    pieces = []
    for h in range(1, hold + 1):
        fwd = d.groupby("permno")["mthret"].shift(-h)
        fwd_mi = d.groupby("permno")["mi"].shift(-h)
        ok = ((fwd_mi - d["mi"]) == h) & fwd.notna()
        t = d.loc[ok, ["mi", "bucket", "mthcap"]].copy()
        t["r"] = fwd[ok]

        if weight.upper() == "VW":
            t["w"] = t["mthcap"].clip(lower=0).fillna(0)
            num = t.assign(x=t["r"] * t["w"]).groupby(["mi", "bucket"])["x"].sum()
            den = t.groupby(["mi", "bucket"])["w"].sum()
            port = (num / den.replace(0, np.nan)).unstack("bucket")
        else:
            port = t.groupby(["mi", "bucket"])["r"].mean().unstack("bucket")

        cnt = t.groupby(["mi", "bucket"])["r"].size().unstack("bucket")
        enough = (cnt.get(top, 0) >= MIN_PER_SIDE) & (cnt.get(bot, 0) >= MIN_PER_SIDE)
        ls = (port.get(top) - port.get(bot)).where(enough) * sign
        ls.index = ls.index + h                      # formation month -> calendar month
        pieces.append(ls.rename(f"h{h}"))

    out = pd.concat(pieces, axis=1).mean(axis=1, skipna=True) * 100.0
    out.index.name = "mi"
    return out.dropna()


# ----------------------------------------------------------------------------- compare
def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return np.nan if len(x) < 24 else x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def main() -> None:
    for p in (PANEL, CZ_RET, CZ_DOC):
        if not p.exists():
            sys.exit(f"Not found: {p}\nRun 03_crsp_pull.py first.")

    print("loading panel ...")
    df = pd.read_parquet(PANEL, columns=["permno", "mthcaldt", "mthret", "mthcap", "primaryexch"])
    df = df.dropna(subset=["mthret"]).sort_values(["permno", "mthcaldt"]).reset_index(drop=True)
    df = add_month_index(df)
    print(f"  {len(df):,} rows, {df['mthcaldt'].min():%Y-%m} to {df['mthcaldt'].max():%Y-%m}")

    doc = pd.read_csv(CZ_DOC).set_index("Acronym")
    cz = pd.read_csv(CZ_RET)
    cz["date"] = pd.to_datetime(cz[cz.columns[0]])
    cz["mi"] = cz["date"].dt.year * 12 + cz["date"].dt.month

    OUT.mkdir(exist_ok=True)
    lines, panels = [], {}
    add = lines.append
    add("REBUILDING CHEN-ZIMMERMANN ANOMALIES FROM RAW CRSP")
    add("=" * 78)
    add(f"CRSP panel: {df['mthcaldt'].min():%Y-%m} to {df['mthcaldt'].max():%Y-%m}, "
        f"{df['permno'].nunique():,} permnos")

    for name in SIGNALS:
        r = doc.loc[name]
        q = float(r["LS Quantile"])
        weight = str(r["Stock Weight"]).upper()
        sign = float(r["Sign"])
        hold = int(r["Portfolio Period"]) if pd.notna(r["Portfolio Period"]) else 1
        nyse = str(r.get("Quantile Filter")).upper() == "NYSE"
        filt = str(r.get("Filter"))
        exch = ["N", "A"] if "c(1,2)" in filt else (["N", "A", "Q"] if "c(1,2,3)" in filt else None)

        add("")
        add("-" * 78)
        add(f"{name}  --  {r['Authors']} ({int(r['Year'])}, {r['Journal']})")
        add(f"  SignalDoc parameters used: sign {sign:+.0f} | {weight} | "
            f"quantile {q} ({int(round(1/q))} buckets) | hold {hold} month(s) | "
            f"NYSE breakpoints {nyse} | exchange filter {exch}")
        add(f"  original paper reported: {r['Return']} %/month, t = {r['T-Stat']}, "
            f"sample {int(r['SampleStartYear'])}-{int(r['SampleEndYear'])}")

        df["signal"] = BUILDERS[name](df)
        mine = long_short_series(df, q, weight, sign, hold, nyse, exch)
        naive = long_short_series(df, q, weight, sign, 1, nyse, exch) if hold != 1 else mine

        noskip = None
        if name == "Mom12m":
            df["signal"] = signal_mom12m(df, skip=0)
            noskip = long_short_series(df, q, weight, sign, hold, nyse, exch)
            df["signal"] = BUILDERS[name](df)

        theirs = cz.set_index("mi")[name].dropna() if name in cz.columns else pd.Series(dtype=float)

        if theirs.empty:
            add("  Chen-Zimmermann column not found.")
        else:
            # Compare every candidate convention on the SAME overlap window and the same
            # yardsticks. This is calibration against a published benchmark, not a search
            # for the prettiest number: the conventions are the ones SignalDoc names plus
            # the two documented ways of misreading it, and all of them are reported.
            variants = [(f"documented   (hold={hold}, skip=1)", mine),
                        ("monthly      (hold=1, skip=1)", naive)]
            if noskip is not None:
                variants.append((f"no skip      (hold={hold}, skip=0)", noskip))

            base = pd.concat({"theirs": theirs}, axis=1)
            lo = hi = None
            add("")
            add(f"  Chen-Zimmermann : mean {theirs.mean():.3f} %/month, t = {tstat(theirs):.2f}"
                f"   ({len(theirs):,} months)")
            add("")
            add("  each candidate convention, on the common overlap window:")
            add("      convention                      mean    t     corr@0   best lag   gap vs CZ")
            scored = []
            for label, s in variants:
                o = base.join(s.rename("mine"), how="inner").dropna()
                if o.empty:
                    continue
                lo = o.index.min() if lo is None else min(lo, o.index.min())
                hi = o.index.max() if hi is None else max(hi, o.index.max())
                corrs = {lag: o["mine"].corr(o["theirs"].shift(lag)) for lag in (-2, -1, 0, 1, 2)}
                bl = max(corrs, key=lambda k: corrs[k])
                gap = o["mine"].mean() - o["theirs"].mean()
                add(f"      {label:<30} {o['mine'].mean():6.3f} {tstat(o['mine']):5.2f}"
                    f"  {corrs[0]:7.3f}   {bl:+d}        {gap:+.3f}")
                scored.append((abs(gap), corrs[0], bl, label))
            if lo is not None:
                add("      overlap window: %d-%02d to %d-%02d" % (*ym(lo), *ym(hi)))
            add("")
            if scored:
                scored.sort()
                gap, c0, bl, label = scored[0]
                add(f"  closest to the published series: {label.strip()}, "
                    f"gap {gap:.3f} %/month, corr {c0:.3f}.")
                if all(s[2] == 0 for s in scored):
                    add("  Every convention peaks at lag 0, so the formation-to-holding offset is right;")
                    add("  what separates them is the holding period, not the timing.")
                else:
                    add("  A convention peaking away from lag 0 has its formation-to-holding offset wrong.")

                spread = max(s[1] for s in scored) - min(s[1] for s in scored)
                if spread < 0.02:
                    add(f"  NOTE: the conventions' correlations differ by only {spread:.3f}. Every variant")
                    add("  is the same anomaly, so they are all ~0.97 correlated with each other and with")
                    add("  the published series. Correlation confirms the timing and rules out a wiring")
                    add("  error; it cannot choose between conventions. Only the level does that.")

                # WHERE is the residual gap? A replication that is exact in the modern era
                # and loose before 1963 is a coverage story, not a coding error: CRSP is
                # NYSE-only until July 1962 and thin before that, so the cross-section a
                # decile sort is cut from is a different object.
                best_label = label
                best_series = dict(variants)[best_label]
                o = base.join(best_series.rename("mine"), how="inner").dropna()
                o = o.assign(decade=[(ym(m)[0] // 10) * 10 for m in o.index])
                dec = o.groupby("decade").agg(
                    months=("mine", "size"), mine=("mine", "mean"), theirs=("theirs", "mean"))
                dec["gap"] = dec["mine"] - dec["theirs"]
                add("")
                add(f"  where the gap sits, decade by decade ({best_label.strip()}):")
                add("      decade   months    mine   theirs     gap")
                for d, r in dec.iterrows():
                    add(f"      {int(d)}s    {int(r['months']):5d}  {r['mine']:6.3f}  {r['theirs']:6.3f}  {r['gap']:+7.3f}")
                yr = np.array([ym(m)[0] for m in o.index])
                pre, post = o[yr < 1963], o[yr >= 1963]
                for tag, part in (("pre-1963 (CRSP is NYSE-only)", pre), ("1963 onward", post)):
                    if len(part) > 24:
                        add(f"      {tag:<32} n={len(part):4d}  mine {part['mine'].mean():6.3f}"
                            f"  theirs {part['theirs'].mean():6.3f}"
                            f"  gap {part['mine'].mean()-part['theirs'].mean():+.3f}")

        both = pd.concat({"mine": mine, "theirs": theirs, "mine_hold1": naive,
                          **({"mine_noskip": noskip} if noskip is not None else {})},
                         axis=1).dropna(subset=["mine"])
        both.to_csv(OUT / f"rebuild_{name}.csv")
        panels[name] = both

    (OUT / "rebuild_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, b) in zip(axes, panels.items()):
        ov = b.dropna(subset=["mine", "theirs"])
        if ov.empty:
            continue
        x = pd.to_datetime(["%d-%02d-01" % ym(m) for m in ov.index])
        ax.plot(x, (1 + ov["mine"] / 100).cumprod(), label="rebuilt from CRSP", lw=1.4, color="#333333")
        ax.plot(x, (1 + ov["theirs"] / 100).cumprod(), label="Chen-Zimmermann", lw=1.4,
                color="#bbbbbb", ls="--")
        ax.set_yscale("log")
        ax.set_title(name, loc="left")
        ax.set_ylabel("cumulative long-short, log scale")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig04_rebuild.png", dpi=160)
    print(f"\nwrote {OUT/'rebuild_summary.txt'}, rebuild_*.csv, {OUT/'fig04_rebuild.png'}")


if __name__ == "__main__":
    main()
