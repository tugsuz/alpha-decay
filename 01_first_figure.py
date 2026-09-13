"""
alpha-decay, step 1: look at one anomaly, and mark the moment it became public.

Two stages, on purpose. Run `inspect` first and READ the output -- you cannot make the
decisions in DECISIONS.md without having seen the actual columns and date ranges.

    python 01_first_figure.py inspect
    python 01_first_figure.py figure

No WRDS needed. The Chen-Zimmermann data is free and public.

--------------------------------------------------------------------------------------
BEFORE RUNNING: download two files from https://www.openassetpricing.com  (Download tab)
and put them in a folder called `data/` next to this script:

    data/PredictorLSretWide.csv   monthly long-short return, one column per signal
    data/SignalDoc.csv            one row per signal: authors, journal, year, sample dates

If the file names differ from these (they change between data releases), just edit
PORTFOLIO_FILE and SIGNALDOC_FILE below. `inspect` will tell you what it actually found.
--------------------------------------------------------------------------------------
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")            # write a PNG instead of opening a window
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "output"

PORTFOLIO_FILE = DATA / "PredictorLSretWide.csv"
SIGNALDOC_FILE = DATA / "SignalDoc.csv"

# ---------------------------------------------------------------------------
# CONFIG -- these column names are VERIFIED against SignalDoc.csv in the
# OpenSourceAP/CrossSection repository (October 2025 release). SIGNAL is the one
# you choose. Run `inspect` first and pick one you can defend.
# ---------------------------------------------------------------------------
SIGNAL = "Mom12m"              # a column in the wide returns file
DATE_COL = "date"              # date column in the wide returns file

DOC_NAME_COL   = "Acronym"          # matches the returns column name
DOC_YEAR_COL   = "Year"             # publication year
DOC_SAMPEND    = "SampleEndYear"    # end of the ORIGINAL paper sample
DOC_QUALITY    = "Signal Rep Quality"   # 1_good, 2_fair, ...
DOC_CATEGORY   = "Cat.Signal"           # Predictor / Placebo / ...


def load_returns() -> pd.DataFrame:
    if not PORTFOLIO_FILE.exists():
        sys.exit(f"Not found: {PORTFOLIO_FILE}\nSee the download note at the top of this file.")
    df = pd.read_csv(PORTFOLIO_FILE)
    # find the date column whatever it is called, and parse it
    date_col = DATE_COL if DATE_COL in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    return df.rename(columns={date_col: "date"})


def load_doc() -> pd.DataFrame:
    if not SIGNALDOC_FILE.exists():
        sys.exit(f"Not found: {SIGNALDOC_FILE}\nSee the download note at the top of this file.")
    return pd.read_csv(SIGNALDOC_FILE)


def inspect() -> None:
    """Print enough of the raw data that you can answer DECISIONS.md questions 1 and 2."""
    rets = load_returns()
    doc = load_doc()

    print("=" * 78)
    print("RETURNS FILE")
    print("=" * 78)
    print(f"shape          : {rets.shape[0]:,} rows x {rets.shape[1]:,} columns")
    print(f"date range     : {rets['date'].min():%Y-%m} to {rets['date'].max():%Y-%m}")
    print(f"first 12 signal columns: {list(rets.columns[1:13])}")
    print("\nhead:")
    print(rets.iloc[:5, :6].to_string(index=False))

    print("\n" + "=" * 78)
    print("SIGNALDOC FILE")
    print("=" * 78)
    print(f"shape   : {doc.shape[0]:,} signals x {doc.shape[1]} fields")
    print(f"columns : {list(doc.columns)}")
    print("\nhead:")
    print(doc.head(5).to_string(index=False))

    # How many signals do we have BOTH returns and a publication year for?
    if DOC_NAME_COL in doc.columns and DOC_YEAR_COL in doc.columns:
        have_rets = set(rets.columns) - {"date"}
        matched = doc[doc[DOC_NAME_COL].isin(have_rets)]
        print("\n" + "-" * 78)
        print(f"signals with returns          : {len(have_rets):,}")
        print(f"...also present in SignalDoc  : {len(matched):,}")
        print(f"publication years             : {matched[DOC_YEAR_COL].min():.0f}"
              f" to {matched[DOC_YEAR_COL].max():.0f}")
        print("\npublication years by decade:")
        print((matched[DOC_YEAR_COL] // 10 * 10).value_counts().sort_index().to_string())
    else:
        print(f"\n!! '{DOC_NAME_COL}' or '{DOC_YEAR_COL}' not in SignalDoc columns above.")
        print("   Edit DOC_NAME_COL / DOC_YEAR_COL at the top of this script and rerun.")

    print("\nNow open DECISIONS.md and answer questions 1 and 2.")


def figure() -> None:
    """One anomaly, cumulative long-short return, with the publication year marked."""
    rets = load_returns()
    doc = load_doc()

    if SIGNAL not in rets.columns:
        sys.exit(f"'{SIGNAL}' is not a column in the returns file. "
                 f"Run `inspect` and pick one, then edit SIGNAL at the top.")

    row = doc.loc[doc[DOC_NAME_COL] == SIGNAL]
    if row.empty:
        sys.exit(f"'{SIGNAL}' not found in SignalDoc under column '{DOC_NAME_COL}'.")
    pub_year = int(row[DOC_YEAR_COL].iloc[0])

    s = rets[["date", SIGNAL]].dropna().sort_values("date")

    # Returns are in percent in this dataset. Compounding a percentage as if it were a
    # decimal is the classic first bug -- check the magnitudes in `inspect` output and
    # change this if needed.
    r = s[SIGNAL] / 100.0
    s = s.assign(cum=(1.0 + r).cumprod())

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(s["date"], s["cum"], linewidth=1.4, color="black")
    ax.axvline(pd.Timestamp(f"{pub_year}-01-01"), color="crimson",
               linestyle="--", linewidth=1.2)
    ax.text(pd.Timestamp(f"{pub_year}-01-01"), ax.get_ylim()[1] * 0.97,
            f"  published {pub_year}", color="crimson", va="top", fontsize=9)

    ax.set_yscale("log")          # a decaying compound series is unreadable on a linear axis
    ax.set_ylabel("cumulative long-short return (log scale, $1 at start)")
    ax.set_xlabel("")
    ax.set_title(f"{SIGNAL}: does the edge survive publication?", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = OUT / f"fig01_{SIGNAL}.png"
    fig.savefig(path, dpi=160)

    # McLean and Pontiff split the timeline in THREE, not two. The middle period is
    # the paper's own sample end -> publication: the pattern is discovered but not yet
    # public. If decay is about publication, period 2 should look like period 1.
    # If it is about the original sample being lucky, period 2 should already be lower.
    samp_end = row[DOC_SAMPEND].iloc[0]
    have_samp = pd.notna(samp_end) and samp_end < pub_year
    if have_samp:
        samp_end = int(samp_end)
        p1 = s.loc[s["date"] <= f"{samp_end}-12-31", SIGNAL].mean()
        p2 = s.loc[(s["date"] > f"{samp_end}-12-31") &
                   (s["date"] < f"{pub_year}-01-01"), SIGNAL].mean()
    else:
        p1 = s.loc[s["date"] < f"{pub_year}-01-01", SIGNAL].mean()
        p2 = float("nan")
    p3 = s.loc[s["date"] >= f"{pub_year}-01-01", SIGNAL].mean()

    print(f"saved {path}")
    print(f"\n{SIGNAL}  (published {pub_year}"
          + (f", original sample ends {samp_end}" if have_samp else "") + ")")
    print("mean monthly long-short return, %/month")
    print(f"  1. in the paper's own sample      : {p1:6.3f}")
    if have_samp:
        print(f"  2. after the sample, pre-publication: {p2:6.3f}"
              f"   ({100*(p2-p1)/abs(p1):+.0f}% vs period 1)")
    print(f"  3. after publication              : {p3:6.3f}"
          f"   ({100*(p3-p1)/abs(p1):+.0f}% vs period 1)")
    print("\nMcLean and Pontiff find roughly -26% for period 2 and -58% for period 3,")
    print("averaged over 97 anomalies. One signal is not 97 -- but this is the shape.")
    print("\nNow answer DECISIONS.md questions 1 and 2 before you believe any of it.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    {"inspect": inspect, "figure": figure}.get(cmd, inspect)()
