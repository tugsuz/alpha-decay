"""
STEP 3 -- pull the CRSP monthly panel from WRDS, in the CIZ format, and cache it.

    source ~/venvs/wrds/bin/activate
    cd ~/code/alpha-decay
    python 03_crsp_pull.py            # pull everything not already cached
    python 03_crsp_pull.py --check    # connect, print coverage, pull nothing

Writes:
    data/crsp_monthly/crsp_monthly_<decade>.parquet     one file per decade, restartable
    data/crsp_monthly.parquet                           the concatenated panel
    data/ff_factors_monthly.parquet                     Fama-French, for risk-adjusted alpha

data/ is gitignored. Nothing downloaded here is redistributable.

==============================================================================================
WHY THE QUERY LOOKS LIKE THIS -- read once, then you can defend every line of it

1. TABLE.  crsp.msf_v2 is the WRDS view over the CIZ core table stkMthSecurityData, already
   joined to the security-information history. The legacy crsp.msf still exists but CRSP's
   last legacy (SIZ) release was February 2025, so it is frozen. msf_v2 runs a year further.

2. DELISTING RETURNS ARE ALREADY IN mthret.  In the old SIZ world the single most common
   error in anomaly code was forgetting to merge crsp.msedelist and splice the delisting
   return onto the last month, which biases returns upward because delistings are mostly
   bad news. In CIZ that merge is gone: CRSP's own crosswalk table maps
       SFZ_MDEL.MDLRET  ->  StkMthSecurityData.MthRet   (Recalc / DelistConv)
   i.e. the delisting return is folded into the monthly total return. Do NOT merge
   stkdelists on top of mthret. You would double-count.

3. THE FILTERS.  In SIZ you wrote shrcd in (10,11) and exchcd in (1,2,3). CIZ splits that
   one code into five columns, so the equivalent US-common-stock screen is:
       sharetype        = 'NS'            not an ADR, not an SBI, not a unit
       securitytype     = 'EQTY'          equity, not a fund
       securitysubtype  = 'COM'           common, not an ETF or ETN
       usincflg         = 'Y'             US incorporated
       issuertype       in ('ACOR','CORP')
       primaryexch      in ('N','A','Q')  NYSE, AMEX, Nasdaq
       conditionaltype  = 'RW'            regular way
       tradingstatusflg = 'A'             active
   These are applied IN SQL, on the server. About 5.2 million rows exist; you download
   roughly two thirds of that once and never again.

4. mthcap IS THE MARKET CAP.  Do not compute abs(prc) * shrout yourself. CRSP's price
   column is negative when it is a bid-ask average rather than a trade, and mthcap already
   handles that, plus the share-adjustment factors.

5. WHY PARQUET, BY DECADE.  A single 3-million-row query over a slow link is one thing that
   can fail at minute nine. Decade chunks are restartable: rerun the script and it skips
   what is already on disk. Parquet keeps dtypes and is ~10x smaller than CSV.
==============================================================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import wrds
except ModuleNotFoundError:
    sys.exit(
        "The 'wrds' package is not installed in THIS interpreter.\n"
        f"Interpreter in use: {sys.executable}\n"
        "Fix:  source ~/venvs/wrds/bin/activate"
    )

HERE = Path(__file__).parent
DATA = HERE / "data"
CHUNKS = DATA / "crsp_monthly"
PANEL = DATA / "crsp_monthly.parquet"
FFOUT = DATA / "ff_factors_monthly.parquet"

DECADES = [(y, y + 9) for y in range(1925, 2030, 10)]

COLUMNS = """
    permno, permco, mthcaldt, mthret, mthretx, mthprc, mthcap, mthvol,
    shrout, siccd, primaryexch, ticker, mthdelflg
"""

SCREEN = """
        sharetype        = 'NS'
    AND securitytype     = 'EQTY'
    AND securitysubtype  = 'COM'
    AND usincflg         = 'Y'
    AND issuertype       IN ('ACOR', 'CORP')
    AND primaryexch      IN ('N', 'A', 'Q')
    AND conditionaltype  = 'RW'
    AND tradingstatusflg = 'A'
"""


def coverage(db) -> None:
    """Which CRSP library is furthest forward? crsp is the annual update; crspq is quarterly."""
    print("\nCoverage check -- the last month present in each CRSP library:")
    for lib in ("crsp", "crspq", "crsp_a_stock", "crsp_q_stock"):
        try:
            df = db.raw_sql(f"SELECT max(mthcaldt) AS last_month FROM {lib}.msf_v2")
            print(f"  {lib+'.msf_v2':<22} {df['last_month'].iloc[0]}")
        except Exception as e:
            print(f"  {lib+'.msf_v2':<22} unavailable ({type(e).__name__})")
    print("\nChen-Zimmermann portfolio data ends 2024-12-31. Anything later than that is")
    print("sample CRSP buys you that the open-source file cannot.\n")


def pull_decade(db, lib: str, y0: int, y1: int) -> Path:
    out = CHUNKS / f"crsp_monthly_{y0}_{y1}.parquet"
    if out.exists():
        print(f"  {y0}-{y1}  cached, skipping")
        return out
    q = f"""
        SELECT {COLUMNS}
        FROM {lib}.msf_v2
        WHERE mthcaldt BETWEEN '{y0}-01-01' AND '{y1}-12-31'
          AND {SCREEN}
    """
    t0 = time.time()
    df = db.raw_sql(q, date_cols=["mthcaldt"])
    if df.empty:
        print(f"  {y0}-{y1}  no rows")
        return out
    df["permno"] = df["permno"].astype("int32")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  {y0}-{y1}  {len(df):>9,} rows   {time.time()-t0:5.1f}s   -> {out.name}")
    return out


def pull_ff(db) -> None:
    print("\nFama-French monthly factors ...")
    df = db.raw_sql(
        "SELECT date, mktrf, smb, hml, rf, umd FROM ff.factors_monthly ORDER BY date",
        date_cols=["date"],
    )
    df.to_parquet(FFOUT, index=False)
    print(f"  {len(df):,} months, {df['date'].min():%Y-%m} to {df['date'].max():%Y-%m} -> {FFOUT.name}")


def assemble() -> None:
    files = sorted(CHUNKS.glob("crsp_monthly_*.parquet"))
    if not files:
        print("No chunks on disk, nothing to assemble.")
        return
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df = df.sort_values(["permno", "mthcaldt"]).reset_index(drop=True)
    df.to_parquet(PANEL, index=False)

    print("\n" + "=" * 70)
    print("CRSP MONTHLY PANEL")
    print("=" * 70)
    print(f"  rows                {len(df):,}")
    print(f"  unique permnos      {df['permno'].nunique():,}")
    print(f"  months              {df['mthcaldt'].min():%Y-%m} to {df['mthcaldt'].max():%Y-%m}")
    print(f"  mthret missing      {df['mthret'].isna().mean():.2%}")
    print(f"  file                {PANEL}  ({PANEL.stat().st_size/1e6:.0f} MB)")

    # Two sanity checks that would catch a broken screen immediately.
    n_2000 = df.loc[df["mthcaldt"].dt.year == 2000, "permno"].nunique()
    print(f"\n  sanity: distinct common stocks in 2000 = {n_2000:,}")
    print("          (the literature's US common-stock universe is roughly 6,000-7,500 here;")
    print("           far more means the screen is too loose, far fewer means too tight)")
    ew = (df.dropna(subset=["mthret"])
            .groupby("mthcaldt")["mthret"].mean()
            .loc["1963":"2024"].mean() * 12)
    print(f"  sanity: equal-weighted mean annual return 1963-2024 = {ew:.1%}")
    print("          (should land in the low-to-mid teens; a wild number means the screen")
    print("           is letting funds or ADRs through)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="connect and report coverage, pull nothing")
    ap.add_argument("--lib", default=None, help="CRSP library to pull from (default: pick the freshest)")
    args = ap.parse_args()

    # Avoid re-typing the username every run: export WRDS_USERNAME=mtugsuz
    user = os.environ.get("WRDS_USERNAME")
    db = wrds.Connection(wrds_username=user) if user else wrds.Connection()
    try:
        coverage(db)
        if args.check:
            return

        lib = args.lib
        if lib is None:
            best, best_date = "crsp", None
            for cand in ("crspq", "crsp"):
                try:
                    d = db.raw_sql(f"SELECT max(mthcaldt) AS m FROM {cand}.msf_v2")["m"].iloc[0]
                    if best_date is None or d > best_date:
                        best, best_date = cand, d
                except Exception:
                    continue
            lib = best
        print(f"Pulling from {lib}.msf_v2\n")

        CHUNKS.mkdir(parents=True, exist_ok=True)
        for y0, y1 in DECADES:
            pull_decade(db, lib, y0, y1)
        pull_ff(db)
    finally:
        try:
            db.close()
        except Exception:
            pass

    assemble()


if __name__ == "__main__":
    main()
