"""
STEP 9 -- a quote-based spread for every CRSP stock-month, to price the trades in 07.

    python 09_spreads.py --check        # what the account holds: tables, columns, dates; pulls nothing
    python 09_spreads.py                # monthly spreads from crsp.dsf_v2, 1994 to 2025, one parquet a year
    python 09_spreads.py --first 2010   # restartable; years already on disk are skipped

The first pass of 07 charged each traded dollar half of JKP's bidaskhl_21d, a Corwin-Schultz
high-low estimate averaged over 21 days. Corwin-Schultz is noisy, overstates spreads for
liquid stocks, and its zero floor pushes the mean up; the traded-weighted spread it implied
was about 1% in a value-weighted book, which is too high after 2001 for stocks above the
NYSE 20th size percentile. The account's TAQ-based tables (contrib_liquidity_taq) cover
2001 to 2016 and carry no spread column, and raw TAQ is out of scope for this stage.

What is pulled instead: from crsp.dsf_v2, for each permno and calendar month,
  quoted    the mean over the month's days of the closing quoted spread
            (ask - bid) / midpoint, using days with ask > bid > 0 (Chung and Zhang 2014
            show the CRSP closing quote tracks TAQ effective spreads closely; it is an
            upper bound for liquid names, since trades print inside the quote)
  ar        the Abdi and Ranaldo (2017) estimate from daily high, low and close,
            sqrt(max(mean of 4 (c - eta)(c - eta_next), 0)), a second check that needs no quotes
  n_quoted, n_days   how many days went into each
The aggregation runs on the WRDS server, so only one row per stock-month comes down
(about 100 thousand rows a year). 07 charges half the quoted spread where it exists and
falls back to Corwin-Schultz where it does not, and reports the share of traded weight on
each source. Files: data/spreads/spreads_<year>.parquet, never committed.

--check lists the liquidity-related libraries and tables, searches the raw-TAQ library
names for the WRDS Intraday Indicators, and lists the quote columns of crsp.dsf_v2, into
data/liquidity_schema.txt. Nothing here is firm-level output except the parquet files,
which stay on this machine.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

import time

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
SPREADS = DATA / "spreads"
SCHEMA_FILE = DATA / "liquidity_schema.txt"
FIRST_YEAR, LAST_YEAR = 1994, 2025


def describe_table(db, lib: str, table: str, lines: list) -> None:
    cols = db.raw_sql(f"""
        SELECT column_name, data_type FROM information_schema.columns
        WHERE table_schema = '{lib}' AND table_name = '{table}' ORDER BY ordinal_position
    """)
    lines.append(f"\n{lib}.{table}: {len(cols)} columns")
    for r in cols.itertuples():
        lines.append(f"    {r.column_name:<28} {r.data_type}")
    date_cols = [c for c, t in zip(cols["column_name"], cols["data_type"])
                 if "date" in t or c.lower() in ("date", "month", "yyyymm", "eom", "mthcaldt", "dlycaldt")]
    try:
        if date_cols:
            d = date_cols[0]
            r = db.raw_sql(f"SELECT count(*) AS n, min({d}) AS first, max({d}) AS last FROM {lib}.{table}")
            lines.append(f"    rows {int(r['n'].iloc[0]):,}; {d} from {r['first'].iloc[0]} to {r['last'].iloc[0]}")
        else:
            r = db.raw_sql(f"SELECT count(*) AS n FROM {lib}.{table}")
            lines.append(f"    rows {int(r['n'].iloc[0]):,}")
    except Exception as e:                       # a huge table or a permission limit
        lines.append(f"    (count skipped: {type(e).__name__})")


def check(db) -> None:
    lines = []
    libs = db.list_libraries()
    hits = sorted(l for l in libs if any(k in l.lower() for k in ("liq", "taq", "iid", "intraday", "spread")))
    lines.append("libraries whose name mentions liquidity, TAQ, intraday or spreads:")
    for l in hits:
        lines.append(f"    {l}")
    for lib in hits:
        if lib.startswith("taqm_") or lib == "taqmsec":          # raw TAQ: list names only
            try:
                tables = db.list_tables(lib)
                lines.append(f"\n{lib}: {len(tables)} tables (raw TAQ, names only), e.g. {', '.join(tables[:5])}")
            except Exception as e:
                lines.append(f"\n{lib}: could not list ({type(e).__name__})")
            continue
        try:
            tables = db.list_tables(lib)
        except Exception as e:
            lines.append(f"\n{lib}: could not list ({type(e).__name__})")
            continue
        lines.append(f"\n=== {lib}: {len(tables)} tables ===")
        for t in tables[:40]:
            describe_table(db, lib, t, lines)
    # the WRDS Intraday Indicators, if the account has them, sit among the raw-TAQ table names
    for lib in ("taqmsec", "taqm_2010", "taqm_2020"):
        if lib in libs:
            try:
                names = db.list_tables(lib)
                hits_iid = [t for t in names if any(k in t.lower() for k in ("iid", "indicator", "wrds_"))]
                lines.append(f"\n{lib}: {len(hits_iid)} table names mentioning iid, indicator or wrds_"
                             + (": " + ", ".join(hits_iid[:12]) if hits_iid else ""))
                if hits_iid:
                    describe_table(db, lib, hits_iid[-1], lines)
            except Exception as e:
                lines.append(f"\n{lib}: could not search ({type(e).__name__})")
    # CRSP daily quote columns, for an Abdi-Ranaldo or closing-quote spread as the fallback
    cols = db.raw_sql("""
        SELECT column_name, data_type FROM information_schema.columns
        WHERE table_schema = 'crsp' AND table_name = 'dsf_v2' ORDER BY ordinal_position
    """)
    quote = [c for c in cols["column_name"] if any(k in c.lower() for k in ("bid", "ask", "high", "low", "open", "close", "prc", "vol"))]
    lines.append("\ncrsp.dsf_v2 price and quote columns: " + ", ".join(quote))
    DATA.mkdir(exist_ok=True)
    SCHEMA_FILE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {SCHEMA_FILE}")


def pull_year(db, year: int) -> None:
    out = SPREADS / f"spreads_{year}.parquet"
    if out.exists():
        print(f"  {year}  cached, skipping")
        return
    t0 = time.time()
    df = db.raw_sql(f"""
        WITH d AS (
            SELECT permno, dlycaldt, dlybid, dlyask, dlyhigh, dlylow,
                   CASE WHEN dlyclose > 0 THEN ln(dlyclose) END AS c,
                   CASE WHEN dlyhigh > 0 AND dlylow > 0 THEN (ln(dlyhigh) + ln(dlylow)) / 2 END AS eta,
                   lead(CASE WHEN dlyhigh > 0 AND dlylow > 0 THEN (ln(dlyhigh) + ln(dlylow)) / 2 END)
                       OVER (PARTITION BY permno ORDER BY dlycaldt) AS eta_next
            FROM crsp.dsf_v2
            WHERE dlycaldt BETWEEN '{year}-01-01' AND '{year}-12-31'
        )
        SELECT permno,
               date_trunc('month', dlycaldt)::date AS month,
               avg(CASE WHEN dlyask > dlybid AND dlybid > 0
                        THEN (dlyask - dlybid) / ((dlyask + dlybid) / 2) END) AS quoted,
               count(CASE WHEN dlyask > dlybid AND dlybid > 0 THEN 1 END) AS n_quoted,
               avg(CASE WHEN dlyhigh > dlylow THEN 4 * (c - eta) * (c - eta_next) END) AS ar_var,
               count(*) AS n_days
        FROM d
        GROUP BY 1, 2
    """, date_cols=["month"])
    if df.empty:
        print(f"  {year}  no rows")
        return
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    for c in ("quoted", "ar_var"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df["ar"] = np.sqrt(np.clip(df["ar_var"].to_numpy(dtype=float), 0, None)).astype(np.float32)
    df = df.drop(columns="ar_var")
    for c in ("n_quoted", "n_days"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")
    SPREADS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    q = df["quoted"]
    print(f"  {year}  {len(df):>8,} stock-months   quoted present {q.notna().mean():.3f}, "
          f"median {q.median():.4f}   ar median {df['ar'].median():.4f}   {time.time()-t0:5.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="list the liquidity tables and their columns")
    ap.add_argument("--first", type=int, default=FIRST_YEAR)
    ap.add_argument("--last", type=int, default=LAST_YEAR)
    args = ap.parse_args()
    import wrds
    user = os.environ.get("WRDS_USERNAME")
    if not user:
        sys.exit("set WRDS_USERNAME to your WRDS login before running this script")
    db = wrds.Connection(wrds_username=user)
    try:
        if args.check:
            check(db)
            return
        print(f"monthly quoted and Abdi-Ranaldo spreads from crsp.dsf_v2, {args.first} to {args.last}\n")
        for year in range(args.first, args.last + 1):
            pull_year(db, year)
        files = sorted(SPREADS.glob("spreads_*.parquet"))
        print(f"\n{len(files)} year files in {SPREADS}, {sum(f.stat().st_size for f in files)/1e6:,.0f} MB")
    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
