"""
STEP 6 -- pull the US slice of the Jensen, Kelly and Pedersen (2023) stock-month panel from
WRDS (contrib.global_factor), one year at a time, and cache it.

    source ~/venvs/wrds/bin/activate
    cd ~/code/alpha-decay
    python 06_jkp_pull.py --check     # columns, row counts, a one-month profile; pulls nothing big
    python 06_jkp_pull.py             # every year not already cached, 1963 onward

Writes:
    data/jkp_columns.txt                       every column the table has, with its type
    data/jkp_profile_2010m01.csv               per-column summary of one month: dtype,
                                               non-missing share, mean, sd, quantiles.
                                               No stock-level rows; this file may leave
                                               the machine and synthetic test data is
                                               built from it
    data/jkp_us/jkp_us_<year>.parquet          one file per year, float32, restartable

data/ is gitignored. Nothing downloaded here is redistributable, and the stock-level
files never leave this machine; only aggregates (tables, figures) do.

==============================================================================================
WHAT THIS TABLE IS, AND WHY THE QUERY LOOKS LIKE THIS

1. TABLE.  contrib.global_factor is the stock-month panel behind "Is There a Replication
   Crisis in Finance?" (Jensen, Kelly and Pedersen, Journal of Finance 2023): returns plus
   153 firm characteristics, built from CRSP and Compustat by their published code. The
   characteristic names are the ones in jkp_characteristics.csv next to this script.

2. THE SCREENS.  The authors' recommended sample is
       excntry = 'USA'          listed in the United States
       common = 1               ordinary common stock
       exch_main = 1            NYSE, AMEX or Nasdaq
       primary_sec = 1          the firm's primary security
       obs_main = 1             one observation per stock-month
   Applied in SQL, on the server.

3. THE TARGET.  ret_exc_lead1m is next month's return in excess of the risk-free rate,
   aligned by the authors so that a row holds the characteristics known at the end of
   month t and the return earned over month t+1. Using it avoids an off-by-one of our own.

4. WHAT ELSE COMES ALONG.  size_grp (mega, large, small, micro, nano by NYSE breakpoints),
   me (market equity), prc, dolvol_126d and bidaskhl_21d (the Corwin-Schultz spread
   estimate), which the portfolio tables and the cost model use.

5. WHY ONE FILE PER YEAR, FLOAT32.  About 170 columns and up to 7,500 stocks a month; a
   year is a few tens of megabytes in float32 parquet, and a rerun skips what is on disk.
==============================================================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
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
CHUNKS = DATA / "jkp_us"
CHARS_FILE = HERE / "jkp_characteristics.csv"

FIRST_YEAR = 1963
LAST_YEAR = 2025
TABLE = "contrib.global_factor"

ID_COLUMNS = ["id", "eom", "permno", "gvkey", "size_grp", "me", "prc", "ret_exc",
              "ret_exc_lead1m", "market_equity", "dolvol_126d", "bidaskhl_21d"]

SCREEN = """
        excntry     = 'USA'
    AND common      = 1
    AND exch_main   = 1
    AND primary_sec = 1
    AND obs_main    = 1
"""


def characteristic_names() -> list[str]:
    chars = pd.read_csv(CHARS_FILE)["characteristic"].tolist()
    if len(chars) != 153:
        sys.exit(f"{CHARS_FILE} should list 153 characteristics, found {len(chars)}")
    return chars


def table_columns(db) -> pd.DataFrame:
    schema, name = TABLE.split(".")
    cols = db.raw_sql(f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = '{name}'
        ORDER BY ordinal_position
    """)
    if cols.empty:
        sys.exit(f"{TABLE} has no columns visible to this account; is the library licensed?")
    return cols


def select_list(available: set[str]) -> tuple[list[str], list[str]]:
    """The columns to pull, and the ones we wanted that the table does not have."""
    wanted = list(dict.fromkeys(ID_COLUMNS + characteristic_names()))   # unique, in order
    assert len(wanted) == len(set(wanted))
    have = [c for c in wanted if c in available]
    missing = [c for c in wanted if c not in available]
    return have, missing


def check(db) -> None:
    cols = table_columns(db)
    DATA.mkdir(exist_ok=True)
    (DATA / "jkp_columns.txt").write_text(
        "\n".join(f"{r.column_name:<28} {r.data_type}" for r in cols.itertuples()) + "\n")
    print(f"{TABLE}: {len(cols)} columns, list written to data/jkp_columns.txt")

    have, missing = select_list(set(cols["column_name"]))
    overlap = sorted(set(ID_COLUMNS) & set(characteristic_names()))
    print(f"  requested {len(have) + len(missing)} distinct columns, present {len(have)}, "
          f"missing {len(missing)}; {len(overlap)} serve as both identifier and characteristic: "
          f"{', '.join(overlap)}")
    if missing:
        print("  missing:", ", ".join(missing))

    print("\nUS rows per year under the recommended screens (every fifth year):")
    counts = db.raw_sql(f"""
        SELECT EXTRACT(year FROM eom)::int AS year, count(*) AS n,
               count(DISTINCT id) AS stocks
        FROM {TABLE}
        WHERE {SCREEN}
          AND mod(EXTRACT(year FROM eom)::int, 5) = 0
        GROUP BY 1 ORDER BY 1
    """)
    for r in counts.itertuples():
        print(f"  {r.year}  {int(r.n):>8,} rows  {int(r.stocks):>6,} stocks")
    last = db.raw_sql(f"SELECT max(eom) AS last FROM {TABLE} WHERE {SCREEN}")["last"].iloc[0]
    print(f"\n  last month in the US panel: {last}")

    print("\nProfiling one month (2010-01) column by column ...")
    q = f"""
        SELECT {", ".join(have)}
        FROM {TABLE}
        WHERE {SCREEN} AND eom = '2010-01-31'
    """
    df = db.raw_sql(q, date_cols=["eom"])
    prof = profile(df)
    out = DATA / "jkp_profile_2010m01.csv"
    prof.to_csv(out)
    print(f"  {len(df):,} stocks in the month, {len(prof)} columns profiled -> {out.name}")
    print("  (the profile holds no stock-level values and is safe to share)")


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, non-missing share, mean, sd and five quantiles."""
    assert df.columns.is_unique, "duplicate column names in the pulled frame"
    rows = {}
    n = len(df)
    for c in df.columns:
        s = df[c]
        row = {"dtype": str(s.dtype), "non_missing": float(s.notna().mean()), "n": n}
        if c in ("id", "permno", "gvkey", "eom"):
            row["distinct"] = int(s.nunique())
        elif c == "size_grp":
            for k, v in s.value_counts(normalize=True).items():
                row[f"share_{k}"] = float(v)
        else:
            x = pd.to_numeric(s, errors="coerce").astype(float)
            row.update({"mean": x.mean(), "sd": x.std(),
                        "q01": x.quantile(0.01), "q25": x.quantile(0.25), "q50": x.quantile(0.5),
                        "q75": x.quantile(0.75), "q99": x.quantile(0.99)})
        rows[c] = row
    return pd.DataFrame(rows).T.rename_axis("column")


def pull_year(db, year: int, cols: list[str]) -> None:
    out = CHUNKS / f"jkp_us_{year}.parquet"
    if out.exists():
        print(f"  {year}  cached, skipping")
        return
    q = f"""
        SELECT {", ".join(cols)}
        FROM {TABLE}
        WHERE {SCREEN}
          AND eom BETWEEN '{year}-01-01' AND '{year}-12-31'
    """
    t0 = time.time()
    df = db.raw_sql(q, date_cols=["eom"])
    if df.empty:
        print(f"  {year}  no rows")
        return
    for c in df.columns:
        if c in ("id", "permno", "gvkey"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        elif c not in ("eom", "size_grp"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  {year}  {len(df):>8,} rows   {out.stat().st_size/1e6:6.1f} MB   {time.time()-t0:5.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="columns, counts and a one-month profile")
    ap.add_argument("--first", type=int, default=FIRST_YEAR)
    ap.add_argument("--last", type=int, default=LAST_YEAR)
    args = ap.parse_args()

    # The username comes from the WRDS_USERNAME environment variable and the password
    # from ~/.pgpass, which the wrds package offers to write on first use. Neither is
    # stored by this script.
    user = os.environ.get("WRDS_USERNAME")
    if not user:
        sys.exit("set WRDS_USERNAME to your WRDS login before running this script")
    db = wrds.Connection(wrds_username=user)
    try:
        if args.check:
            check(db)
            return
        have, missing = select_list(set(table_columns(db)["column_name"]))
        if missing:
            print("not in the table, skipped:", ", ".join(missing))
        print(f"Pulling {len(have)} columns, {args.first} to {args.last}\n")
        CHUNKS.mkdir(parents=True, exist_ok=True)
        for year in range(args.first, args.last + 1):
            pull_year(db, year, have)
    finally:
        try:
            db.close()
        except Exception:
            pass

    files = sorted(CHUNKS.glob("jkp_us_*.parquet"))
    total = sum(f.stat().st_size for f in files) / 1e6
    print(f"\n{len(files)} year files on disk, {total:,.0f} MB in total")


if __name__ == "__main__":
    main()
