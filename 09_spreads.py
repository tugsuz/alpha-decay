"""
STEP 9 -- an independent spread series to check the Corwin-Schultz costs against.

    python 09_spreads.py --check        # what the account holds: tables, columns, dates; pulls nothing

The portfolio costs in 07 charge each traded dollar half of JKP's bidaskhl_21d, a
Corwin-Schultz high-low estimate averaged over 21 days. Corwin-Schultz is noisy, overstates
spreads for liquid stocks, and its zero floor pushes the mean up. Before a net Sharpe ratio
is reported the spread behind it has to be checked against a series built from quotes.

--check lists what is available for that: the TAQ-based liquidity tables in
contrib_liquidity_taq and any library whose name mentions liquidity, TAQ or intraday
indicators, with their columns, row counts and date ranges, and the quote columns of
crsp.dsf_v2 (daily bid and ask, high and low, for an Abdi-Ranaldo estimate). The result
goes to data/liquidity_schema.txt. The pull itself is written once the schema is known.

Nothing here is firm-level output; the schema file holds table and column names only.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
SCHEMA_FILE = DATA / "liquidity_schema.txt"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="list the liquidity tables and their columns")
    args = ap.parse_args()
    if not args.check:
        sys.exit("only --check is implemented; the pull is written once the schema is known")
    import wrds
    user = os.environ.get("WRDS_USERNAME")
    if not user:
        sys.exit("set WRDS_USERNAME to your WRDS login before running this script")
    db = wrds.Connection(wrds_username=user)
    try:
        check(db)
    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
