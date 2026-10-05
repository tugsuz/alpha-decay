"""
STEP 7 -- Stage 1: predict next month's stock return from the JKP characteristics, with and
without look-ahead in the choice of characteristics.

    source ~/venvs/wrds/bin/activate
    cd ~/code/alpha-decay
    python 07_stage1_models.py --synthetic --smoke      # a few minutes, no data needed
    python 07_stage1_models.py --synthetic --smoke --rebuild-panel   # after changing the panel code
    python 07_stage1_models.py --timing                 # one year, NN3 on cpu and on mps
    python 07_stage1_models.py --every 2                # first pass, refit every other year
    python 07_stage1_models.py                          # annual refit, 1995 to 2024
    python 07_stage1_models.py --eval-only              # rebuild the tables from saved forecasts
    python 07_stage1_models.py --y-check                # tails of the realised returns, by test year
    python 07_stage1_models.py --every 2 --robust       # clipped fitting target, Huber trees

Reads   data/jkp_us/jkp_us_<year>.parquet   (06_jkp_pull.py) and jkp_characteristics.csv.
Writes  data/stage1/preds_<year>.parquet    stock-level forecasts. These stay on this machine.
        output/stage1_*.csv                 aggregates only: R2, tests, portfolios, importance.
A --synthetic run writes to data/stage1_synthetic/ and output/synthetic/ instead, so made-up
forecasts never sit next to real ones. The tables are built from the test years of the
current run only; forecast files outside that range, or from a different panel, are listed
and left out.

--------------------------------------------------------------------------------------
THE DESIGN

Rows are stock-months. The target is next month's excess return (ret_exc_lead1m). Every
characteristic is ranked within its month across the whole screened universe and mapped
to [-1, 1], missing to 0; the ranks are computed before any size filter, so the features
do not depend on which portfolio is evaluated later.

Two information sets, fitted on the same rows with the same grids and seeds:
    public   only characteristics whose paper appeared before the test year
    full     all 153 characteristics, which is the usual setup and contains look-ahead
The gap between them is what a forecaster gains from knowing today which characteristics
would later be published.

Time is kept by the target month. A test year holds the forecasts whose target month is
in that year; training rows have target months up to December of the year before the
validation block, validation rows the eight years before the test year, and an assert
checks at every refit that no training or validation target month reaches the first test
month. Nothing is refit after validation: the model that is scored on the test year is
the one that was early-stopped or selected on the validation years (Gu, Kelly and Xiu 2020).

Models: OLS with a Huber loss, elastic net, principal-component regression, partial least
squares, gradient boosted trees, and feed-forward networks with one to three hidden layers
(32, 16, 8 units), each an average over five seeds.

Two worker processes. LightGBM links Homebrew's libomp and torch ships its own OpenMP
runtime; loaded into one process they deadlock on the first network. So the script runs
the tree-and-linear models in one child process and the networks in another, each
importing only what it needs, and merges the forecasts year by year. The ranked panel is
built once and cached in data/stage1/ so the second worker does not rank again.

The realised returns have a few absurd values (several thousand percent in a month). They
dominate squared-error sums wherever they fall: a test year with one such row has an R2 of
about zero for every model, and a training sample with a few of them pulls least-squares
fits and tree leaves towards those rows. --y-check tabulates them by year. --robust clips
the fitting and selection targets at the 0.1 and 99.9 percentiles of the training returns
and gives the trees a Huber objective; the Huber regression keeps the raw target, since its
loss is the protection. Forecasts are always scored against the raw realised returns, and
the R2 tables are reported on four row sets: all rows, without nano and micro caps, without
rows whose realised return is beyond +-100%, and rows with a CRSP return.

Runs. Every run gets an id (start time and commit) that goes into each forecast file and
each row of the per-year csvs, with a manifest in output/stage1_run.json (flags, test years,
panel, versions). The tables are built only from rows carrying the current id. --skip-done
continues the run in the manifest; without it a new run starts, the previous run's tables
are copied to output/runs/<id>/, and its rows are dropped from the live tables.

Costs. Each traded dollar is charged half a spread. The spread is the month's mean CRSP
closing quoted spread (09_spreads.py) where a stock-month has one, and JKP's Corwin-Schultz
estimate where it does not; the tables report both the cost on that series and the cost on
Corwin-Schultz alone, the share of traded weight priced from quotes, and the two spreads by
size group, with Abdi-Ranaldo as a third reading.

Outputs: out-of-sample R2 against a zero forecast, pooled and by year; rank IC (monthly
Spearman between forecast and realised return, all rows and ex nano/micro); Diebold-Mariano
statistics on monthly cross-sectional loss differences with a Newey-West variance; decile
long-short portfolios, equal- and value-weighted, with and without micro and nano caps;
turnover; a cost charge of turnover times each stock's half spread (Corwin-Schultz,
floored at zero); and, for the full model, the permutation importance of every
characteristic in every test year, which 08_stage1_eventtime.py puts in event time around
each characteristic's publication year.
--------------------------------------------------------------------------------------
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE / "data"
CHUNKS = DATA / "jkp_us"
PRIVATE = DATA / "stage1"
OUT = HERE / "output"
CHARS_FILE = HERE / "jkp_characteristics.csv"
PROFILE_FILE = DATA / "jkp_profile_2010m01.csv"

FIRST_YEAR = 1963
FIRST_TEST = 1995
LAST_TEST = 2024
VAL_YEARS = 8
SEEDS = 5
BATCH = 10000
MAX_EPOCHS = 100
PATIENCE = 5

MODELS = ["huber", "enet", "pcr", "pls", "gbrt", "nn1", "nn2", "nn3"]
IMPORTANCE_MODELS = ["huber", "gbrt", "nn3"]
ID_COLS = ["id", "eom", "permno", "prc", "size_grp", "me", "ret_exc_lead1m", "bidaskhl_21d", "source_crsp"]
PANEL_VERSION = 4          # bump when the cached panel's columns change
SPREADS = DATA / "spreads"  # 09_spreads.py: monthly quoted and Abdi-Ranaldo spreads by permno

import importlib.util
import json
import os
import shutil
import subprocess

HAVE_LGB = importlib.util.find_spec("lightgbm") is not None
MANIFEST_NAME = "stage1_run.json"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def start_run(args, tag):
    """Every run gets an id that is written into each output row and file, and a manifest
    in output/. The tables are built only from rows carrying the current id, so nothing
    from an earlier run can leak into them. --skip-done continues the run in the manifest."""
    manifest = OUT / MANIFEST_NAME
    prev = json.loads(manifest.read_text()) if manifest.exists() else None
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if args.skip_done and prev and prev.get("panel") == tag:
        prev.setdefault("continued", []).append(now)
        manifest.write_text(json.dumps(prev, indent=1))
        return prev["run_id"]
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + git_commit()
    if prev and prev.get("run_id"):
        # the earlier run's tables move to output/runs/<its id>/ before this run's rows replace them
        archive = OUT / "runs" / prev["run_id"]
        archive.mkdir(parents=True, exist_ok=True)
        for f in list(OUT.glob("stage1_*.csv")) + [manifest]:
            shutil.copy2(f, archive / f.name)
        print(f"tables of run {prev['run_id']} archived in {archive}")
    flags = {k: v for k, v in vars(args).items() if k not in ("worker", "run_id")}
    manifest.write_text(json.dumps({"run_id": run_id, "commit": git_commit(), "started": now,
                                    "panel": tag, "flags": flags,
                                    "test_years": list(range(args.first_test, args.last_test + 1, args.every)),
                                    "python": sys.version.split()[0], "numpy": np.__version__,
                                    "pandas": pd.__version__}, indent=1))
    return run_id


def current_run_id():
    manifest = OUT / MANIFEST_NAME
    if not manifest.exists():
        sys.exit(f"no {MANIFEST_NAME} in {OUT}; run the fits first")
    return json.loads(manifest.read_text())["run_id"]
HAVE_TORCH = importlib.util.find_spec("torch") is not None
NN_MODELS = ["nn1", "nn2", "nn3"]


# ---------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------

def rank_transform(df, cols):
    """Map each characteristic to [-1, 1] by its rank within one month.

    Ties get the average rank. Missing values become 0, which is the
    cross-sectional median after the mapping. A column with a single
    non-missing value maps it to 0; an all-missing column is all 0.
    """
    x = df[cols].replace([np.inf, -np.inf], np.nan)
    r = x.rank(method="average")          # 1..n among non-missing, NaN stays NaN
    n = x.notna().sum()                   # non-missing count per column
    denom = (n - 1).where(n > 1)          # NaN when n <= 1, avoids 0/0
    z = 2 * (r - 1) / denom - 1
    return z.fillna(0).astype(np.float32)


def rank_transform_panel(df, cols, date_col="eom"):
    """Same mapping for a panel, month by month. Run it one year at a time."""
    x = df[cols].replace([np.inf, -np.inf], np.nan)
    g = x.groupby(df[date_col])
    r = g.rank(method="average")
    n = g.transform("count")
    z = 2 * (r - 1) / (n - 1).where(n > 1) - 1
    return z.fillna(0).astype(np.float32)


def characteristics() -> pd.DataFrame:
    return pd.read_csv(CHARS_FILE)


def save_private(df: pd.DataFrame, path: Path) -> Path:
    """Parquet when pyarrow is installed, csv otherwise (the synthetic test machine)."""
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        return alt


def load_private(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def private_files():
    return sorted(list(PRIVATE.glob("preds_*.parquet")) + list(PRIVATE.glob("preds_*.csv")))


# ---------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------

def load_panel(chars: List[str], first: int, last: int,
               synthetic: bool = False, rebuild: bool = False) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Return (meta, X, cols): one row per stock-month, X the ranked features.

    The ranked panel is cached in data/stage1 so the two worker processes rank once."""
    PRIVATE.mkdir(parents=True, exist_ok=True)
    tag = "synthetic" if synthetic else "jkp"
    manifest = PRIVATE / f"panel_{tag}.json"
    xfile, mfile = PRIVATE / f"panel_{tag}_X.npy", PRIVATE / f"panel_{tag}_meta.parquet"
    want = {"first": first, "last": last, "version": PANEL_VERSION}
    if not rebuild and manifest.exists() and xfile.exists():
        m = json.loads(manifest.read_text())
        if m.get("first") == first and m.get("last") == last and m.get("version") == PANEL_VERSION:
            meta = load_private(pred_path_like(mfile))
            meta["eom"] = pd.to_datetime(meta["eom"])
            meta["target_month"] = pd.PeriodIndex(meta["target_month"], freq="M")
            return meta, np.load(xfile), m["cols"]
    if synthetic:
        meta, X, cols = synthetic_panel(chars, first, last)
    else:
        files = sorted(CHUNKS.glob("jkp_us_*.parquet"))
        files = [f for f in files if first <= int(f.stem.split("_")[-1]) <= last]
        if not files:
            sys.exit(f"no year files in {CHUNKS}; run 06_jkp_pull.py first")
        first_cols = set(pd.read_parquet(files[0]).columns)
        avail = [c for c in chars if c in first_cols]
        id_have = [c for c in ID_COLS if c in first_cols]
        metas, blocks = [], []
        for f in files:
            df = pd.read_parquet(f, columns=list(dict.fromkeys(id_have + avail)))
            for c in ID_COLS:
                if c not in df:
                    df[c] = np.nan
            blocks.append(rank_transform_panel(df, avail).to_numpy())
            metas.append(df[ID_COLS])
            print(f"  ranked {f.stem}: {len(df):,} rows", flush=True)
        meta = pd.concat(metas, ignore_index=True)
        meta = add_spreads(meta)
        meta, X, cols = prepare_meta(meta), np.vstack(blocks), avail
    np.save(xfile, X)
    save_private(meta.assign(target_month=meta["target_month"].astype(str)), mfile)
    manifest.write_text(json.dumps({**want, "cols": cols}))
    return meta, X, cols


def pred_path_like(p: Path) -> Path:
    return p if p.exists() or not p.with_suffix(".csv").exists() else p.with_suffix(".csv")


def add_spreads(meta: pd.DataFrame) -> pd.DataFrame:
    """Join the monthly quoted and Abdi-Ranaldo spreads from 09_spreads.py on permno and
    the month of eom. Without the files the columns are missing and the costs fall back to
    Corwin-Schultz throughout."""
    files = sorted(SPREADS.glob("spreads_*.parquet"))
    if not files:
        print("  no spread files in data/spreads; costs use Corwin-Schultz only", flush=True)
        return meta
    sp = pd.concat((pd.read_parquet(f, columns=["permno", "month", "quoted", "ar", "n_quoted"]) for f in files),
                   ignore_index=True)
    sp = sp[sp["n_quoted"] >= 5]              # a month needs five quoted days to count
    sp["month"] = pd.to_datetime(sp["month"]).dt.to_period("M")
    sp["permno"] = sp["permno"].astype("float64")
    m = meta.copy()
    m["month"] = pd.to_datetime(m["eom"]).dt.to_period("M")
    m["permno"] = m["permno"].astype("float64")
    m = m.merge(sp.drop(columns="n_quoted"), on=["permno", "month"], how="left").drop(columns="month")
    print(f"  quoted spread found for {m['quoted'].notna().mean():.3f} of rows", flush=True)
    return m


def prepare_meta(meta: pd.DataFrame) -> pd.DataFrame:
    m = meta.copy()
    m["eom"] = pd.to_datetime(m["eom"])
    m["target_month"] = m["eom"].dt.to_period("M") + 1
    m["target_year"] = m["target_month"].dt.year.astype(int)
    m["y"] = m["ret_exc_lead1m"].astype(np.float32)
    # Two spreads. Corwin-Schultz (JKP's bidaskhl_21d) can come out negative; a negative
    # spread is noise, not a rebate, so it is floored at zero. The quoted spread from CRSP's
    # closing bid and ask (09_spreads.py) is the one the costs use where it exists; the
    # Corwin-Schultz figure fills the stock-months without a quote.
    m["half_spread_cs"] = (m["bidaskhl_21d"].clip(lower=0) / 2).astype(np.float32)
    if "quoted" not in m:
        m["quoted"] = np.nan
    if "ar" not in m:
        m["ar"] = np.nan
    m["half_spread"] = np.where(m["quoted"].notna(), m["quoted"] / 2, m["half_spread_cs"]).astype(np.float32)
    m["spread_source"] = np.where(m["quoted"].notna(), "quoted", "cs")
    m["small"] = m["size_grp"].astype(str).str.lower().isin(["micro", "nano"])
    # a CRSP return: JKP's source flag when the column is there, else a permno
    if "source_crsp" in m and m["source_crsp"].notna().any():
        m["crsp"] = m["source_crsp"].fillna(0).to_numpy() == 1
    else:
        m["crsp"] = m["permno"].notna() if "permno" in m else True
    m["spread_raw"] = m["bidaskhl_21d"].astype(np.float32)
    return m


def synthetic_panel(chars: List[str], first: int, last: int,
                    n_stocks: int = 400, seed: int = 0):
    """A panel with the shape of the real one, for testing code paths without WRDS.

    Missingness per characteristic comes from data/jkp_profile_2010m01.csv when that file
    exists. The target is a small linear signal in the first six characteristics plus
    noise; the coefficient on each of those decays after the characteristic's publication
    year, so the event-time design in 08 has something to find.
    """
    rng = np.random.default_rng(seed)
    info = characteristics().set_index("characteristic")
    miss = pd.Series(0.1, index=chars)
    if PROFILE_FILE.exists():
        prof = pd.read_csv(PROFILE_FILE, index_col=0)
        for c in chars:
            if c in prof.index:
                miss[c] = 1 - float(prof.loc[c, "non_missing"])
    months = pd.period_range(f"{first}-01", f"{last}-12", freq="M")
    rows, blocks = [], []
    signal_cols = chars[:6]
    for m in months:
        n = n_stocks
        raw = rng.standard_normal((n, len(chars))).astype(np.float32)
        for j, c in enumerate(chars):
            drop = rng.random(n) < miss[c]
            raw[drop, j] = np.nan
        df = pd.DataFrame(raw, columns=chars)
        z = rank_transform(df, chars).to_numpy()
        y = rng.standard_normal(n).astype(np.float32) * 0.12
        for j, c in enumerate(signal_cols):
            pub = int(info.loc[c, "pub_year"])
            b = 0.01 if m.year < pub else 0.003
            y += b * z[:, j]
        # a few absurd returns, as in the real panel: about one row in 5,000 gets a
        # return between 500% and 50,000%
        wild = rng.random(n) < 0.0002
        y[wild] = rng.uniform(5, 500, wild.sum()).astype(np.float32)
        me = np.exp(rng.normal(6, 2, n)).astype(np.float32)
        permno = np.where(rng.random(n) < 0.06, np.nan, np.arange(n) + 10000).astype(np.float32)
        source = np.where(np.isnan(permno), 0.0, np.where(rng.random(n) < 0.02, 0.0, 1.0)).astype(np.float32)
        q = pd.qcut(me, [0, .2, .5, .8, .95, 1], labels=["nano", "micro", "small", "large", "mega"])
        rows.append(pd.DataFrame({
            "id": np.arange(n), "eom": m.to_timestamp(how="end").normalize(),
            "permno": permno, "prc": np.exp(rng.normal(2.5, 1.2, n)).astype(np.float32),
            "source_crsp": source,
            "size_grp": q.astype(str), "me": me, "ret_exc_lead1m": y,
            "bidaskhl_21d": rng.normal(0.01, 0.01, n).astype(np.float32),
            "quoted": np.where(rng.random(n) < 0.9, np.abs(rng.normal(0.006, 0.004, n)), np.nan).astype(np.float32),
            "ar": np.abs(rng.normal(0.007, 0.005, n)).astype(np.float32)}))
        blocks.append(z)
    meta = pd.concat(rows, ignore_index=True)
    return prepare_meta(meta), np.vstack(blocks), list(chars)


# ---------------------------------------------------------------------------------
# models: each fit_* returns (predict, chosen_setting)
# ---------------------------------------------------------------------------------

def mse(y, p):
    return float(np.mean((y - p) ** 2))


def fit_huber(Xtr, ytr, Xva, yva, smoke=False):
    """OLS with a Huber loss by iteratively reweighted least squares.

    Observations with |residual| above xi get weight xi/|residual|, which is the
    first-order condition of the Huber objective. xi is a quantile of the absolute
    residuals of a plain OLS fit, chosen on the validation years.
    """
    def solve(X, y, xi):
        A = np.hstack([np.ones((len(X), 1), np.float32), X])
        w = np.ones(len(y), np.float32)
        beta = None
        for _ in range(3 if smoke else 10):
            Aw = A * w[:, None]
            beta = np.linalg.solve(A.T @ Aw + 1e-6 * np.eye(A.shape[1]), Aw.T @ y)
            r = np.abs(y - A @ beta)
            w = np.minimum(1.0, xi / np.maximum(r, 1e-12)).astype(np.float32)
        return beta

    A = np.hstack([np.ones((len(Xtr), 1), np.float32), Xtr])
    ols = np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ ytr)
    res = np.abs(ytr - A @ ols)
    best = None
    for q in ([0.99] if smoke else [0.95, 0.99, 0.999]):
        xi = float(np.quantile(res, q))
        beta = solve(Xtr, ytr, xi)
        v = mse(yva, beta[0] + Xva @ beta[1:])
        if best is None or v < best[0]:
            best = (v, q, beta)
    _, q, beta = best
    return (lambda Z: (beta[0] + Z @ beta[1:]).astype(np.float32)), {"xi_quantile": q}


def fit_enet(Xtr, ytr, Xva, yva, smoke=False):
    from sklearn.linear_model import ElasticNet
    mx, my = Xtr.mean(axis=0), float(ytr.mean())
    best = None
    grid = [(1e-3, 0.5)] if smoke else [(a, l) for a in (1e-4, 1e-3, 1e-2) for l in (0.5, 0.9)]
    for alpha, l1 in grid:
        m = ElasticNet(alpha=alpha, l1_ratio=l1, fit_intercept=False, precompute=True,
                       max_iter=2000, tol=1e-4).fit(Xtr - mx, ytr - my)
        v = mse(yva, my + m.predict(Xva - mx))
        if best is None or v < best[0]:
            best = (v, alpha, l1, m)
    _, alpha, l1, m = best
    return (lambda Z: (my + m.predict(Z - mx)).astype(np.float32)), {"alpha": alpha, "l1": l1}


def fit_pcr(Xtr, ytr, Xva, yva, smoke=False):
    """Principal components of the training features, then OLS on the first K."""
    mx, my = Xtr.mean(axis=0), float(ytr.mean())
    C = ((Xtr - mx).T @ (Xtr - mx)) / len(Xtr)
    vals, vecs = np.linalg.eigh(C.astype(np.float64))
    order = np.argsort(vals)[::-1]
    V = vecs[:, order].astype(np.float32)
    best = None
    for K in ([5] if smoke else [1, 3, 5, 10, 20, 30]):
        K = min(K, V.shape[1])
        Ptr = (Xtr - mx) @ V[:, :K]
        g = np.linalg.solve(Ptr.T @ Ptr + 1e-6 * np.eye(K), Ptr.T @ (ytr - my))
        v = mse(yva, my + ((Xva - mx) @ V[:, :K]) @ g)
        if best is None or v < best[0]:
            best = (v, K, g)
    _, K, g = best
    return (lambda Z: (my + ((Z - mx) @ V[:, :K]) @ g).astype(np.float32)), {"K": K}


def fit_pls(Xtr, ytr, Xva, yva, smoke=False, max_rows=1_500_000):
    from sklearn.cross_decomposition import PLSRegression
    idx = np.arange(len(Xtr))
    if len(idx) > max_rows:                      # PLS is the slow one; subsample its fit
        idx = np.random.default_rng(0).choice(idx, max_rows, replace=False)
    best = None
    for K in ([3] if smoke else [1, 3, 5, 10]):
        m = PLSRegression(n_components=K, scale=False, max_iter=200).fit(Xtr[idx], ytr[idx])
        v = mse(yva, m.predict(Xva).ravel())
        if best is None or v < best[0]:
            best = (v, K, m)
    _, K, m = best
    return (lambda Z: m.predict(Z).ravel().astype(np.float32)), {"K": K}


def fit_gbrt(Xtr, ytr, Xva, yva, smoke=False, robust=False):
    """With robust=True LightGBM minimises a Huber loss whose threshold is 1.345 robust
    standard deviations of the training target (Gu, Kelly and Xiu's GBRT+H)."""
    best = None
    grid = [(2, 0.1)] if smoke else [(d, lr) for d in (2, 3) for lr in (0.05, 0.1)]
    for depth, lr in grid:
        if HAVE_LGB:
            import lightgbm as lgb
            loss = {}
            if robust:
                mad = float(np.median(np.abs(ytr - np.median(ytr)))) * 1.4826
                loss = {"objective": "huber", "alpha": 1.345 * (mad or float(np.std(ytr)))}
            m = lgb.LGBMRegressor(max_depth=depth, num_leaves=2 ** depth, learning_rate=lr,
                                  n_estimators=50 if smoke else 1000, subsample=0.5,
                                  subsample_freq=1, colsample_bytree=0.5,
                                  min_child_samples=500, reg_lambda=1.0, verbose=-1,
                                  random_state=0, **loss)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            n_best = int(m.best_iteration_ or m.n_estimators)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr,
                                              max_iter=50 if smoke else 1000,
                                              min_samples_leaf=500, l2_regularization=1.0,
                                              early_stopping=False, random_state=0).fit(Xtr, ytr)
            curve = [mse(yva, p) for p in m.staged_predict(Xva)]
            n_best = int(np.argmin(curve)) + 1
            m.set_params(max_iter=n_best)
            m = m.fit(Xtr, ytr)
        v = mse(yva, m.predict(Xva))
        if best is None or v < best[0]:
            best = (v, depth, lr, n_best, m)
    _, depth, lr, n_best, m = best
    return (lambda Z: m.predict(Z).astype(np.float32)), {"depth": depth, "lr": lr, "trees": n_best}


def nn_layers(name: str) -> List[int]:
    return {"nn1": [32], "nn2": [32, 16], "nn3": [32, 16, 8]}[name]


def fit_nn_torch(Xtr, ytr, Xva, yva, layers, device, smoke=False, seeds=SEEDS):
    """Feed-forward net: ReLU, batch normalisation, Adam, L1 penalty, early stopping on
    the validation loss, averaged over seeds.

    The target is divided by its training standard deviation inside this function and
    the forecasts are multiplied back, so the loss is on a unit scale whatever the
    return units are; the output layer starts near zero, so an untrained net forecasts
    the mean, not noise of its own."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    dev = torch.device(device)
    y_sd = float(ytr.std()) or 1.0
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr)).to(dev)
    yt = torch.from_numpy(np.ascontiguousarray(ytr / y_sd)).to(dev)
    Xv = torch.from_numpy(np.ascontiguousarray(Xva)).to(dev)
    yv = torch.from_numpy(np.ascontiguousarray(yva / y_sd)).to(dev)

    def build(seed):
        torch.manual_seed(seed)
        mods, d = [], Xtr.shape[1]
        for h in layers:
            mods += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        head = nn.Linear(d, 1)
        with torch.no_grad():
            head.weight.mul_(0.01)
            head.bias.zero_()
        mods.append(head)
        return nn.Sequential(*mods).to(dev)

    def train(seed, l1):
        net = build(seed)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        best_v, best_state, bad = np.inf, None, 0
        n = len(Xt)
        g = torch.Generator(device="cpu").manual_seed(seed)
        epochs = 10 if smoke else MAX_EPOCHS
        for epoch in range(epochs):
            net.train()
            perm = torch.randperm(n, generator=g).to(dev)
            for i in range(0, n, BATCH):
                b = perm[i:i + BATCH]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                pred = net(Xt[b]).squeeze(1)
                loss = ((pred - yt[b]) ** 2).mean()
                if l1 > 0:
                    loss = loss + l1 * sum(p.abs().sum() for p in net.parameters())
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(((net(Xv).squeeze(1) - yv) ** 2).mean())
            if v < best_v - 1e-7:
                best_v, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return net, best_v, epoch + 1

    best = None
    for l1 in ([1e-5] if smoke else [1e-5, 1e-4]):
        nets, vs, eps = [], [], []
        for s_ in range(1 if smoke else seeds):
            net, v, ep = train(s_, l1)
            nets.append(net); vs.append(v); eps.append(ep)
        with torch.no_grad():
            ens = torch.stack([n_(Xv).squeeze(1) for n_ in nets]).mean(0)
            v_ens = float(((ens - yv) ** 2).mean())
        if best is None or v_ens < best[0]:
            best = (v_ens, l1, nets, eps)
    _, l1, nets, eps = best

    def predict(Z):
        Zt = torch.from_numpy(np.ascontiguousarray(Z)).to(dev)
        with torch.no_grad():
            out = []
            for i in range(0, len(Zt), 200000):
                out.append(torch.stack([n_(Zt[i:i + 200000]).squeeze(1) for n_ in nets]).mean(0).cpu())
        return (torch.cat(out).numpy() * y_sd).astype(np.float32)
    return predict, {"l1": l1, "epochs": eps}


def fit_nn_sklearn(Xtr, ytr, Xva, yva, layers, smoke=False, seeds=SEEDS):
    """Fallback when torch is not installed: early stopping on the validation years is
    done by hand through partial_fit over epochs."""
    from sklearn.neural_network import MLPRegressor
    y_sd = float(ytr.std()) or 1.0
    ytr, yva = ytr / y_sd, yva / y_sd
    best = None
    for alpha in ([1e-4] if smoke else [1e-4, 1e-3]):
        nets, eps = [], []
        for s in range(1 if smoke else seeds):
            m = MLPRegressor(hidden_layer_sizes=tuple(layers), alpha=alpha, batch_size=BATCH,
                             learning_rate_init=1e-3, max_iter=1, warm_start=True,
                             random_state=s)
            best_v, bad, best_coef = np.inf, 0, None
            for ep in range(10 if smoke else MAX_EPOCHS):
                m.fit(Xtr, ytr)
                v = mse(yva, m.predict(Xva))
                if v < best_v - 1e-7:
                    best_v, bad = v, 0
                    best_coef = ([c.copy() for c in m.coefs_], [b.copy() for b in m.intercepts_])
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            m.coefs_, m.intercepts_ = best_coef
            nets.append(m); eps.append(ep + 1)
        v_ens = mse(yva, np.mean([n_.predict(Xva) for n_ in nets], axis=0))
        if best is None or v_ens < best[0]:
            best = (v_ens, alpha, nets, eps)
    _, alpha, nets, eps = best
    return (lambda Z: (y_sd * np.mean([n_.predict(Z) for n_ in nets], axis=0)).astype(np.float32)), \
        {"alpha": alpha, "epochs": eps}


def fit_model(name, Xtr, ytr, Xva, yva, device, smoke, robust=False):
    if name == "huber":
        return fit_huber(Xtr, ytr, Xva, yva, smoke)
    if name == "enet":
        return fit_enet(Xtr, ytr, Xva, yva, smoke)
    if name == "pcr":
        return fit_pcr(Xtr, ytr, Xva, yva, smoke)
    if name == "pls":
        return fit_pls(Xtr, ytr, Xva, yva, smoke)
    if name == "gbrt":
        return fit_gbrt(Xtr, ytr, Xva, yva, smoke, robust)
    if name in ("nn1", "nn2", "nn3"):
        if HAVE_TORCH:
            return fit_nn_torch(Xtr, ytr, Xva, yva, nn_layers(name), device, smoke)
        return fit_nn_sklearn(Xtr, ytr, Xva, yva, nn_layers(name), smoke)
    raise ValueError(name)


# ---------------------------------------------------------------------------------
# the expanding window
# ---------------------------------------------------------------------------------

def splits(meta: pd.DataFrame, y: int):
    """Boolean masks by target month. The boundary assert is the no-look-ahead rule."""
    ty = meta["target_year"].to_numpy()
    tr = ty <= y - VAL_YEARS - 1
    va = (ty >= y - VAL_YEARS) & (ty <= y - 1)
    te = ty == y
    tm = meta["target_month"]
    assert tm[tr].max() < tm[va].min() <= tm[va].max() < tm[te].min(), y
    return tr, va, te


def pred_path(y):
    p = PRIVATE / f"preds_{y}.parquet"
    return p if p.exists() or not p.with_suffix(".csv").exists() else p.with_suffix(".csv")


def run(meta, X, cols, chars, args):
    """Fit args.models for every test year and merge the forecasts into data/stage1."""
    PRIVATE.mkdir(parents=True, exist_ok=True)
    pub = chars.set_index("characteristic")["pub_year"]
    col_idx = {c: i for i, c in enumerate(cols)}
    years = list(range(args.first_test, args.last_test + 1, args.every))
    tuning, timing, imp_rows = [], [], []
    valid = ~np.isnan(meta["y"].to_numpy())
    tag = "synthetic" if args.synthetic else "jkp"

    for y in years:
        pf = pred_path(y)
        existing = load_private(pf) if pf.exists() else None
        if existing is not None and (existing.get("panel", pd.Series([None])).iloc[0] != tag
                                     or existing.get("run_id", pd.Series([None])).iloc[0] != args.run_id):
            print(f"{y}: {pf.name} holds forecasts from another run; it will be replaced", flush=True)
            existing = None
        wanted = [f"{m}_{s_}" for m in args.models for s_ in ("public", "full")]
        if args.skip_done and existing is not None and all(c in existing.columns for c in wanted):
            print(f"{y}: forecasts on disk, skipping")
            continue
        tr, va, te = splits(meta, y)
        tr &= valid; va &= valid; te &= valid
        if args.crsp_only:
            crsp = meta["crsp"].to_numpy()
            tr &= crsp; va &= crsp; te &= crsp
        if te.sum() == 0:
            continue
        sets = {"public": [c for c in cols if pub[c] < y], "full": list(cols)}
        out = meta.loc[te, ["id", "eom", "target_month", "size_grp", "me", "half_spread", "half_spread_cs",
                            "spread_source", "spread_raw", "quoted", "ar", "y", "crsp"]].copy()
        out["panel"] = tag
        out["run_id"] = args.run_id
        ytr, yva = meta.loc[tr, "y"].to_numpy(), meta.loc[va, "y"].to_numpy()
        y_te = out["y"].to_numpy()
        # --robust: the fitting and selection targets are clipped at the 0.1 and 99.9
        # percentiles of the training returns (nothing from the test year is used); the
        # Huber regression keeps the raw training target, its loss is the protection.
        # Forecasts are always scored against the raw returns.
        if args.robust:
            lo, hi = np.quantile(ytr, [0.001, 0.999])
            ytr_fit, yva_fit = np.clip(ytr, lo, hi), np.clip(yva, lo, hi)
            clipped = int((ytr < lo).sum() + (ytr > hi).sum())
        else:
            lo, hi, ytr_fit, yva_fit, clipped = np.nan, np.nan, ytr, yva, 0
        print(f"{y}: train {tr.sum():,} val {va.sum():,} test {te.sum():,} rows; "
              f"public set {len(sets['public'])} of {len(cols)}"
              + (f"; training target clipped to [{lo:+.3f}, {hi:+.3f}] ({clipped:,} rows)" if args.robust else ""),
              flush=True)
        for set_name, scols in sets.items():
            j = [col_idx[c] for c in scols]
            Xtr, Xva, Xte = X[tr][:, j], X[va][:, j], X[te][:, j]
            for name in args.models:
                t0 = time.time()
                predict, setting = fit_model(name, Xtr, ytr if name == "huber" else ytr_fit, Xva, yva_fit,
                                             args.device, args.smoke, args.robust)
                p = predict(Xte)
                out[f"{name}_{set_name}"] = p
                dt = time.time() - t0
                # two scale checks, both in units of the test-year return sd. The forecasts
                # should be far less dispersed than returns (ratio well below one), and their
                # mean should sit near the mean return (offset near zero); a model with a
                # small ratio and a large offset is mis-centred, which the R2 punishes and a
                # rank-based portfolio never notices
                y_sd = float(np.std(y_te) or 1.0)
                ratio = float(np.std(p) / y_sd)
                offset = float((np.mean(p) - np.mean(y_te)) / y_sd)
                tuning.append({"run_id": args.run_id, "year": y, "set": set_name, "model": name,
                               "setting": str(setting),
                               "pred_sd_over_y_sd": ratio, "pred_mean_minus_y_mean_over_y_sd": offset,
                               "robust": bool(args.robust), "clip_lo": lo, "clip_hi": hi,
                               "train_rows_clipped": clipped, "crsp_only": bool(args.crsp_only)})
                timing.append({"run_id": args.run_id, "year": y, "set": set_name, "model": name, "seconds": dt})
                print(f"   {set_name:<6} {name:<6} {dt:6.0f}s  sd ratio {ratio:.3f}  offset {offset:+.3f}  {setting}",
                      flush=True)
                if set_name == "full" and name in args.importance_models:
                    base = mse(y_te, p)
                    rng = np.random.default_rng(y)
                    for k, c in enumerate(scols):
                        Zp = Xte.copy()
                        Zp[:, k] = rng.permutation(Zp[:, k])
                        imp_rows.append({"run_id": args.run_id, "year": y, "model": name, "characteristic": c,
                                         "mse_increase": mse(y_te, predict(Zp)) - base})
            del Xtr, Xva, Xte
        if existing is not None:
            keep = [c for c in existing.columns if c not in out.columns]
            out = pd.concat([out.reset_index(drop=True), existing[keep].reset_index(drop=True)], axis=1)
        save_private(out, PRIVATE / f"preds_{y}.parquet")
        merge_csv(OUT / "stage1_tuning.csv", pd.DataFrame(tuning), ["year", "set", "model"])
        merge_csv(OUT / "stage1_timing.csv", pd.DataFrame(timing), ["year", "set", "model"])
        if imp_rows:
            merge_csv(OUT / "stage1_importance.csv", pd.DataFrame(imp_rows), ["year", "model"])


def merge_csv(path, new, keys):
    """Replace the rows of `path` that share `keys` with `new`, keep the rest."""
    if path.exists():
        prev = pd.read_csv(path)
        k_new = set(map(tuple, new[keys].astype(str).to_numpy()))
        mask = [tuple(r) not in k_new for r in prev[keys].astype(str).to_numpy()]
        new = pd.concat([prev[mask], new], ignore_index=True)
    new.to_csv(path, index=False)


# ---------------------------------------------------------------------------------
# evaluation, from the saved forecasts
# ---------------------------------------------------------------------------------

def r2_oos(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum(y ** 2)


def drop_other_runs(path, run_id, years):
    """Keep only the rows of a per-year csv that carry this run's id and forecast years."""
    if not path.exists():
        return
    d = pd.read_csv(path)
    keep = d["year"].isin(years)
    keep &= (d["run_id"] == run_id) if "run_id" in d else False
    if (~keep).any():
        print(f"{path.name}: dropped {(~keep).sum():,} rows from other runs or years")
        d[keep].to_csv(path, index=False)


def rank_ic(P, col, mask):
    """Monthly cross-sectional Spearman correlation between the forecast and the realised
    return; mean, plain and Newey-West t-statistics, share of positive months."""
    sub = P.loc[mask, ["target_month", col, "y"]]
    r = sub.groupby("target_month")[[col, "y"]].rank()
    r["target_month"] = sub["target_month"].to_numpy()
    ic = r.groupby("target_month").apply(lambda g: g[col].corr(g["y"])).dropna()
    x = ic.to_numpy()
    return {"ic_mean": x.mean(), "ic_sd": x.std(ddof=1), "t": x.mean() / (x.std(ddof=1) / np.sqrt(len(x))),
            "t_nw": x.mean() / np.sqrt(nw_var(x)), "months": len(x), "share_positive": (x > 0).mean()}


def target_check(meta, path):
    """What the realised next-month returns look like, test year by test year: how heavy
    the tails are and how much of the squared-return total a handful of rows carry.
    Aggregates only."""
    d = meta[~np.isnan(meta["y"].to_numpy())]
    rows, top_rows = [], []
    for yr, g in d.groupby("target_year"):
        if yr < FIRST_TEST or yr > LAST_TEST:
            continue
        y = g["y"].to_numpy(np.float64)
        ay = np.abs(y)
        top = np.sort(y ** 2)[::-1]
        ext = ay > 1
        rows.append({"year": int(yr), "n": len(y), "mean": y.mean(), "sd": y.std(), "min": y.min(),
                     "max": y.max(), "q001": np.quantile(y, 0.001), "q999": np.quantile(y, 0.999),
                     "n_abs_gt_1": int(ext.sum()), "n_abs_gt_5": int((ay > 5).sum()),
                     "n_abs_gt_10": int((ay > 10).sum()),
                     "share_sq_top1": top[0] / top.sum(), "share_sq_top10": top[:10].sum() / top.sum(),
                     "share_sq_abs_gt_1": (y[ext] ** 2).sum() / top.sum(),
                     "ext_share_no_permno": float(g["permno"].isna().to_numpy()[ext].mean()) if ext.any() else np.nan,
                     "ext_share_not_crsp": float((~g["crsp"].to_numpy()[ext]).mean()) if ext.any() else np.nan,
                     "ext_share_small": float(g["small"].to_numpy()[ext].mean()) if ext.any() else np.nan,
                     "ext_share_prc_lt_1": float((g["prc"].to_numpy()[ext] < 1).mean()) if ext.any() else np.nan,
                     "share_no_permno": float(g["permno"].isna().mean()),
                     "share_not_crsp": float((~g["crsp"].to_numpy()).mean())})
        # the three largest rows of the year, as categories only: where the squared-return
        # total sits and what kind of row carries it
        order = np.argsort(ay)[::-1][:3]
        for r, i in enumerate(order):
            row = g.iloc[i]
            prc = row["prc"]
            top_rows.append({"year": int(yr), "rank": r + 1, "share_sq": float(y[i] ** 2 / top.sum()),
                             "sign": "+" if y[i] > 0 else "-",
                             "crsp_return": bool(row["crsp"]), "has_permno": bool(pd.notna(row["permno"])),
                             "prc_bucket": "missing" if pd.isna(prc) else ("<0.1" if prc < 0.1 else "<1" if prc < 1 else "<5" if prc < 5 else ">=5"),
                             "size_grp": str(row["size_grp"])})
    t = pd.DataFrame(rows)
    t.to_csv(path, index=False)
    tt = pd.DataFrame(top_rows)
    tt.to_csv(path.with_name(path.stem + "_top.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(t[["year", "n", "sd", "max", "q999", "n_abs_gt_1", "n_abs_gt_10", "share_sq_top1",
                 "share_sq_top10", "share_sq_abs_gt_1", "ext_share_not_crsp", "ext_share_small",
                 "ext_share_prc_lt_1", "share_not_crsp"]].round(3).to_string(index=False))
        print("\nthe largest row of each year:")
        print(tt[tt["rank"] == 1].drop(columns="rank").round(3).to_string(index=False))
    print(f"wrote {path} and {path.stem}_top.csv")


def nw_var(d, lags=12):
    d = d - d.mean()
    T = len(d)
    v = np.sum(d * d) / T
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * np.sum(d[L:] * d[:-L]) / T
    return v / T


def diebold_mariano(P, a, b):
    """Monthly cross-sectional mean squared error, model a minus model b; positive means
    b has the lower loss."""
    la = (P["y"] - P[a]) ** 2
    lb = (P["y"] - P[b]) ** 2
    d = (la - lb).groupby(P["target_month"]).mean().to_numpy()
    return float(d.mean() / np.sqrt(nw_var(d)))


def decile_portfolios(P, col, weighting="ew", universe="all"):
    """Each month: sort on the forecast into deciles, long the top, short the bottom.
    Returns monthly gross return, turnover and cost of the long-short position."""
    rows = []
    prev_w = None
    by_size = {}
    sub = P if universe == "all" else P[~P["small"]]
    for m, g in sub.groupby("target_month"):
        if len(g) < 100:
            continue
        d = pd.qcut(g[col].rank(method="first"), 10, labels=False)
        top, bot = g[d == 9], g[d == 0]
        if weighting == "ew":
            w_top = pd.Series(1 / len(top), index=top["id"])
            w_bot = pd.Series(1 / len(bot), index=bot["id"])
        else:
            w_top = top.set_index("id")["me"] / top["me"].sum()
            w_bot = bot.set_index("id")["me"] / bot["me"].sum()
        w = pd.concat([w_top, -w_bot])
        ret = float((w_top * top.set_index("id")["y"]).sum() - (w_bot * bot.set_index("id")["y"]).sum())
        if prev_w is None:
            turn, cost, cost_cs, quoted_share = np.nan, np.nan, np.nan, np.nan
        else:
            dw = w.sub(prev_w, fill_value=0.0)
            adw = dw.abs()
            gi = g.set_index("id")
            hs = gi["half_spread"].reindex(dw.index)
            hs = hs.fillna(hs.median() if hs.notna().any() else 0.0)
            hs_cs = gi["half_spread_cs"].reindex(dw.index)
            hs_cs = hs_cs.fillna(hs_cs.median() if hs_cs.notna().any() else 0.0)
            is_q = (gi["spread_source"].reindex(dw.index) == "quoted")
            turn = float(adw.sum() / 2)
            cost = float((adw * hs).sum())
            cost_cs = float((adw * hs_cs).sum())
            quoted_share = float((adw * is_q).sum() / adw.sum()) if adw.sum() > 0 else np.nan
            sg = gi["size_grp"].astype(str).reindex(dw.index).fillna("left_universe")
            parts = pd.DataFrame({"t": adw, "c": adw * hs, "c_cs": adw * hs_cs, "tq": adw * is_q}).groupby(sg).sum()
            for k, r_ in parts.iterrows():
                acc = by_size.setdefault(k, [0.0, 0.0, 0.0, 0.0])
                acc[0] += r_["t"]; acc[1] += r_["c"]; acc[2] += r_["c_cs"]; acc[3] += r_["tq"]
        rows.append({"target_month": m, "gross": ret, "turnover": turn, "cost": cost, "cost_cs": cost_cs,
                     "quoted_share": quoted_share})
        prev_w = w
    out = pd.DataFrame(rows).set_index("target_month")
    out["net"] = out["gross"] - out["cost"].fillna(0)
    out["net_cs"] = out["gross"] - out["cost_cs"].fillna(0)
    out.attrs["by_size"] = pd.DataFrame(by_size, index=["traded", "cost", "cost_cs", "traded_quoted"]).T
    return out


def summarise(x):
    m, s = x.mean(), x.std(ddof=1)
    return {"mean_pm": 100 * m, "vol_pm": 100 * s, "sharpe": (m / s) * np.sqrt(12),
            "t": m / (s / np.sqrt(len(x)))}


def evaluate(args):
    tag = "synthetic" if args.synthetic else "jkp"
    files, left_out = [], []
    for f in private_files():
        year = int(f.stem.split("_")[-1])
        if args.first_test <= year <= args.last_test:
            files.append(f)
        else:
            left_out.append(f.name)
    if not files:
        sys.exit(f"no forecasts for {args.first_test} to {args.last_test} in {PRIVATE}; run the fits first")
    frames = []
    for f in files:
        d = load_private(f)
        if (d.get("panel", pd.Series([None])).iloc[0] == tag
                and d.get("run_id", pd.Series([None])).iloc[0] == args.run_id):
            frames.append(d)
        else:
            left_out.append(f.name)
    if left_out:
        print(f"left out of the tables ({len(left_out)} files outside {args.first_test} to "
              f"{args.last_test} or from another run): {', '.join(sorted(left_out))}")
    if not frames:
        sys.exit(f"no forecasts from run {args.run_id} in the test years; run the fits first")
    print(f"tables from {len(frames)} forecast files, test years {args.first_test} to {args.last_test}, "
          f"run {args.run_id}")
    P = pd.concat(frames, ignore_index=True)
    P["small"] = P["size_grp"].astype(str).str.lower().isin(["micro", "nano"])
    P["target_month"] = pd.PeriodIndex(P["target_month"], freq="M")
    P["year"] = P["target_month"].dt.year
    pred_cols = [c for c in P.columns if "_" in c and c.rsplit("_", 1)[0] in MODELS]
    run_years = sorted(P["year"].unique())
    for name in ("stage1_tuning.csv", "stage1_timing.csv", "stage1_importance.csv"):
        drop_other_runs(OUT / name, args.run_id, run_years)

    # R2 on four row sets: every row; without nano and micro caps; without the rows whose
    # realised return is beyond +-100%, which carry most of the squared-return total in some
    # years; and the rows with a CRSP return. The paper reports all four.
    universes = {"all": np.ones(len(P), bool), "ex_small": ~P["small"].to_numpy(),
                 "abs_y_le_1": (P["y"].abs() <= 1).to_numpy()}
    if "crsp" in P:
        universes["crsp"] = P["crsp"].fillna(False).to_numpy().astype(bool)
    rows = []
    for uname, mask in universes.items():
        sub = P[mask]
        y = sub["y"].to_numpy()
        for c in pred_cols:
            rows.append({"model": c.rsplit("_", 1)[0], "set": c.rsplit("_", 1)[1], "universe": uname,
                         "year": "all", "r2": r2_oos(y, sub[c].to_numpy()), "n": len(sub)})
        for yr, g in sub.groupby("year"):
            for c in pred_cols:
                rows.append({"model": c.rsplit("_", 1)[0], "set": c.rsplit("_", 1)[1], "universe": uname,
                             "year": yr, "r2": r2_oos(g["y"].to_numpy(), g[c].to_numpy()), "n": len(g)})
    pd.DataFrame(rows).to_csv(OUT / "stage1_r2.csv", index=False)

    # rank IC, added after the first pass: immune to the return tails and to the
    # market-level offset that moves the R2
    ics = []
    for c in pred_cols:
        for uname in ("all", "ex_small"):
            row = {"model": c.rsplit("_", 1)[0], "set": c.rsplit("_", 1)[1], "universe": uname}
            row.update(rank_ic(P, c, universes[uname]))
            ics.append(row)
    ics = pd.DataFrame(ics)
    ics.to_csv(OUT / "stage1_ic.csv", index=False)

    # the spreads behind the cost numbers, by size group over the test rows: Corwin-Schultz
    # (raw, share floored, floored mean), the CRSP closing quoted spread and Abdi-Ranaldo
    sp = P.assign(cs=P["spread_raw"].astype(float), quoted=P["quoted"].astype(float), ar=P["ar"].astype(float))
    panel_sp = sp.groupby("size_grp").agg(
        rows=("cs", "size"), cs_median=("cs", "median"), cs_share_negative=("cs", lambda v: (v < 0).mean()),
        cs_mean_floored=("cs", lambda v: v.clip(lower=0).mean()),
        quoted_coverage=("quoted", lambda v: v.notna().mean()), quoted_median=("quoted", "median"),
        quoted_mean=("quoted", "mean"), ar_median=("ar", "median")).reset_index()
    panel_sp.to_csv(OUT / "stage1_spreads_panel.csv", index=False)

    dm = []
    for c in pred_cols:
        model, s = c.rsplit("_", 1)
        if model != "huber" and f"huber_{s}" in P:
            dm.append({"model": model, "set": s, "benchmark": "huber", "dm_t": diebold_mariano(P, f"huber_{s}", c)})
        if s == "full" and f"{model}_public" in P:
            dm.append({"model": model, "set": "full", "benchmark": f"{model}_public",
                       "dm_t": diebold_mariano(P, f"{model}_public", c)})
    pd.DataFrame(dm).to_csv(OUT / "stage1_dm.csv", index=False)

    monthly, summary, traded = {}, [], []
    for c in pred_cols:
        for weighting in ("ew", "vw"):
            for universe in ("all", "ex_small"):
                port = decile_portfolios(P, c, weighting, universe)
                bs = port.attrs["by_size"]
                for sg, v in bs.iterrows():
                    traded.append({"model": c.rsplit("_", 1)[0], "set": c.rsplit("_", 1)[1],
                                   "weighting": weighting, "universe": universe, "size_grp": sg,
                                   "traded_weight_share": v["traded"] / bs["traded"].sum(),
                                   "quoted_share_of_traded": v["traded_quoted"] / v["traded"] if v["traded"] > 0 else np.nan,
                                   "full_spread_used": 2 * v["cost"] / v["traded"] if v["traded"] > 0 else np.nan,
                                   "full_spread_cs": 2 * v["cost_cs"] / v["traded"] if v["traded"] > 0 else np.nan})
                key = f"{c}_{weighting}_{universe}"
                monthly[key + "_gross"] = port["gross"]
                monthly[key + "_net"] = port["net"]
                monthly[key + "_turnover"] = port["turnover"]
                row = {"model": c.rsplit("_", 1)[0], "set": c.rsplit("_", 1)[1],
                       "weighting": weighting, "universe": universe,
                       "turnover_pm": port["turnover"].mean(), "cost_pm": 100 * port["cost"].mean(),
                       "cost_cs_pm": 100 * port["cost_cs"].mean(),
                       "quoted_share_of_traded": port["quoted_share"].mean()}
                row.update({f"gross_{k}": v for k, v in summarise(port["gross"]).items()})
                row.update({f"net_{k}": v for k, v in summarise(port["net"]).items()})
                row.update({f"net_cs_{k}": v for k, v in summarise(port["net_cs"]).items()})
                summary.append(row)
    pd.DataFrame(monthly).to_csv(OUT / "stage1_portfolios_monthly.csv")
    pd.DataFrame(summary).to_csv(OUT / "stage1_portfolio_summary.csv", index=False)
    traded = pd.DataFrame(traded)
    traded.to_csv(OUT / "stage1_spreads_traded.csv", index=False)

    r2 = pd.DataFrame(rows)
    for uname in universes:
        sub = r2[(r2["year"] == "all") & (r2["universe"] == uname)]
        piv = sub.pivot(index="model", columns="set", values="r2") * 100
        print(f"\nout-of-sample R2 (%), pooled, rows: {uname} (n = {int(sub['n'].iloc[0]):,})")
        print(piv.round(3).to_string())
    print("\nrank IC (monthly Spearman, forecast against realised return), mean and t:")
    piv = ics.pivot_table(index="model", columns=["universe", "set"], values=["ic_mean", "t"])
    print(piv.round(3).to_string())
    print("\nportfolios, value-weighted, all stocks, long-short:")
    s = pd.DataFrame(summary)
    s = s[(s["weighting"] == "vw") & (s["universe"] == "all")]
    print(s[["model", "set", "gross_mean_pm", "gross_sharpe", "turnover_pm", "cost_pm", "net_sharpe",
             "cost_cs_pm", "net_cs_sharpe", "quoted_share_of_traded"]].round(3).to_string(index=False))
    print("\nspreads over the test rows, by size group (full spreads; cs = Corwin-Schultz, quoted = CRSP closing quote):")
    print(panel_sp.round(4).to_string(index=False))
    print("\ntraded-weighted full spread inside the Huber VW long-short, by size group:")
    tw = traded[(traded["model"] == "huber") & (traded["set"] == "full") & (traded["weighting"] == "vw")]
    print(tw[["universe", "size_grp", "traded_weight_share", "quoted_share_of_traded", "full_spread_used",
              "full_spread_cs"]].round(4).to_string(index=False))


# ---------------------------------------------------------------------------------

def timing_run(meta, X, cols, args):
    """NN3 on one year, on cpu and on mps, to pick the device."""
    if not HAVE_TORCH:
        sys.exit("torch is not installed in this interpreter")
    import torch
    y = args.last_test
    tr, va, te = splits(meta, y)
    valid = ~np.isnan(meta["y"].to_numpy())
    tr &= valid; va &= valid
    ytr, yva = meta.loc[tr, "y"].to_numpy(), meta.loc[va, "y"].to_numpy()
    for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        t0 = time.time()
        fit_nn_torch(X[tr], ytr, X[va], yva, nn_layers("nn3"), dev, smoke=False, seeds=1)
        print(f"  nn3, one seed, {tr.sum():,} training rows, device {dev}: {time.time()-t0:.0f}s")


def worker_groups(models):
    """The two groups that must not share a process."""
    nets = [m for m in models if m in NN_MODELS]
    others = [m for m in models if m not in NN_MODELS]
    return [g for g in (others, nets) if g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="made-up panel, no WRDS data")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grids, few epochs, test years 2005 to 2007 (public set 58 to 86 of 153)")
    ap.add_argument("--timing", action="store_true", help="time nn3 on cpu and mps for one year")
    ap.add_argument("--eval-only", action="store_true", help="tables from saved forecasts only")
    ap.add_argument("--skip-done", action="store_true", help="skip test years whose forecasts are already on disk")
    ap.add_argument("--rebuild-panel", action="store_true", help="ignore the cached ranked panel")
    ap.add_argument("--every", type=int, default=1, help="refit every N test years")
    ap.add_argument("--first-test", type=int, default=None)
    ap.add_argument("--last-test", type=int, default=None)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--importance-models", default=",".join(IMPORTANCE_MODELS))
    ap.add_argument("--device", default="cpu", help="cpu or mps, for the networks")
    ap.add_argument("--y-check", action="store_true",
                    help="describe the realised returns by test year (tails, concentration) and stop")
    ap.add_argument("--robust", action="store_true",
                    help="clip the fitting and selection targets at the training 0.1/99.9 percentiles; "
                         "Huber loss for the trees; forecasts still scored on raw returns")
    ap.add_argument("--crsp-only", action="store_true", help="keep only rows with a CRSP permno")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.models = [m for m in args.models.split(",") if m]
    args.importance_models = args.importance_models.split(",")
    if args.first_test is None:
        args.first_test = 2005 if args.smoke else FIRST_TEST
    if args.last_test is None:
        args.last_test = 2007 if args.smoke else LAST_TEST
    global PRIVATE, OUT
    if args.synthetic:
        PRIVATE, OUT = DATA / "stage1_synthetic", OUT / "synthetic"
    OUT.mkdir(parents=True, exist_ok=True)

    tag = "synthetic" if args.synthetic else "jkp"
    if args.eval_only:
        args.run_id = current_run_id()
        evaluate(args)
        return
    if not args.worker and not args.timing and not args.y_check:
        args.run_id = start_run(args, tag)

    if args.timing:                       # timing needs torch only, so it stays in-process
        args.models = [m for m in args.models if m in NN_MODELS] or ["nn3"]
    if args.y_check:                      # no fitting, so no worker split
        args.models = ["huber"]
    groups = worker_groups(args.models)
    if not args.worker and len(groups) > 1:
        # one child per group, so LightGBM's OpenMP and torch's never meet in one process
        for g in groups:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--models", ",".join(g)]
            for flag in ("synthetic", "smoke", "skip_done", "rebuild_panel", "robust", "crsp_only"):
                if getattr(args, flag):
                    cmd.append("--" + flag.replace("_", "-"))
            cmd += ["--every", str(args.every), "--first-test", str(args.first_test),
                    "--last-test", str(args.last_test), "--device", args.device,
                    "--importance-models", ",".join(args.importance_models), "--run-id", args.run_id]
            print(f"\n=== worker: {', '.join(g)} ===", flush=True)
            subprocess.run(cmd, check=True)
        evaluate(args)
        return

    chars = characteristics()
    first = FIRST_YEAR if not args.synthetic else args.first_test - 20
    t0 = time.time()
    meta, X, cols = load_panel(chars["characteristic"].tolist(), first, args.last_test,
                               synthetic=args.synthetic, rebuild=args.rebuild_panel)
    print(f"panel: {len(meta):,} rows, {len(cols)} characteristics, "
          f"{meta['eom'].min():%Y-%m} to {meta['eom'].max():%Y-%m}  [{time.time()-t0:.0f}s]")
    print(f"models: {', '.join(args.models)}; lightgbm: {HAVE_LGB}, torch: {HAVE_TORCH}, device: {args.device}")

    if args.timing:
        timing_run(meta, X, cols, args)
        return
    if args.y_check:
        target_check(meta, OUT / "stage1_target.csv")
        return
    run(meta, X, cols, chars[chars["characteristic"].isin(cols)], args)
    if args.worker:
        return
    evaluate(args)


if __name__ == "__main__":
    main()
