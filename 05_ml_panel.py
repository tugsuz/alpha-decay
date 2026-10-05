"""
alpha-decay, step 5: can a model pick, in real time, which published signals will pay
next month, and does machine learning do it better than a linear regression?

    python 05_ml_panel.py            # fit everything, about ten minutes
    python 05_ml_panel.py figures    # redraw the figures from the saved outputs

Needs the same two files as step 2, in data/:
    data/PredictorLSretWide.csv
    data/SignalDoc.csv

Writes to output/:
    ml_summary.txt        the headline table
    ml_r2.csv             out-of-sample R2 by model, pooled and by event-time bucket
    ml_dm.csv             Diebold-Mariano tests, model against model
    ml_portfolio.csv      monthly returns of the signal-selection portfolios
    ml_importance.csv     permutation importance of each feature group
    ml_decay_curve.csv    return against months since publication: raw bins and the
                          partial dependence of the fitted models
    ml_tuning.csv         every hyperparameter setting that was tried, by test year
    fig05_r2.png, fig05_decay.png, fig05_portfolio.png

--------------------------------------------------------------------------------------
THE SETUP

One row is a signal in a month. The target is that signal's long-short return next
month. The question is whether anything known this month predicts it.

Information set. A forecaster standing in month t knows:
  - which signals have been published (publication year < year of t), their documented
    sample windows and the characteristics of each paper;
  - every published signal's return history up to t, including the months before the
    paper appeared, because the data can be backfilled once the recipe is public;
  - nothing about signals that have not yet been published.
So a signal enters the panel, for training and for testing, only in years after its
publication year. That is the whole of the no-look-ahead rule and it costs most of the
pre-1995 sample, when fewer than two dozen signals were public.

Time is kept by the target month. A test year holds the forecasts whose target month
falls in that year; the model for it is fit on rows whose target month is earlier, and
an assert checks the boundary at every refit.

Models. Two benchmarks that fit nothing (a zero forecast and the signal's own historical
mean), a linear regression, an elastic net, gradient boosted trees and a small neural
network. Every model sees the same features. Hyperparameters are chosen on the last
eight years of each training sample and the model is then refit on the whole of it.

Evaluation. Expanding window, one refit per calendar year, test years 1995 to 2024.
Out-of-sample R2 against a zero forecast, as in Gu, Kelly and Xiu (2020), and against
the signal's own mean. Diebold-Mariano tests on the monthly loss differential with a
Newey-West variance. Portfolios that each month go long the fifth of signals with the
highest forecast and short the fifth with the lowest.
--------------------------------------------------------------------------------------
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "output"

PORTFOLIO_FILE = DATA / "PredictorLSretWide.csv"
SIGNALDOC_FILE = DATA / "SignalDoc.csv"

FIRST_TEST_YEAR = 1995
LAST_TEST_YEAR = 2024
VALIDATION_YEARS = 8          # last years of each training sample, for tuning
MIN_HISTORY = 60              # months of own history a row needs before it enters
NW_LAGS = 12                  # Newey-West lags for the Diebold-Mariano variance
SEED = 0


# ---------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------

def load():
    for p in (PORTFOLIO_FILE, SIGNALDOC_FILE):
        if not p.exists():
            sys.exit(f"Not found: {p}. See the note at the top of 02_decay_panel.py.")
    rets = pd.read_csv(PORTFOLIO_FILE)
    rets = rets.rename(columns={rets.columns[0]: "date"})
    rets["date"] = pd.to_datetime(rets["date"])
    doc = pd.read_csv(SIGNALDOC_FILE)
    doc = doc[doc["Cat.Signal"] == "Predictor"].dropna(subset=["Year", "SampleEndYear"])
    return rets, doc


def signal_characteristics(doc):
    """One row per signal: what the paper itself tells you, fixed at publication."""
    d = doc.set_index("Acronym")
    x = pd.DataFrame(index=d.index)
    x["pub_year"] = d["Year"].astype(int)
    x["samp_start"] = d["SampleStartYear"].astype(int)
    x["samp_end"] = d["SampleEndYear"].astype(int)
    x["sample_len"] = x["samp_end"] - x["samp_start"] + 1
    # the original paper's own evidence, as recorded by Chen and Zimmermann
    x["orig_tstat"] = pd.to_numeric(d["T-Stat"], errors="coerce")
    x["orig_return"] = pd.to_numeric(d["Return"], errors="coerce")
    x["ls_quantile"] = pd.to_numeric(d["LS Quantile"], errors="coerce")
    x["hold_months"] = pd.to_numeric(d["Portfolio Period"], errors="coerce")
    x["value_weighted"] = (d["Stock Weight"] == "VW").astype(float)
    x["discrete"] = (d["Cat.Form"] == "discrete").astype(float)
    x["top_journal"] = d["Journal"].isin(["JF", "JFE", "RFS"]).astype(float)
    x["clear_in_op"] = (d["Predictability in OP"] == "1_clear").astype(float)
    for cat in ["Accounting", "Price", "Analyst", "Trading"]:
        x[f"data_{cat.lower()}"] = (d["Cat.Data"] == cat).astype(float)
    return x


def build_panel(rets, chars):
    """Long panel with features known at t and the target r_{t+1}."""
    use = [c for c in rets.columns if c != "date" and c in chars.index]
    wide = rets.set_index("date")[use].sort_index()
    wide.index = wide.index.to_period("M")

    # market state, built from the panel itself: the equal-weighted return across the
    # signals that were public in that month. Nothing from outside the two input files.
    pub_year = chars.loc[use, "pub_year"]
    public = pd.DataFrame({c: wide.index.year > pub_year[c] for c in use}, index=wide.index)
    cs = wide.where(public).mean(axis=1)
    state = pd.DataFrame({
        "cs_ret_1": cs,
        "cs_mean_12": cs.rolling(12, min_periods=12).mean(),
        "cs_vol_12": cs.rolling(12, min_periods=12).std(),
    })

    rows = []
    for c in use:
        r = wide[c].dropna()
        if len(r) < MIN_HISTORY + 1:
            continue
        f = pd.DataFrame(index=r.index)
        f["signal"] = c
        f["ret_1"] = r
        f["mean_12"] = r.rolling(12, min_periods=12).mean()
        f["mean_36"] = r.rolling(36, min_periods=36).mean()
        f["mean_60"] = r.rolling(60, min_periods=60).mean()
        f["mean_all"] = r.expanding(min_periods=MIN_HISTORY).mean()
        f["vol_12"] = r.rolling(12, min_periods=12).std()
        mi = r.index.year * 12 + r.index.month
        f["months_since_pub"] = mi - (chars.loc[c, "pub_year"] * 12 + 1)
        f["months_since_sample_end"] = mi - (chars.loc[c, "samp_end"] * 12 + 12)
        f["post_pub"] = (f["months_since_pub"] >= 0).astype(float)
        f["in_sample"] = (f["months_since_sample_end"] <= 0).astype(float)
        f["target"] = r.shift(-1)
        rows.append(f)
    panel = pd.concat(rows)
    panel = panel.join(state, how="left")
    panel = panel.join(chars, on="signal")
    panel["year"] = panel.index.year
    # The target is next month's return, so a row's forecast date is the month after
    # its feature month. Test years, training cut-offs and the no-look-ahead assert all
    # work on the target month, never on the feature month.
    panel["target_month"] = panel.index + 1
    panel["target_year"] = panel["target_month"].dt.year
    panel.index.name = "month"
    panel = panel.reset_index()
    # a row needs 60 months of own history and a target; the universe rule is applied
    # at fit time, because it depends on the forecast year
    panel = panel.dropna(subset=["mean_60", "target", "cs_mean_12"])
    return panel


FEATURES = [
    # own history
    "ret_1", "mean_12", "mean_36", "mean_60", "mean_all", "vol_12",
    # event time
    "months_since_pub", "months_since_sample_end", "post_pub", "in_sample",
    # the paper
    "pub_year", "sample_len", "orig_tstat", "orig_return", "ls_quantile", "hold_months",
    "value_weighted", "discrete", "top_journal", "clear_in_op",
    "data_accounting", "data_price", "data_analyst", "data_trading",
    # market state
    "cs_ret_1", "cs_mean_12", "cs_vol_12",
]

GROUPS = {
    "own history": ["ret_1", "mean_12", "mean_36", "mean_60", "mean_all", "vol_12"],
    "event time": ["months_since_pub", "months_since_sample_end", "post_pub", "in_sample"],
    "original evidence": ["orig_tstat", "orig_return", "sample_len", "clear_in_op"],
    "construction": ["ls_quantile", "hold_months", "value_weighted", "discrete"],
    "data type and journal": ["data_accounting", "data_price", "data_analyst",
                              "data_trading", "top_journal", "pub_year"],
    "market state": ["cs_ret_1", "cs_mean_12", "cs_vol_12"],
}


# ---------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------

class Impute:
    """Median imputation fitted on the training rows. Trees take NaN natively; the
    linear models and the network do not."""

    def fit(self, X):
        self.med = np.nanmedian(X, axis=0)
        return self

    def transform(self, X):
        X = X.copy()
        idx = np.where(np.isnan(X))
        X[idx] = np.take(self.med, idx[1])
        return X


def fit_linear(Xtr, ytr):
    imp = Impute().fit(Xtr)
    m = LinearRegression().fit(imp.transform(Xtr), ytr)
    return lambda X: m.predict(imp.transform(X)), {}


def fit_enet(Xtr, ytr, Xva, yva, log):
    imp = Impute().fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    A, B = sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xva))
    best = None
    for alpha in [0.001, 0.01, 0.1]:
        for l1 in [0.1, 0.5, 0.9]:
            m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000).fit(A, ytr)
            mse = np.mean((yva - m.predict(B)) ** 2)
            log.append(("enet", f"alpha={alpha},l1={l1}", mse))
            if best is None or mse < best[0]:
                best = (mse, alpha, l1)
    _, alpha, l1 = best
    X = np.vstack([Xtr, Xva]); y = np.concatenate([ytr, yva])
    imp = Impute().fit(X); sc = StandardScaler().fit(imp.transform(X))
    m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000).fit(sc.transform(imp.transform(X)), y)
    return (lambda Z: m.predict(sc.transform(imp.transform(Z)))), {"alpha": alpha, "l1": l1}


def fit_gbrt(Xtr, ytr, Xva, yva, log):
    best = None
    for depth in [2, 3]:
        for lr in [0.05, 0.1]:
            m = HistGradientBoostingRegressor(
                max_depth=depth, learning_rate=lr, max_iter=400, l2_regularization=1.0,
                min_samples_leaf=200, early_stopping=False, random_state=SEED,
            )
            # pick the number of trees on the validation sample by staged prediction
            m.fit(Xtr, ytr)
            curve = [np.mean((yva - p) ** 2) for p in m.staged_predict(Xva)]
            n_best = int(np.argmin(curve)) + 1
            mse = curve[n_best - 1]
            log.append(("gbrt", f"depth={depth},lr={lr},trees={n_best}", mse))
            if best is None or mse < best[0]:
                best = (mse, depth, lr, n_best)
    _, depth, lr, n_best = best
    X = np.vstack([Xtr, Xva]); y = np.concatenate([ytr, yva])
    m = HistGradientBoostingRegressor(
        max_depth=depth, learning_rate=lr, max_iter=n_best, l2_regularization=1.0,
        min_samples_leaf=200, early_stopping=False, random_state=SEED,
    ).fit(X, y)
    return m.predict, {"depth": depth, "lr": lr, "trees": n_best}


def fit_mlp(Xtr, ytr, Xva, yva, log):
    imp = Impute().fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    A, B = sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xva))
    best = None
    for alpha in [0.01, 0.1]:
        preds = []
        for seed in range(2):
            m = MLPRegressor(hidden_layer_sizes=(16,), alpha=alpha, learning_rate_init=0.001,
                             batch_size=1024, max_iter=80, early_stopping=True,
                             validation_fraction=0.1, random_state=seed).fit(A, ytr)
            preds.append(m.predict(B))
        mse = np.mean((yva - np.mean(preds, axis=0)) ** 2)
        log.append(("mlp", f"alpha={alpha},hidden=16,seeds=2", mse))
        if best is None or mse < best[0]:
            best = (mse, alpha)
    _, alpha = best
    X = np.vstack([Xtr, Xva]); y = np.concatenate([ytr, yva])
    imp = Impute().fit(X); sc = StandardScaler().fit(imp.transform(X))
    Z = sc.transform(imp.transform(X))
    models = [MLPRegressor(hidden_layer_sizes=(16,), alpha=alpha, learning_rate_init=0.001,
                           batch_size=1024, max_iter=80, early_stopping=True,
                           validation_fraction=0.1, random_state=seed).fit(Z, y)
              for seed in range(3)]

    def predict(W):
        W = sc.transform(imp.transform(W))
        return np.mean([m.predict(W) for m in models], axis=0)
    return predict, {"alpha": alpha}


# ---------------------------------------------------------------------------------
# the expanding window
# ---------------------------------------------------------------------------------

def run(panel):
    preds = []
    tuning = []
    fitted = {}      # year -> {model: predict function}, kept for the diagnostics
    t0 = time.time()
    for y in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        known = panel[panel["pub_year"] < y]            # the universe known at the start of y
        train = known[known["target_year"] <= y - 1]      # every target observed by December y-1
        test = known[known["target_year"] == y]           # forecasts for January to December y
        if len(test) == 0:
            continue
        # the boundary: nothing in training may have a target month at or after the first
        # test month
        assert train["target_month"].max() < test["target_month"].min(), y
        va_mask = train["target_year"] >= y - VALIDATION_YEARS
        Xtr, ytr = train.loc[~va_mask, FEATURES].to_numpy(), train.loc[~va_mask, "target"].to_numpy()
        Xva, yva = train.loc[va_mask, FEATURES].to_numpy(), train.loc[va_mask, "target"].to_numpy()
        Xall, yall = train[FEATURES].to_numpy(), train["target"].to_numpy()
        Xte = test[FEATURES].to_numpy()

        log = []
        f_ols, _ = fit_linear(Xall, yall)
        f_enet, h_enet = fit_enet(Xtr, ytr, Xva, yva, log)
        f_gbrt, h_gbrt = fit_gbrt(Xtr, ytr, Xva, yva, log)
        f_mlp, h_mlp = fit_mlp(Xtr, ytr, Xva, yva, log)
        fitted[y] = {"ols": f_ols, "enet": f_enet, "gbrt": f_gbrt, "mlp": f_mlp}
        for name, setting, mse in log:
            tuning.append({"test_year": y, "model": name, "setting": setting, "val_mse": mse})

        out = test[["signal", "month", "target_month", "target", "months_since_pub", "mean_all"]].copy()
        out["zero"] = 0.0
        out["own_mean"] = test["mean_all"].to_numpy()
        out["ols"] = f_ols(Xte)
        out["enet"] = f_enet(Xte)
        out["gbrt"] = f_gbrt(Xte)
        out["mlp"] = f_mlp(Xte)
        preds.append(out)
        print(f"  {y}: train {len(train):>7,} rows, test {len(test):>5,} rows, "
              f"{test['signal'].nunique():>3} signals, gbrt {h_gbrt}, enet {h_enet}, "
              f"mlp {h_mlp}  [{time.time()-t0:5.0f}s]")
    return pd.concat(preds, ignore_index=True), pd.DataFrame(tuning), fitted


MODELS = ["zero", "own_mean", "ols", "enet", "gbrt", "mlp"]
LABELS = {"zero": "zero forecast", "own_mean": "own historical mean", "ols": "OLS",
          "enet": "elastic net", "gbrt": "gradient boosted trees", "mlp": "neural net (16 units)"}


def r2_oos(y, yhat, bench=None):
    """1 - SSE(model) / SSE(benchmark). The benchmark is zero unless given."""
    b = np.zeros_like(y) if bench is None else bench
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - b) ** 2)


def nw_var(d, lags):
    """Newey-West long-run variance of the mean of d."""
    d = d - d.mean()
    T = len(d)
    v = np.sum(d * d) / T
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        v += 2 * w * np.sum(d[L:] * d[:-L]) / T
    return v / T


def diebold_mariano(P, a, b):
    """Monthly average loss differential across signals, then a t-statistic with a
    Newey-West variance. Positive means model b has the lower loss."""
    la = (P["target"] - P[a]) ** 2
    lb = (P["target"] - P[b]) ** 2
    d = (la - lb).groupby(P["target_month"]).mean().to_numpy()
    return d.mean() / np.sqrt(nw_var(d, NW_LAGS))


def portfolios(P, model):
    """Each month, long the top fifth of signals by forecast, short the bottom fifth."""
    rows = []
    for m, g in P.groupby("target_month"):
        if len(g) < 10:
            continue
        q = pd.qcut(g[model].rank(method="first"), 5, labels=False)
        top, bot = g.loc[q == 4, "target"].mean(), g.loc[q == 0, "target"].mean()
        rows.append({"month": m, "top": top, "bottom": bot, "spread": top - bot,
                     "all": g["target"].mean()})
    return pd.DataFrame(rows).set_index("month")


def perf(x):
    """Annualised mean, volatility, Sharpe, and the t-statistic of the monthly mean."""
    m, s = x.mean(), x.std(ddof=1)
    return {"mean_pm": m, "vol_pm": s, "sharpe": (m / s) * np.sqrt(12) if s > 0 else np.nan,
            "t": m / (s / np.sqrt(len(x)))}


def importance(panel, P, fitted):
    """Permutation importance by feature group: the rise in pooled test MSE when a
    group's columns are shuffled within each test year, for the tree model and OLS."""
    rng = np.random.default_rng(SEED)
    res = {}
    for model in ["gbrt", "ols"]:
        base = np.sum((P["target"] - P[model]) ** 2)
        for grp, cols in GROUPS.items():
            sse = 0.0
            for y, fs in fitted.items():
                test = panel[(panel["pub_year"] < y) & (panel["target_year"] == y)]
                X = test[FEATURES].to_numpy().copy()
                for c in cols:
                    j = FEATURES.index(c)
                    X[:, j] = rng.permutation(X[:, j])
                sse += np.sum((test["target"].to_numpy() - fs[model](X)) ** 2)
            res[(model, grp)] = (sse - base) / len(P)
    out = pd.Series(res).unstack(0)
    out.columns = [f"{c}_mse_increase" for c in out.columns]
    return out.sort_values("gbrt_mse_increase", ascending=False)


def decay_curve(panel, fitted):
    """Return against months since publication.

    Raw: the mean target in bins of event time over the post-1995 test rows, with a
    standard error. Fitted: the partial dependence of the final-year models, which
    moves months_since_pub (and the indicators that depend on it) across a grid and
    averages the prediction over the training rows' other features."""
    y = LAST_TEST_YEAR
    train = panel[(panel["pub_year"] < y) & (panel["target_year"] <= y - 1)]
    X0 = train[FEATURES].to_numpy()
    j_pub = FEATURES.index("months_since_pub")
    j_post = FEATURES.index("post_pub")
    grid = np.arange(-120, 241, 12)
    rows = []
    sub = train.sample(min(len(train), 20000), random_state=SEED)[FEATURES].to_numpy()
    for g in grid:
        X = sub.copy()
        X[:, j_pub] = g
        X[:, j_post] = float(g >= 0)
        rows.append({"months_since_pub": g,
                     "pd_gbrt": fitted[y]["gbrt"](X).mean(),
                     "pd_ols": fitted[y]["ols"](X).mean()})
    pdep = pd.DataFrame(rows).set_index("months_since_pub")
    return pdep


def raw_bins(P):
    bins = np.arange(-120, 253, 12)
    cut = pd.cut(P["months_since_pub"], bins, right=False)
    g = P.groupby(cut, observed=True)["target"]
    out = pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.size()), "n": g.size()})
    out.index = [iv.left + 6 for iv in out.index]
    out.index.name = "months_since_pub"
    return out


# ---------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------

def fig_r2(r2, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    models = ["own_mean", "ols", "enet", "gbrt", "mlp"]
    names = ["own historical\nmean", "OLS", "elastic net", "gradient\nboosted trees",
             "neural net\n(16 units)"]
    vals = [100 * r2.loc[m, "r2_vs_zero"] for m in models]
    ax.bar(names, vals, color="#333333", width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02 if v >= 0 else v - 0.02, f"{v:+.2f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("out-of-sample R², %")
    ax.set_title("Predicting next month's long-short return, 1995 to 2024", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    lo, hi = min(vals + [0]), max(vals + [0])
    ax.set_ylim(lo - 0.15, hi + 0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def fig_decay(raw, pdep, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(raw.index / 12, raw["mean"], yerr=1.96 * raw["se"], fmt="o", ms=4,
                color="#555555", capsize=2, lw=0.8, label="mean return in event-time bin (95% band)")
    ax.plot(pdep.index / 12, pdep["pd_gbrt"], color="#1f77b4", lw=2,
            label="gradient boosted trees, partial dependence")
    ax.plot(pdep.index / 12, pdep["pd_ols"], color="#d62728", lw=1.5, ls="--",
            label="OLS, partial dependence")
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("years since publication")
    ax.set_ylabel("long-short return, %/month")
    ax.set_title("Return against time since publication", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def fig_portfolio(monthly, path):
    """Cumulative value of the top-fifth portfolios against holding every published signal."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, color, ls in [("gbrt", "#1f77b4", "-"), ("ols", "#d62728", "--"),
                             ("own_mean", "#777777", ":")]:
        cum = (1 + monthly[f"{model}_top"] / 100).cumprod()
        ax.plot(cum.index.to_timestamp(), cum, color=color, ls=ls, lw=1.5,
                label=f"top fifth by {LABELS[model]}")
    cum = (1 + monthly["all"] / 100).cumprod()
    ax.plot(cum.index.to_timestamp(), cum, color="black", lw=1.2, alpha=0.7,
            label="all published signals, equal-weighted")
    ax.set_yscale("log")
    ax.set_ylabel("cumulative value of one dollar, log scale")
    ax.set_title("Choosing among published signals, 1995 to 2024", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def redraw():
    """Rebuild the three figures from the saved csv files, without refitting."""
    r2 = pd.read_csv(OUT / "ml_r2.csv", index_col=0)
    curve = pd.read_csv(OUT / "ml_decay_curve.csv", index_col=0)
    monthly = pd.read_csv(OUT / "ml_portfolio.csv", index_col=0)
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    fig_r2(r2, OUT / "fig05_r2.png")
    fig_decay(curve.dropna(subset=["mean"]), curve.dropna(subset=["pd_gbrt"]), OUT / "fig05_decay.png")
    fig_portfolio(monthly, OUT / "fig05_portfolio.png")
    print("redrew fig05_*.png from the saved outputs")


# ---------------------------------------------------------------------------------

def main():
    rets, doc = load()
    chars = signal_characteristics(doc)
    panel = build_panel(rets, chars)
    print(f"panel: {len(panel):,} signal-months, {panel['signal'].nunique()} signals")
    print("expanding window:")
    P, tuning, fitted = run(panel)

    # out-of-sample R2, pooled and by event-time bucket
    y = P["target"].to_numpy()
    r2 = pd.DataFrame(index=MODELS)
    r2["r2_vs_zero"] = [r2_oos(y, P[m].to_numpy()) for m in MODELS]
    r2["r2_vs_own_mean"] = [r2_oos(y, P[m].to_numpy(), P["own_mean"].to_numpy()) for m in MODELS]
    buckets = {"0 to 3 years": (0, 36), "3 to 10 years": (36, 120), "over 10 years": (120, 10**6)}
    for name, (lo, hi) in buckets.items():
        sub = P[(P["months_since_pub"] >= lo) & (P["months_since_pub"] < hi)]
        r2[f"r2_vs_zero: {name}"] = [r2_oos(sub["target"].to_numpy(), sub[m].to_numpy()) for m in MODELS]
    r2["n_test_rows"] = len(P)

    # Diebold-Mariano
    pairs = [("zero", "ols"), ("own_mean", "ols"), ("ols", "enet"), ("ols", "gbrt"),
             ("ols", "mlp"), ("gbrt", "mlp"), ("zero", "gbrt"), ("own_mean", "gbrt")]
    dm = pd.DataFrame([{"benchmark": a, "model": b, "dm_t": diebold_mariano(P, a, b)}
                       for a, b in pairs])

    # portfolios
    port = {m: portfolios(P, m) for m in ["own_mean", "ols", "enet", "gbrt", "mlp"]}
    prow = []
    for m, df in port.items():
        prow.append({"portfolio": f"{LABELS[m]}: top fifth minus bottom fifth", **perf(df["spread"])})
        prow.append({"portfolio": f"{LABELS[m]}: top fifth", **perf(df["top"])})
    prow.append({"portfolio": "all published signals, equal-weighted", **perf(port["ols"]["all"])})
    ptable = pd.DataFrame(prow).set_index("portfolio")
    monthly = pd.concat({m: df[["top", "bottom", "spread"]] for m, df in port.items()}, axis=1)
    monthly.columns = [f"{a}_{b}" for a, b in monthly.columns]
    monthly["all"] = port["ols"]["all"]

    imp = importance(panel, P, fitted)
    pdep = decay_curve(panel, fitted)
    raw = raw_bins(P)
    curve = raw.join(pdep, how="outer")

    OUT.mkdir(exist_ok=True)
    r2.to_csv(OUT / "ml_r2.csv")
    dm.to_csv(OUT / "ml_dm.csv", index=False)
    monthly.to_csv(OUT / "ml_portfolio.csv")
    imp.to_csv(OUT / "ml_importance.csv")
    curve.to_csv(OUT / "ml_decay_curve.csv")
    tuning.to_csv(OUT / "ml_tuning.csv", index=False)
    fig_r2(r2, OUT / "fig05_r2.png")
    fig_decay(raw, pdep, OUT / "fig05_decay.png")
    fig_portfolio(monthly, OUT / "fig05_portfolio.png")

    lines = []
    add = lines.append
    add("STEP 5 --- predicting next month's long-short return across published signals")
    add("=" * 78)
    add(f"test years {FIRST_TEST_YEAR} to {LAST_TEST_YEAR}, {len(P):,} signal-months, "
        f"{P['signal'].nunique()} signals; one refit per year, expanding window")
    add(f"features: {len(FEATURES)}; validation: last {VALIDATION_YEARS} years of each training sample")
    add("")
    add("Out-of-sample R2 (%), pooled over all test rows:")
    add(f"  {'model':<26}{'vs zero':>10}{'vs own mean':>14}{'0-3y':>9}{'3-10y':>9}{'10y+':>9}")
    for m in MODELS:
        r = r2.loc[m]
        add(f"  {LABELS[m]:<26}{100*r['r2_vs_zero']:>+10.2f}{100*r['r2_vs_own_mean']:>+14.2f}"
            f"{100*r['r2_vs_zero: 0 to 3 years']:>+9.2f}{100*r['r2_vs_zero: 3 to 10 years']:>+9.2f}"
            f"{100*r['r2_vs_zero: over 10 years']:>+9.2f}")
    add("")
    add("Diebold-Mariano t-statistics (positive: the model beats the benchmark):")
    for _, r in dm.iterrows():
        add(f"  {LABELS[r['model']]:<26} vs {LABELS[r['benchmark']]:<22} t = {r['dm_t']:+.2f}")
    add("")
    add("Portfolios of signals, monthly, %:")
    add(f"  {'portfolio':<58}{'mean':>7}{'vol':>7}{'Sharpe':>8}{'t':>7}")
    for name, r in ptable.iterrows():
        add(f"  {name:<58}{r['mean_pm']:>7.3f}{r['vol_pm']:>7.3f}{r['sharpe']:>8.2f}{r['t']:>7.2f}")
    add("")
    add("Permutation importance, rise in pooled test MSE when a group is shuffled:")
    for grp, r in imp.iterrows():
        add(f"  {grp:<26} trees {r['gbrt_mse_increase']:+.4f}   OLS {r['ols_mse_increase']:+.4f}")
    (OUT / "ml_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwrote ml_*.csv, ml_summary.txt and fig05_*.png to {OUT}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "figures":
        redraw()
    else:
        main()
