# How to run this

Four things can be run, in increasing order of what they need:

| | Needs | Time |
|---|---|---|
| **The decay results**: the three-window table and the panel regression | two free CSV files | about a second |
| **The forecasting section**: expanding-window fits of six models | the same two files | about ten minutes |
| **The CRSP rebuild**: two anomalies from security-level data | a WRDS subscription | about ten minutes |
| **The paper**: the write-up as HTML and PDF | Quarto, and the above | about a minute |

---

## Environment

A virtual environment is a private folder holding one Python and its packages, so
installing things for this project cannot break the rest of the machine, and the rest of
the machine cannot break this project.

```bash
mkdir -p ~/venvs
/usr/bin/python3 -m venv ~/venvs/wrds
source ~/venvs/wrds/bin/activate
pip install --upgrade pip
pip install wrds pandas pyarrow matplotlib scikit-learn jupyter
```

The prompt now starts with `(wrds)`. Every new terminal window needs the activation line
again:

```bash
source ~/venvs/wrds/bin/activate
```

Forgetting it is the usual cause of `ModuleNotFoundError`. `which python` should print a
path inside `~/venvs/wrds`; if it prints `/usr/bin/python3` or something with `anaconda`
in it, the environment is not active.

The decay results need only `pandas`, `numpy` and `matplotlib`; the forecasting section
adds `scikit-learn`. The cluster-robust regression is written out in NumPy, so there is
no `statsmodels` to install.

---

## Data

Both files are free, from <https://www.openassetpricing.com> (Data tab), and go in
`data/`:

- **`PredictorLSretWide.csv`**: *Featured Portfolio Return Datasets → Monthly portfolio
  returns → "Monthly long-short returns of 212 predictors following OPs (wide csv)"*
- **`SignalDoc.csv`**: *Documentation → "Signal Documentation csv file"*

`data/` is gitignored. The files are not mine to redistribute, and the download takes a
minute.

---

## The decay results, no WRDS needed

```bash
python 01_first_figure.py inspect     # what the raw files actually contain
python 01_first_figure.py figure      # one anomaly, publication date marked
python 02_decay_panel.py              # the three-window decay, all signals
```

Writes `output/result1_summary.txt`, `output/result1_windows.csv` (one row per signal)
and `output/fig02_decay.png`.

`inspect` exists because a new data release can rename a column. If it shows names the
scripts do not expect, edit the CONFIG block at the top of the script. The estimator was
checked against synthetic data with a known decay built in and recovered the true
coefficients to within 0.75 standard errors.

---

## The forecasting section, no WRDS needed

```bash
python 05_ml_panel.py                 # fit everything; about ten minutes
python 05_ml_panel.py figures         # redraw the figures from the saved outputs
```

Writes `output/ml_summary.txt`, `output/ml_r2.csv`, `output/ml_dm.csv`,
`output/ml_portfolio.csv`, `output/ml_importance.csv`, `output/ml_decay_curve.csv`,
`output/ml_tuning.csv` and `output/fig05_*.png`. All of them are committed, so the paper
renders without refitting.

---

## The CRSP rebuild, WRDS subscription required

```bash
python 03_crsp_pull.py --check        # which library is freshest; downloads nothing
python 03_crsp_pull.py                # the monthly panel, by decade, to parquet
python 04_build_anomalies.py          # rebuild Size and Mom12m, score them
```

The first connection asks for a WRDS username and password. Nothing is echoed while the
password is typed; that is normal. WRDS then offers to create a `.pgpass` file; accept,
and later runs connect without asking. **No credential is written into any file in this
repository**; the username can also be supplied as the `WRDS_USERNAME` environment
variable.

The pull is restartable: one parquet per decade, and a rerun skips whatever is already
cached. Expect about 3.9 million rows and 97 MB. It ends with two sanity checks that would
catch a broken screen immediately: the count of distinct US common stocks in 2000, and
the equal-weighted mean annual return 1963 to 2024.

`04_build_anomalies.py` writes `output/rebuild_summary.txt`, `output/rebuild_Size.csv`,
`output/rebuild_Mom12m.csv` and `output/fig04_rebuild.png`. All four are committed, so the
replication check is readable without a subscription.

---

## The paper

```bash
quarto install tinytex                # once, for the PDF
./paper/publish.sh
```

That renders `paper/alpha-decay.qmd` to HTML and PDF and copies both into `docs/`, which
GitHub Pages serves at <https://tugsuz.github.io/alpha-decay/>. See
[`paper/README.md`](paper/README.md) for the details. `quarto render paper/alpha-decay.qmd
--profile course` renders the version without the forecasting section.

---

## Troubleshooting

**`ModuleNotFoundError`**: the virtual environment is not active. `source
~/venvs/wrds/bin/activate`.

**`quarto render` says "Jupyter is not available in this Python installation"**: Quarto
found a different interpreter from the active one. Set `QUARTO_PYTHON` to the right one:
`export QUARTO_PYTHON="$(which python)"`. `quarto check jupyter` prints the one it found.

**A WRDS query hangs**: the monthly table is large and the screen runs server-side. The
per-decade cache means a killed run picks up from the last complete decade.

