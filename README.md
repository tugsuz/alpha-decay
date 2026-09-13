# Alpha Decay: the half-life of a market edge

How fast does a published equity-market anomaly stop working once it is public?

Between the 1980s and today, several hundred cross-sectional return predictors have been
documented in the finance literature. If those edges are real risk premia, publication
should not change them. If they are mispricing, publication should destroy them — because
publication is precisely the moment when enough capital learns about them to trade them
away. McLean and Pontiff (2016) measured this on 97 anomalies and found returns roughly
26% lower out of sample and 58% lower after publication.

This project reruns that question on the current universe and through the end of the data,
and then asks the part the original paper left open: **not how much decays, but how fast,
and to what floor.**

## The design

Each signal's history is cut into three windows using the publishing paper's own dates:

| window | definition | what it isolates |
|---|---|---|
| 1. in-sample | up to the paper's `SampleEndYear` | the edge as originally reported |
| 2. post-sample, pre-publication | sample end → publication year | discovered, not yet public |
| 3. post-publication | publication year onward | public, and being traded on |

The middle window is what makes this a test rather than a description. If decay is caused
by *publication*, window 2 should look like window 1 and only window 3 should fall. If it
is caused by the original sample being lucky — overfitting, data mining, selection — then
window 2 is already lower before anyone outside the authors knew. The two declines
separate the two explanations.

## Results

`python 02_decay_panel.py` produces the headline table.

<!-- RESULT 1 GOES HERE once the data is downloaded and the script has run.
     Paste output/result1_summary.txt, and state the two declines in the first line
     of this section so a reader sees a number before they see a method. -->

Estimation: equal-weighted across signals, and separately a panel regression with signal
fixed effects and standard errors clustered by date, since in any given month every
long-short portfolio is exposed to the same market. The cluster-robust covariance is
written out in `02_decay_panel.py` rather than called from a library.

## What comes next

- **Decay rate and half-life per signal.** A three-window average says the edge fell; it
  does not say how quickly, or whether it fell to zero. Fitting a state-space model to each
  signal's post-publication path returns a decay rate, a half-life, and a long-run floor —
  each with a standard error rather than a curve fitted by eye.
- **Zero or a floor?** Grossman and Stiglitz (1980) implies the floor should be positive:
  if an anomaly's return went to exactly zero, nobody would be paid to arbitrage it and it
  would reappear. Estimating the floor is a test of that.
- **Staggered treatment.** Publication arrives at a different date for every signal, so the
  publication effect is estimated with Callaway–Sant'Anna and Sun–Abraham estimators rather
  than the two-way fixed-effects specification the original literature used.

## Data

Two public files from the Chen–Zimmermann Open Source Asset Pricing dataset
(<https://www.openassetpricing.com>, Download tab), placed in `data/`:

    data/PredictorLSretWide.csv    monthly long-short return, one column per signal
    data/SignalDoc.csv             one row per signal: acronym, journal, year, sample dates

`data/` is gitignored. The files are large and are not mine to redistribute; the download
takes a minute. No WRDS subscription is required for anything in this repository.

## Running it

```bash
python 01_first_figure.py inspect   # what the raw files actually contain
python 01_first_figure.py figure    # one anomaly, publication date marked
python 02_decay_panel.py            # Result 1: the three-window decay, all signals
```

Requires `pandas`, `numpy`, `matplotlib`. Nothing else.

## Notes on method

Decisions that change the answer are recorded in `DECISIONS.md` rather than buried in the
code: which signals are excluded and why, how the windows are assigned when a paper's
sample end and publication year coincide, and whether returns are treated as percent or
decimal. Signals categorised as placebos by the original authors are excluded, and a signal
must have at least 24 monthly observations in *every* window to enter the average — a
two-year window 2 should not be represented by three months of data.

## Reference

Chen, A. Y. and T. Zimmermann (2022), "Open Source Cross-Sectional Asset Pricing",
*Critical Finance Review*.
McLean, R. D. and J. Pontiff (2016), "Does Academic Research Destroy Stock Return
Predictability?", *Journal of Finance* 71(1), 5–32.
