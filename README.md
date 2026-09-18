# Alpha Decay: the half-life of a market edge

How fast does a published equity-market anomaly stop working once it is public?

**Headline result.** Across the 189 usable signals in the Chen–Zimmermann universe
(1926–2024, 173,302 signal-months), the mean monthly long-short return falls from
**0.603%** inside the publishing paper's own sample to **0.329%** after publication — a
**45% decline**. The median signal loses **53%**. McLean and Pontiff (2016) reported 58%
on 97 anomalies; this is that number, recomputed on a universe twice the size and extended
to the end of the current data.

![three-window decay](output/fig02_decay.png)

Between the 1980s and today, several hundred cross-sectional return predictors have been
documented in the finance literature. If those edges are real risk premia, publication
should not change them. If they are mispricing, publication should destroy them — because
publication is precisely the moment when enough capital learns about them to trade them
away.

This project reruns that question on the current universe and then asks the part the
original paper left open: **not how much decays, but how fast, and to what floor.**

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
window 2 is already lower before anyone outside the authors knew.

**Composition is held fixed.** A signal enters only if its paper's sample ends strictly
before publication *and* it has at least 24 monthly observations in **every** window. The
same 189 signals therefore appear in all three bars, so the decline cannot be an artefact
of the sample changing between windows.

## Results

`python 02_decay_panel.py` — runs in about a second, needs only pandas, numpy, matplotlib.

### Three windows, equal-weighted across 189 signals

| window | mean, %/month | median, %/month | mean vs. window 1 |
|---|---|---|---|
| in-sample | 0.603 | 0.507 | — |
| post-sample, pre-publication | 0.437 | 0.347 | −28% |
| post-publication | 0.329 | 0.237 | **−45%** |

The mean falls less than the median (−45% against −53%) because a minority of signals
survive more or less intact and pull the average up. Decay is not uniform: **151 of 189
(80%)** earn less after publication than in sample, and **32 (17%)** have an outright
negative mean long-short return once public.

### Panel regression

r_it = a_i + b₁·postsample_it + b₂·postpub_it + e_it

Signal fixed effects are absorbed by within-demeaning, so b₁ and b₂ come from variation
over time *within* a signal rather than from some signals being stronger than others.
Standard errors are clustered by **date** (1,188 clusters), because in any given month
every long-short portfolio is exposed to the same market. The cluster-robust covariance is
written out in `02_decay_panel.py` rather than called from a library.

| | full panel (212 signals) | matched panel (189) |
|---|---|---|
| b₁ post-sample | −0.1449 (se 0.0747, t = −1.94) | −0.1277 (se 0.0752, t = −1.70) |
| b₂ post-publication | −0.2718 (se 0.0531, **t = −5.12**) | −0.2518 (se 0.0548, t = −4.60) |
| implied decline | −24% and −45% | −21% and −42% |

### What this does *not* show

The tempting conclusion is that publication, rather than overfitting, is what kills the
edge: b₂ is sharply estimated and b₁ is not. That conclusion does not survive a test.
Imposing the linear restriction b₂ = b₁ on the same clustered covariance matrix:

| | b₂ − b₁, %/month | se | t |
|---|---|---|---|
| full panel | −0.1269 | 0.0761 | **−1.67** |
| matched 189 | −0.1241 | 0.0755 | −1.64 |

The *incremental* effect of publication, over and above the decline that had already
happened by the time the original sample ended, is **not significant at the 5% level**. So
the three-window design as run here does not separate the arbitrage channel from the
data-mining channel on this universe. The point estimates are consistent with publication
mattering; the difference is not sharp enough to claim it.

McLean and Pontiff get a cleaner separation (−26% against −58%) on 97 anomalies. This
universe is twice as large and includes weaker and less-replicated signals, which is the
obvious first explanation and is the next thing to check by re-running on their 97.

## What comes next

- **Re-run on the McLean–Pontiff 97**, to see whether the channel separation is a
  universe-composition effect.
- **Decay rate and half-life per signal.** A three-window average says the edge fell; it
  does not say how quickly, or whether it fell to zero. Fitting a state-space model to each
  signal's post-publication path returns a decay rate, a half-life and a long-run floor —
  each with a standard error rather than a curve fitted by eye.
- **Zero or a floor?** Grossman and Stiglitz (1980) implies the floor should be positive:
  if an anomaly's return went to exactly zero, nobody would be paid to arbitrage it and it
  would reappear. Estimating the floor is a test of that.
- **Staggered treatment.** Publication arrives at a different date for every signal, so the
  publication effect should be estimated with Callaway–Sant'Anna and Sun–Abraham estimators
  rather than the two-way fixed-effects specification used above and in the original
  literature.
- **Cross-section of decay speed** against arbitrage-cost proxies (Amihud illiquidity,
  size, idiosyncratic volatility), which is where the story about *why* edges die gets
  tested rather than asserted.

## Data

Two public files from the Chen–Zimmermann Open Source Asset Pricing dataset
(<https://www.openassetpricing.com>, Data tab), placed in `data/`:

    data/PredictorLSretWide.csv   monthly long-short returns, one column per signal (3.3 MB)
    data/SignalDoc.csv            one row per signal: acronym, journal, year, sample dates

- `PredictorLSretWide.csv` is *Featured Portfolio Return Datasets → Monthly portfolio
  returns → "Monthly long-short returns of 212 predictors following OPs (wide csv)"*.
- `SignalDoc.csv` is under *Documentation → "Signal Documentation csv file"*.

**Release used: October 2025, v2.0.0.** Most series run through December 2024;
option-implied-volatility predictors end December 2022. This is the first release in which
all signals are produced by the authors' Python translation rather than the original Stata
code; `ChNAnalyst`, `PriceDelayTstat` and `Recomm_ShortInterest` carry major revisions from
bug fixes in that translation, and all three are in the universe used here.

`data/` is gitignored — the files are not mine to redistribute and the download takes a
minute. **No WRDS subscription is required for anything in this repository.** A separate
strand of the project builds two or three anomalies from raw CRSP for comparison; that code
is not here yet.

## Running it

```bash
python 01_first_figure.py inspect   # what the raw files actually contain
python 01_first_figure.py figure    # one anomaly, publication date marked
python 02_decay_panel.py            # the three-window decay, all signals
```

Writes `output/result1_summary.txt`, `output/result1_windows.csv` (one row per signal) and
`output/fig02_decay.png`.

## Notes on method

Decisions that change the answer are recorded in `DECISIONS.md` rather than buried in the
code. As run: signals the original authors categorise as placebos are excluded; a signal
must have at least 24 monthly observations in every window; returns are equal-weighted
across signals; and the event date is the publication year given in `SignalDoc.csv`, not
first working-paper circulation — which, if informed traders act on working papers, makes
the estimated post-publication decline a **lower** bound on the true one.

The small-sample correction on the clustered covariance is (G/(G−1))·((n−1)/(n−k)) with
k = 2; it does not subtract the absorbed fixed effects from the degrees of freedom, which
with n = 173,302 changes the standard errors by about 0.1%.

## Reference

Chen, A. Y. and T. Zimmermann (2022), "Open Source Cross-Sectional Asset Pricing",
*Critical Finance Review*.

McLean, R. D. and J. Pontiff (2016), "Does Academic Research Destroy Stock Return
Predictability?", *Journal of Finance* 71(1), 5–32.

Grossman, S. J. and J. E. Stiglitz (1980), "On the Impossibility of Informationally
Efficient Markets", *American Economic Review* 70(3), 393–408.
