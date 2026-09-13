# Decisions the code cannot make for you

The script hands you a figure. These are the questions the figure raises. Answer them in writing —
one or two sentences each — before writing any more code. **Every one of these is an interview
question waiting to happen**, and the answer "that is what the script did" is the wrong answer.

Answer format: the choice, then the alternative you rejected, then why.

---

### 1. Which anomalies enter the sample?

Chen–Zimmermann carry 200+ signals of varying quality. Their `SignalDoc.csv` classifies them —
some are clearly documented predictors, some are "not replicated" or indirect.

- Do you take all of them, or only the well-documented ones?
- Do you require a minimum pre-publication sample length? How long, and why?
- Do you want spread across categories (value, momentum, profitability, investment, liquidity), or
  is an unbalanced set fine?

**Your answer:**

---

### 2. What exactly is the event date?

`SignalDoc.csv` gives a publication year. But the paper circulated as a working paper before that,
often for years.

- Publication year, or the end of the original sample period, or first working-paper circulation?
- McLean and Pontiff distinguish "out-of-sample" (after the original sample ends, before
  publication) from "post-publication". Do you keep that three-way split, or collapse it?
- What does your choice do to the interpretation? If informed traders act on working papers, using
  the publication date understates decay. Which direction does that bias your estimate?

**Your answer:**

---

### 3. Equal-weighted or value-weighted portfolio returns?

- Which do you report as the headline, and which as robustness?
- Equal weighting overweights small illiquid stocks, where the anomaly is largest and arbitrage is
  hardest. Does that make decay look faster or slower? Argue it.

**Your answer:**

---

### 4. What is the event window?

- How many months before and after publication do you show?
- Anomalies published in 2015 have ten years of post data; ones published in 1990 have thirty-five.
  Do you truncate to a common window, or use everything and accept an unbalanced panel?
- If unbalanced, the composition of the sample changes along the x-axis of your figure. Is the
  downward slope decay, or is it composition? **This is the first thing a good interviewer will
  ask about that figure.**

**Your answer:**

---

### 5. Returns in what units?

- Raw long-short returns, or alpha relative to a factor model? Which factor model?
- If raw: a decline could be a change in factor exposure rather than a change in mispricing.
- If risk-adjusted: you have imposed a model, and the result is conditional on it being right.

**Your answer:**

---

### 6. What would falsify this?

- What pattern in the data would tell you that decay is *not* happening?
- What is the most credible alternative explanation for a post-publication decline that has nothing
  to do with arbitrage?

**Your answer:**

---

When all six have written answers, the Method notes section of the README writes itself, and the
project is genuinely yours regardless of who typed the code.
