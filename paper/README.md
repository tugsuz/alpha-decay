# paper/

`alpha-decay.qmd` is the project written out: the question, the estimand, the estimator
derived in LaTeX, the code that implements it, the number, and the test that the number
survives — in that order, with the maths and the code interleaved rather than separated.

It is the source for three outputs: the HTML that the repository points at, the PDF for
the ECN 593 applied project, and (via `format: revealjs`) the slides.

## Rendering

```bash
quarto render paper/alpha-decay.qmd --to html     # → paper/_output/alpha-decay.html
quarto render paper/alpha-decay.qmd --to pdf      # needs a LaTeX distribution
quarto render paper/alpha-decay.qmd               # both, per _quarto.yml
```

Requirements:

- **Quarto ≥ 1.4** — the document uses inline `` `{python} ...` `` expressions, which
  arrived in 1.4. `quarto --version` to check; `brew install quarto` or the installer at
  quarto.org.
- **Python** with `pandas`, `numpy`, `matplotlib`, `jupyter`. The project venv already has
  the first three; `pip install jupyter` if `quarto render` complains about a kernel.
- **PDF only:** a TeX installation. `quarto install tinytex` is the least painful route.
- **Data:** `data/PredictorLSretWide.csv` and `data/SignalDoc.csv`, both free from
  openassetpricing.com. No WRDS subscription is needed to render this document — the CRSP
  section reads committed result files.

## The rule this document is built on

**No number in the prose is typed by hand.** Every figure in the text is either computed
by a chunk in the document or read out of a file in `output/`. The last chunk asserts that
the three-window means recomputed here equal the ones `02_decay_panel.py` wrote; if the
script and the paper ever drift apart, the render fails instead of quietly disagreeing.

That rule exists because this project has twice been bitten by a number that was true when
it was written and false a week later. A document that recomputes cannot go stale.

## Converting to a notebook

```bash
quarto convert paper/alpha-decay.qmd      # → alpha-decay.ipynb, runnable cell by cell
```

The `.qmd` stays the source of truth: it is plain text, so `git diff` is readable and
merges work. The `.ipynb` is a by-product and is gitignored.

## Still to do

- `references.bib` carries a header flagging that volume and page numbers need checking
  against the journals before submission. Do that before this goes to anyone.
- An event-time figure: average return by months since publication, which the two window
  dummies compress into a pair of numbers.
- The staggered difference-in-differences section is written as a plan, not a result.
  Callaway–Sant'Anna and Sun–Abraham estimates go in §8 when they exist.
