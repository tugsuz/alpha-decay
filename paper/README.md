# paper/

`alpha-decay.qmd` is the project written out: the question, the estimand, the estimator
derived in LaTeX, the code that implements it, the number, and the test that the number
survives — in that order, with the maths and the code interleaved rather than separated.

It is the source for three outputs: the HTML that the repository points at, the PDF for
the ECN 593 applied project, and (via `format: revealjs`) the slides.

## Rendering

```bash
source ~/venvs/wrds/bin/activate
python -m pip install jupyter                     # once; Quarto runs the code through it
export QUARTO_PYTHON="$(which python)"            # pin Quarto to THIS interpreter

quarto render paper/alpha-decay.qmd --to html     # → paper/_output/alpha-decay.html
quarto render paper/alpha-decay.qmd --to pdf      # needs a LaTeX distribution
quarto render paper/alpha-decay.qmd               # both, per _quarto.yml
```

`QUARTO_PYTHON` is the line that saves an afternoon. This machine has a conda `base`, a
`wrds` venv and a system `python3`, and Quarto picks one by its own rules rather than by
which prompt you are looking at. Pointing it at the interpreter that actually holds
`pandas`, `numpy`, `matplotlib` and `jupyter` removes the guesswork. `quarto check jupyter`
prints the one it found, which is the first thing to look at if a render fails with
`ModuleNotFoundError` or `Jupyter is not available in this Python installation`.

To make it permanent, add the export to `~/.zshrc` with the path written out:

```bash
echo 'export QUARTO_PYTHON="$HOME/venvs/wrds/bin/python"' >> ~/.zshrc
```

Requirements:

- **Quarto ≥ 1.4** — the document uses inline `` `{python} ...` `` expressions, which
  arrived in 1.4. `quarto --version` to check; `brew install quarto` or the installer at
  quarto.org.
- **Python** with `pandas`, `numpy`, `matplotlib` and `jupyter`. Quarto also needs `pyyaml`,
  which comes in with `jupyter`.
- **PDF only:** a TeX installation. `quarto install tinytex` is the least painful route.
- **Data:** `data/PredictorLSretWide.csv` and `data/SignalDoc.csv`, both free from
  openassetpricing.com. No WRDS subscription is needed to render this document — the CRSP
  section reads committed result files.

## Publishing it

```bash
./paper/publish.sh
git add docs && git commit -m "Publish the paper" && git push
```

That renders to `paper/_output/`, copies the single self-contained HTML to
`docs/index.html` and drops a `.nojekyll` beside it. GitHub Pages serves `docs/` on the
`main` branch at <https://tugsuz.github.io/alpha-decay/>, which is the link that belongs
on a résumé or a LinkedIn profile — a recruiter who will not clone a repository will open
a web page.

Enable it once: repo **Settings → Pages → Source: Deploy from a branch → main → /docs**.

The rendered HTML is committed. That is not how build artefacts are normally treated, and
the reason is specific: the render needs the two Chen–Zimmermann files, which are not
redistributable and so are not in the repository, so no CI job could rebuild the page.
Committing the output is what makes the paper readable by someone who only has the link.

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
