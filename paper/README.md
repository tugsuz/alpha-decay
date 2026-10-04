# paper/

`alpha-decay.qmd` is the paper. It is the source for the HTML page the repository points
at and for the PDF next to it. The code that produces each number sits in the document,
folded on the HTML page and hidden in the PDF.

## Rendering

```bash
source ~/venvs/wrds/bin/activate
python -m pip install jupyter                     # once; Quarto runs the code through it
export QUARTO_PYTHON="$(which python)"            # pin Quarto to THIS interpreter

quarto render paper/alpha-decay.qmd --to html     # → paper/_output/alpha-decay.html
quarto render paper/alpha-decay.qmd --to pdf      # needs a LaTeX distribution
quarto render paper/alpha-decay.qmd               # both, per _quarto.yml
```

`QUARTO_PYTHON` matters on a machine with more than one Python. Quarto picks an
interpreter by its own rules; pointing it at the one that holds `pandas`, `numpy`,
`matplotlib`, `scikit-learn` and `jupyter` removes the guesswork. `quarto check jupyter`
prints the one it found, which is the first thing to look at if a render fails with
`ModuleNotFoundError` or `Jupyter is not available in this Python installation`.

To make it permanent, add the export to `~/.zshrc` with the path written out:

```bash
echo 'export QUARTO_PYTHON="$HOME/venvs/wrds/bin/python"' >> ~/.zshrc
```

Requirements:

- **Quarto 1.4 or later**: the document uses inline `` `{python} ...` `` expressions,
  which arrived in 1.4. `quarto --version` to check.
- **Python** with `pandas`, `numpy`, `matplotlib`, `scikit-learn` and `jupyter`.
- **PDF only:** a TeX installation. `quarto install tinytex` is the least painful route.
- **Data:** `data/PredictorLSretWide.csv` and `data/SignalDoc.csv`, both free from
  openassetpricing.com. No WRDS subscription is needed to render this document; the CRSP
  and forecasting sections read committed result files.

The course version, without the forecasting section:

```bash
quarto render paper/alpha-decay.qmd --profile course
```

`_quarto-course.yml` swaps the abstract; the blocks marked `unless-profile="course"` in
the source are dropped.

## Publishing it

```bash
./paper/publish.sh
git add docs && git commit -m "Publish the paper" && git push
```

That renders to `paper/_output/`, copies the single self-contained HTML to
`docs/index.html` and the PDF to `docs/alpha-decay.pdf`, and drops a `.nojekyll` beside
them. GitHub Pages serves `docs/` on the `main` branch at
<https://tugsuz.github.io/alpha-decay/>.

Enable it once: repo **Settings → Pages → Source: Deploy from a branch → main → /docs**.

The rendered HTML and PDF are committed. The render needs the two Chen and Zimmermann
files, which are not redistributable and so are not in the repository, so no CI job could
rebuild the page. Committing the output is what makes the paper readable by someone who
only has the link.

## The rule this document is built on

No number in the prose is typed by hand. Every figure in the text is either computed by
a chunk in the document or read out of a file in `output/`. The last chunk asserts that
the three-window means recomputed here equal the ones `02_decay_panel.py` wrote; if the
script and the paper drift apart, the render fails.

## Converting to a notebook

```bash
quarto convert paper/alpha-decay.qmd      # → alpha-decay.ipynb, runnable cell by cell
```

The `.qmd` stays the source of truth: it is plain text, so `git diff` is readable and
merges work. The `.ipynb` is a by-product and is gitignored.

