#!/usr/bin/env bash
#
# Render the paper and put it where GitHub Pages will serve it.
#
#   ./paper/publish.sh
#   git add docs && git commit -m "Publish the paper" && git push
#
# The rendered output is committed, which is not normally how build artefacts are
# treated. The reason is specific: rendering needs data/PredictorLSretWide.csv and
# data/SignalDoc.csv, which are not redistributable and therefore not in the repo, so a
# CI job could not rebuild the page. Committing the output is what makes the paper
# readable by someone who only has the link.
#
# The PDF is built BEFORE the HTML is copied into place. The page links to the PDF by
# hand, so publishing a page whose PDF is missing would ship a broken link; ordering it
# this way makes that impossible.

set -euo pipefail
cd "$(dirname "$0")"

# Quarto picks a Python by its own rules. On a machine with a conda base, a venv and a
# system python3 that is not necessarily the active one, so say which.
if [[ -z "${QUARTO_PYTHON:-}" ]]; then
  QUARTO_PYTHON="$(command -v python)"
  export QUARTO_PYTHON
fi
echo "rendering with QUARTO_PYTHON=$QUARTO_PYTHON"
echo

echo "--- PDF ---------------------------------------------------------------"
if ! quarto render alpha-decay.qmd --to pdf; then
  cat <<'MSG'

The PDF did not build, so nothing was published: the page links to it and a broken
link on the site is worse than a stale page.

Almost always this is a missing LaTeX installation. One command fixes it:

    quarto install tinytex

Then run this script again.
MSG
  exit 1
fi

echo
echo "--- HTML --------------------------------------------------------------"
quarto render alpha-decay.qmd --to html

mkdir -p ../docs
cp _output/alpha-decay.pdf  ../docs/alpha-decay.pdf
cp _output/alpha-decay.html ../docs/index.html
touch ../docs/.nojekyll          # stop GitHub Pages running the files through Jekyll

echo
echo "wrote docs/index.html        ($(du -h ../docs/index.html | cut -f1))"
echo "      docs/alpha-decay.pdf   ($(du -h ../docs/alpha-decay.pdf | cut -f1))"
echo
echo "commit both, then GitHub Pages serves them at"
echo "    https://tugsuz.github.io/alpha-decay/"
echo "    https://tugsuz.github.io/alpha-decay/alpha-decay.pdf"
echo
echo "first time only: repo Settings -> Pages -> Source: Deploy from a branch,"
echo "branch main, folder /docs."
