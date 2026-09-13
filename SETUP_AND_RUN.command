#!/bin/bash
# Double-click this file in Finder. It opens Terminal and does the setup for you.
# Every command is echoed before it runs, so you can see exactly what is happening.

set -u
cd "$(dirname "$0")"
echo "================================================================"
echo " alpha-decay setup"
echo " working folder: $(pwd)"
echo "================================================================"

VENV="$HOME/venvs/alpha"

if [ ! -d "$VENV" ]; then
  echo
  echo "--> Creating a virtual environment at $VENV"
  echo "    (a private folder holding one Python and its packages,"
  echo "     so nothing here can break your Anaconda install)"
  echo "    \$ /usr/bin/python3 -m venv $VENV"
  /usr/bin/python3 -m venv "$VENV" || { echo "FAILED to create venv"; read -r; exit 1; }
else
  echo
  echo "--> Virtual environment already exists at $VENV"
fi

echo
echo "--> Activating it"
echo "    \$ source $VENV/bin/activate"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "    python is now: $(which python)"

echo
echo "--> Installing packages (skipped if already present)"
echo "    \$ pip install --quiet --upgrade pip pandas matplotlib"
pip install --quiet --upgrade pip pandas matplotlib || { echo "pip FAILED"; read -r; exit 1; }
echo "    done."

mkdir -p data

if [ ! -f data/SignalDoc.csv ] || ! ls data/*LSret*.csv >/dev/null 2>&1; then
  echo
  echo "================================================================"
  echo " MISSING DATA -- one manual step, then run this file again"
  echo "================================================================"
  echo
  echo " 1. Open  https://www.openassetpricing.com/data/   (October 2025 release)"
  echo " 2. Download TWO things into this folder's  data/  subfolder:"
  echo "      - the WIDE long-short portfolio returns csv"
  echo "        (named something like PredictorLSretWide.csv)"
  echo "      - SignalDoc.csv"
  echo " 3. Double-click this file again."
  echo
  echo " The data/ folder is here:"
  echo "   $(pwd)/data"
  echo
  open "$(pwd)/data" 2>/dev/null || true
  echo "Press Enter to close."
  read -r
  exit 0
fi

echo
echo "================================================================"
echo " Running: python 01_first_figure.py inspect"
echo "================================================================"
echo
python 01_first_figure.py inspect

echo
echo "================================================================"
echo " Read the output above properly."
echo
echo " Then, when you have chosen a signal:"
echo "   1. open 01_first_figure.py, set SIGNAL = \"...\" near the top"
echo "   2. in Terminal:"
echo "        source $VENV/bin/activate"
echo "        cd \"$(pwd)\""
echo "        python 01_first_figure.py figure"
echo "================================================================"
echo
echo "Press Enter to close."
read -r
