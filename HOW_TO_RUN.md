# How to actually run any of this

Written assuming no prior venv experience. Everything happens in **Terminal**
(Applications -> Utilities -> Terminal, or Cmd+Space then type "Terminal").

## What a venv is, in one sentence

A private folder holding one Python and its packages, so installing things for this project
cannot break Anaconda, and Anaconda being broken cannot break this project.

## One-time setup

```bash
mkdir -p ~/venvs
/usr/bin/python3 -m venv ~/venvs/wrds
source ~/venvs/wrds/bin/activate
pip install --upgrade pip
pip install wrds pandas pyarrow matplotlib jupyter
```

You will know it worked because your prompt now starts with `(wrds)`.

## Every time you open a new Terminal window

```bash
source ~/venvs/wrds/bin/activate
```

That is the step people forget. If you get `ModuleNotFoundError`, this is almost always why.
Check with `which python` — it must print `/Users/mehmettugsuz/venvs/wrds/bin/python`.
If it prints `/usr/bin/python3` or something with `anaconda` in it, the venv is not active.

## Running the WRDS check

```bash
source ~/venvs/wrds/bin/activate
cd ~/"My Drive/wrds_setup"
python explore_wrds.py
```

First run only: it asks for your WRDS **username**, then your **password** (nothing is echoed
while you type — that is normal, keep typing and press Enter). Then it offers to create a
`.pgpass` file; answer **y**. From then on it connects without asking.

Output goes to the screen and to `wrds_entitlements.txt` in the same folder.

## Running the first figure — no WRDS needed

```bash
source ~/venvs/wrds/bin/activate
cd ~/"My Drive/alpha-decay-starter"
mkdir -p data
```

Download from <https://www.openassetpricing.com> (Download tab) into that `data/` folder:

- the **wide long-short returns** file (`PredictorLSretWide.csv` or whatever the current release
  calls it)
- **`SignalDoc.csv`**

Then:

```bash
python 01_first_figure.py inspect     # READ this output properly
python 01_first_figure.py figure      # writes output/fig01_<signal>.png
python 02_decay_panel.py              # RESULT 1: three-window decay, all signals
```

`02_decay_panel.py` needs only pandas, numpy and matplotlib — the cluster-robust
regression is written out by hand, so there is no statsmodels install to do. It has been
tested against synthetic data with a known decay built in (8 draws, recovered the true
coefficients to within 0.75 standard errors), so if it errors on the real files it is
almost certainly a column-name change in a new data release, not a bug in the estimator.
Fix the CONFIG block at the top and rerun.

If `inspect` shows different column names, edit the CONFIG block at the top of the script.
That is expected, not a failure — the file names and schema change between data releases.

## Putting it on GitHub

Claude cannot do this: creating the repository and pushing needs your GitHub credentials, which
it does not handle. Three steps, yours.

**1. Create an empty repository** at <https://github.com/new>
   - name: `alpha-decay`
   - public
   - do **not** tick "add a README" (you already have one)

**2. Copy the starter out of Google Drive.** Git and Drive sync fight each other over the `.git`
folder, so the repository should live outside Drive:

```bash
mkdir -p ~/code
cp -R ~/"My Drive/alpha-decay-starter" ~/code/alpha-decay
cd ~/code/alpha-decay
mv README_draft.md README.md          # after you have rewritten the placeholders
```

**3. Initialise and push:**

```bash
cd ~/code/alpha-decay
printf 'data/\noutput/\n__pycache__/\n.venv/\n.DS_Store\n' > .gitignore
git init -b main
git add .
git commit -m "First figure: post-publication decay for one anomaly"
git remote add origin https://github.com/tugsuz/alpha-decay.git
git push -u origin main
```

If it asks for a password, GitHub wants a **personal access token**, not your account password:
github.com -> Settings -> Developer settings -> Personal access tokens -> Fine-grained tokens ->
Generate, with `Contents: Read and write` on this one repository. Paste it as the password.
Simpler alternative: install the GitHub CLI (`brew install gh`), run `gh auth login` once, and
authenticate in the browser.

`data/` is in `.gitignore` on purpose. WRDS data cannot be redistributed, and the
Chen–Zimmermann files are large — the download instructions are in the README instead.
