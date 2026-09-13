# Git, properly — enough to be fluent, not just to copy commands

Written for the exact situation: an empty `alpha-decay` repository already created on GitHub, and
starter files sitting in Google Drive.

---

## 1. The mental model (this is the part that makes git click)

A file you are working on lives in **four** places, and almost every git command is about moving it
between them.

```
   working directory  ──git add──▶  staging area  ──git commit──▶  local repo  ──git push──▶  GitHub
   (the files you edit)              (the draft of              (permanent history        (the copy
                                      your next snapshot)        on your Mac)              others see)
```

Most confusion comes from forgetting the **staging area** exists. It is a feature, not bureaucracy:
it lets you edit five files and commit only two of them, so that each commit is one coherent idea
rather than "stuff I did today".

A **commit** is a snapshot of the whole project plus a message, permanently addressable. A **branch**
is just a moving label pointing at the newest commit. `main` is a branch. That is all a branch is.

---

## 2. One-time setup on your Mac

```bash
git config --global user.name  "Mehmet Tugsuz"
git config --global user.email "mehmettugsuz@gmail.com"
git config --global init.defaultBranch main
git config --global pull.rebase false
```

The name and email are stamped into every commit and are public in a public repository — that is
normal and expected.

### Authentication

GitHub stopped accepting account passwords over HTTPS. Two options; the second is easier.

**Option A — personal access token.** github.com → Settings → Developer settings → Personal access
tokens → **Fine-grained tokens** → Generate new token. Give it access to the `alpha-decay`
repository only, with permission `Contents: Read and write`. When git asks for a password, paste the
token. Store it so you are not asked every time:

```bash
git config --global credential.helper osxkeychain
```

**Option B — GitHub CLI (recommended).**

```bash
brew install gh
gh auth login          # choose GitHub.com, HTTPS, authenticate in browser
```

This handles the token for you. Handle your own token or login — never paste a token into a chat,
a script, or a file that gets committed.

---

## 3. Getting your repository going

### Why not inside Google Drive

Drive constantly syncs; git constantly rewrites small files inside `.git/`. The two fight, and the
usual result is a corrupted `.git` folder. **Keep the repository outside Drive.**

```bash
mkdir -p ~/code
cp -R ~/"My Drive/alpha-decay-starter" ~/code/alpha-decay
cd ~/code/alpha-decay
```

### Tidy up before the first commit

```bash
mv README_draft.md README.md      # only after you have rewritten the placeholders
rm -f .Rhistory                   # stray file, not part of the project

printf 'data/\noutput/\n__pycache__/\n*.pyc\n.venv/\n.DS_Store\n' > .gitignore
```

`.gitignore` lists things git should pretend not to see. `data/` is ignored because WRDS data
cannot be redistributed and the Chen–Zimmermann files are large; the README carries download
instructions instead. `output/` is ignored because figures are regenerated from code — **except**
the one figure the README displays, which you add deliberately:

```bash
git add -f output/fig01_Mom12m.png    # -f overrides .gitignore for this one file
```

### First commit and push

```bash
git init                 # creates .git/ — this folder is now a repository
git add .                # stage everything not ignored
git status               # ALWAYS look before committing
git commit -m "Starter: research question, decision log, first-figure script"
git remote add origin https://github.com/tugsuz/alpha-decay.git
git push -u origin main
```

`origin` is just a nickname for that URL. `-u` sets it as the default, so later pushes are
`git push` with nothing after it.

If it complains that the remote already has commits (because you ticked "add a README" when
creating it):

```bash
git pull origin main --allow-unrelated-histories
# resolve any conflict, then
git push -u origin main
```

---

## 4. The daily loop

```bash
git status                          # what changed?
git diff                            # what exactly changed, line by line?
git add 01_first_figure.py          # stage the specific files this commit is about
git commit -m "Handle percent vs decimal returns in cumulative series"
git push
```

**Commit messages.** Present tense, say *what changed and why*, not "update". Good ones from this
project would be:

- `Add publication-year merge from SignalDoc`
- `Switch to log y-axis; linear axis hides the post-1990 decay`
- `Document the equal- vs value-weighting choice in README`

Commit often — after each idea that works, not once a week. A commit is free and a small commit is
easy to undo.

---

## 5. Reading the state

```bash
git log --oneline --graph -15    # recent history, compact
git show HEAD                    # exactly what the last commit changed
git diff                         # unstaged changes
git diff --staged                # staged but not yet committed
```

`HEAD` means "the commit I am currently sitting on".

---

## 6. Undoing things — the part everyone is scared of

```bash
git restore 01_first_figure.py           # discard edits to one file (NOT recoverable)
git restore --staged 01_first_figure.py  # unstage, keep the edits
git commit --amend -m "Better message"   # fix the last commit (only if NOT pushed)
git revert <hash>                        # new commit that undoes an old one — safe, keeps history
```

Rule that saves you: **once a commit is pushed, undo it with `git revert`, never by rewriting
history.** Rewriting pushed history is what breaks things.

Anything committed is recoverable, even if it looks lost — `git reflog` shows every position HEAD
has held. Things *not* yet committed are the only things you can truly lose.

---

## 7. What belongs in the repository

| Commit | Do not commit |
|---|---|
| Code, notebooks, `.gitignore`, README | Raw WRDS or CRSP data (licence forbids redistribution) |
| The one or two figures the README shows | Everything in `output/` regenerated by code |
| Derived summary tables, small enough to read | Virtualenvs, `__pycache__`, `.DS_Store` |
| Data *download* scripts and instructions | Any token, password or `.pgpass` |

If a credential ever gets committed, deleting it in a later commit is **not** enough — it stays in
history. Rotate the credential immediately.

---

## 8. A twenty-minute exercise that teaches more than reading

In `~/code/alpha-decay`, before you touch anything real:

1. `git log --oneline` — see your one commit.
2. Edit one line of `README.md`. Run `git status`, then `git diff`. Watch what each tells you.
3. `git add README.md`, then `git status` again. Notice it moved to "staged".
4. `git restore --staged README.md` — it moves back. The edit is still there.
5. `git commit -am "Test commit"`, then `git log --oneline` — two commits.
6. `git revert HEAD` — a third commit that undoes the second. History intact.

After that the four-box diagram at the top will be something you can see rather than memorise.
