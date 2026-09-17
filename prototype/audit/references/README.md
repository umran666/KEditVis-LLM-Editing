# Reference clones

This directory holds local checkouts of the third-party repositories that were
inspected while designing the context-robust MEMIT profile. **They are not
vendored** — the directory is gitignored, so a fresh clone will not contain them.

They are listed here so the attribution trail in
[`OPTIMIZATION_NOTES.md`](../../OPTIMIZATION_NOTES.md) and
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) stays reproducible.

## Populating

```bash
cd prototype/audit/references

git clone https://github.com/zjunlp/EasyEdit.git
git -C EasyEdit checkout 14cea8245f06715684592ab55184939b99d70784

git clone https://github.com/jianghoucheng/AlphaEdit.git
git -C AlphaEdit checkout b84624f44dfe8fc6cd9e41df916c44124a0c46dc

git clone https://github.com/jianghoucheng/AnyEdit.git
git -C AnyEdit checkout 057a77f185f7ffb55818f6bd9add37f43bb447e7
```

## What each is used for

| Repository | Revision | Role |
| --- | --- | --- |
| EasyEdit (CORE) | `14cea8245f06715684592ab55184939b99d70784` | Source of the context-consistency idea that `editing_optimizations.py` adapts. |
| AlphaEdit | `b84624f44dfe8fc6cd9e41df916c44124a0c46dc` | Null-space projection; inspected and **deferred** (this app restores each edit, so sequential-edit projection is separate work). |
| AnyEdit | `057a77f185f7ffb55818f6bd9add37f43bb447e7` | Autoregressive decomposition for extended targets; inspected and **deferred** (the reported failure is a one-token target). |

MEMIT itself is not cloned here: it is cloned into the Modal image at the pinned
commit `80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b` (see `MEMIT_COMMIT` in
`modal_app.py`), and production code imports only that package plus the local
`editing_optimizations.py`. No EasyEdit dependency tree is installed.
