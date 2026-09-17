# Pre-audit source snapshot (frozen)

This directory is a **frozen snapshot of the source as it stood before the
first-pass audit**, kept deliberately. It is not live code — nothing imports it,
nothing builds from it, and it is expected to diverge from `prototype/`.

## Why it is here

[`../KEDITVIS_AUDIT.md`](../KEDITVIS_AUDIT.md) cites findings by line number into
the *original* files, and [`../E2E_VERIFICATION.md`](../E2E_VERIFICATION.md)
documents the replacement code that fixed them. Keeping both sides means the
before/after pair is verifiable:

| File | Role |
| --- | --- |
| `modal_app.py` | The pre-fix backend. Every `[CRITICAL]`/`[HIGH]` finding's "original location" points into this file. |
| `frontend/` | The pre-fix dashboard (`src/`, `package.json`). |

Without this snapshot, the audit's line-number citations would point into a file
that no longer exists in that form, and the findings would be unverifiable.

## Do not edit

Treat these as read-only evidence. If you need the current source, it is one
directory up. If a finding needs re-verifying, diff this snapshot against the
live file:

```bash
cd prototype
diff -u audit/before/modal_app.py modal_app.py
```
