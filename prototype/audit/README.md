# Audit evidence tree

Index of the evidence behind the claims in [`../../README.md`](../../README.md),
[`../../EVALUATION.md`](../../EVALUATION.md) and
[`../OPTIMIZATION_NOTES.md`](../OPTIMIZATION_NOTES.md).

| Directory | Contents | Regenerable? |
| --- | --- | --- |
| [`evaluation/`](./evaluation/) | `raw_results.json` (per-fact GPU run), `summary.json` (derived statistics), `verification.json` (SHA-256 manifest) | `summary.json` yes — `python run_experiments.py --mode analyze-only`. `raw_results.json` needs a GPU run. |
| [`development/`](./development/) | CLI outputs from the exploratory phase: batch/scheme comparisons, single-edit probes, the correlation inputs behind `../README.md` | Needs a GPU run. |
| [`live/`](./live/) | Per-model live GPU request/response records for ROME and MEMIT, with restored-probe checks | Needs a GPU run. |
| [`optimization/`](./optimization/) | Context-v3 development trials (including the retained failed ones) and browser evidence | Needs a GPU run. |
| [`before/`](./before/) | Frozen pre-audit source snapshot | No — intentionally frozen. |
| [`references/`](./references/) | Instructions for cloning the pinned third-party repos (not vendored) | Clone script in its README. |

## Frozen-build caveat

The results in `evaluation/`, `development/`, `live/` and `optimization/` were
produced by a build whose optimizer loop applied **one fewer gradient update than
requested**: `v_num_grad_steps = 40` ran 39 updates, because the loop broke on its
final iteration before calling `backward()`/`step()`.

That off-by-one is **fixed** in `editing_optimizations.py` — 40 now means 40 — so
the current code does not reproduce these frozen numbers exactly. The fix was
deliberately not reverted, because the off-by-one was a genuine defect.

To reproduce the frozen results with the fixed code, run the profile with an
explicit budget of **39** (the exact equivalent of the old build's 40):

```python
from editing_optimizations import OPTIMIZATION_PROFILES, configure_context
OPTIMIZATION_PROFILES["context_v3"]["minimum_gradient_steps"] = 39
```

A live GPU run is required either way; there is no offline path to regenerating
this evidence. Re-run `python test_live.py --label <new-name> --profile context
--extra-paraphrases` to produce current-build numbers under a new label, then
update `EVALUATION.md` and the README tables to match.

## Verifying the manifest

```bash
cd prototype
python verify_manifest.py            # read-only; exits non-zero if anything is stale
python verify_manifest.py --update   # accept the current hashes
```

The manifest covers the source, the benchmark manifest and the two derived
statistics files. It does **not** cover the frozen GPU evidence above, since those
hashes are recorded inside each `audit/optimization/*/result.json` as
`local_source_sha256`.
