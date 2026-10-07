# Audit evidence tree

Index of the evidence behind the claims in [`../../README.md`](../../README.md),
[`../../EVALUATION.md`](../../EVALUATION.md) and
[`../OPTIMIZATION_NOTES.md`](../OPTIMIZATION_NOTES.md).

| Directory | Contents | Regenerable? |
| --- | --- | --- |
| [`evaluation/`](./evaluation/) | Frozen `raw_results.json` (per-fact GPU run), `summary.json` (historical derived statistics), `verification.json` (file-hash snapshot) | New statistics can be derived offline into `run/`; fresh raw measurements need a GPU run. |
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

A 39-update cap approximates the old target-fitting budget, but does not establish
exact reproduction across later source or deployed-build changes. Keep the corrected
semantics for current runs and use a separate output directory.

A live GPU run is required either way; there is no offline path to regenerating
this evidence. Re-run `python test_live.py --label <new-name> --profile context
--extra-paraphrases` to produce current-build numbers under a new label, then
update `EVALUATION.md` and the README tables to match.

## Verifying the manifest

```bash
cd prototype
python verify_manifest.py --update   # writes a separate audit/run/hash_snapshot.json
python verify_manifest.py --manifest audit/run/hash_snapshot.json
```

Without `--manifest`, comparison uses the historical verification file; changed
source hashes are expected after this review.

The manifest covers selected source files, the benchmark manifest, and the frozen
evaluation raw and summary files. Hash agreement establishes file integrity only;
it does not establish tests passed or identify the producing GPU build. Older
`local_source_sha256` fields describe the caller's checkout, which may differ from
the deployed backend. New live runs record the backend's own source identity.
