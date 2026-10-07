# Code review and fixes

Review date: 2026-10-07. Starting revision: `5582275`.

The review covered the active backend, optimization adapters, frontend, benchmark
pipeline, offline analysis, document export, and verification helpers. Frozen
source snapshots and measured GPU records were retained as historical evidence.

## Confirmed fixes

- Reject incomplete or blank per-neighborhood answers before model work, and use
  the corresponding answer and ordered result for each frontend row, including
  repeated prompt text.
- Restore original parameter gradient flags and remove partially registered
  probe hooks after failures.
- Use the same complete, finite cosine policy in frontend and Python, including
  six-layer GPT-J MEMIT windows and float32 rounding tolerance near one.
- Keep missing telemetry and partial evaluation results unknown. Measured zero
  variance and zero cosine activity remain zero.
- Prevent layer controls from changing an in-flight request, and clear previous
  results before a new edit or comparison begins.
- Bind benchmark recovery to model, manifest, exact fact subset, driver source,
  deployed backend source and available model revision. Save failed observations
  and stop when a restoration check fails. Smoke runs use development facts.
- Require a complete-source content pin for dataset regeneration; reject stale
  manifests, duplicate facts, split leakage, invalid labels and smaller samples.
- Report missing-data denominators, true Jaccard overlap, small KL values, and
  the direction of statistically significant differences.
- Compensate for the pinned upstream MEMIT loop's final evaluation-only iteration
  in `standard-budget-v2`. A configured 41-iteration upstream loop permits 40
  updates, matching the local context profile's cap. Early stopping still applies.
- Validate the PCA projection used for small neighborhood samples in live tests.
- Preserve original documents during abstract export and retain paragraph/run
  formatting rather than forcing justification.
- Write regenerated browser, benchmark and report artifacts under `audit/run`.
  Hash snapshots state `HASH_SNAPSHOT_ONLY`; report generation does not certify
  current runtime behavior or assign current source hashes to historical runs.
- Correct documentation that claimed telemetry superiority, catastrophic-failure
  prevention, isolated consistency benefits, universal restoration, or exact
  reproduction of measurements from a different build.

## Verification

- `python -m unittest discover -s . -p 'test_*.py' -v`: **69 CPU tests passed**.
  Backend route tests use controlled model fixtures, not loaded GPU models.
- `npm run build`: TypeScript and Vite production build passed.
- Frontend browser audit: **23 fixture groups passed**, zero runtime errors.
- `node audit/verify-optimization.mjs`: four checks of preserved historical
  optimization records passed; this script executes no GPU edits.
- Historical raw results and summary remained byte-identical:
  - raw: `fdadaabdc641423299c6847b3810cea2d1e33e51fda37866d8b8524db6e98193`
  - summary: `5b51a13f3612a90c37a811e05c7cdc0bca664af6a89832af35e6ac6de19fbf5b`

## Limits and reproduction

The updated backend was not deployed, and GPU edits were not rerun in this
review. The historical ten-fact pilot does not establish telemetry superiority
or an incremental consistency benefit. Older raw records lack the actual
deployed source identity; their producing build cannot be inferred from current
checkout hashes. The original documents, presentations and demo launchers were
left untouched.

From `prototype`, derive statistics and inspect current file integrity separately:

```bash
python run_experiments.py --mode analyze-only --raw-results audit/evaluation/raw_results.json --output-dir audit/run/evaluation-reanalysis
python verify_manifest.py --update
python verify_manifest.py --manifest audit/run/hash_snapshot.json
```

A current GPU benchmark requires deploying the updated backend, which now reports
its mounted source hashes through health, and choosing a fresh output directory.
