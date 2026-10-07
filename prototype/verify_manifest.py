"""Verify, and optionally regenerate, prototype/audit/evaluation/verification.json.

Without ``--update`` this is a read-only check: it recomputes the SHA-256 of every
tracked artifact and compares it against the stored manifest, exiting non-zero if
any hash is stale, missing, or unrecorded. The reported status is derived from
that comparison, never assumed.

With ``--update`` it writes a separate current hash snapshot. Hash agreement does
not establish runtime validation or which source revision produced saved GPU
measurements. Historical verification manifests are never overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent
EVAL_DIR = BASE_DIR / "audit" / "evaluation"
MANIFEST_PATH = EVAL_DIR / "verification.json"

FILES_TO_HASH = [
    BASE_DIR / "modal_app.py",
    BASE_DIR / "editing_optimizations.py",
    BASE_DIR / "layer_selection.py",
    BASE_DIR / "local_probe.py",
    BASE_DIR / "run_experiments.py",
    BASE_DIR / "prepare_benchmark.py",
    BASE_DIR / "analyze_schemes.py",
    BASE_DIR / "analyze_batch.py",
    BASE_DIR / "error_analysis.py",
    BASE_DIR / "export_doc.py",
    BASE_DIR / "verify_manifest.py",
    BASE_DIR / "test_backend.py",
    BASE_DIR / "test_optimizations.py",
    BASE_DIR / "test_live.py",
    BASE_DIR / "test_live_backend.py",
    BASE_DIR / "test_benchmarks.py",
    BASE_DIR / "test_tools.py",
    BASE_DIR / "requirements.txt",
    BASE_DIR / "frontend" / "package.json",
    BASE_DIR / "frontend" / "package-lock.json",
    BASE_DIR / "data" / "benchmark_manifest.json",
    BASE_DIR / "audit" / "evaluation" / "summary.json",
    BASE_DIR / "audit" / "evaluation" / "raw_results.json",
    REPO_ROOT / "README.md",
    REPO_ROOT / "EVALUATION.md",
    BASE_DIR / "README.md",
    BASE_DIR / "OPTIMIZATION_NOTES.md",
    BASE_DIR / "THIRD_PARTY_NOTICES.md",
    BASE_DIR / "audit" / "README.md",
    BASE_DIR / "audit" / "generate-report.mjs",
    BASE_DIR / "audit" / "generate-e2e-report.mjs",
    BASE_DIR / "audit" / "verify-optimization.mjs",
    BASE_DIR / "frontend" / "tests" / "component_harness.tsx",
    BASE_DIR / "frontend" / "vite.config.ts",
    BASE_DIR / "frontend" / "test_ui_buttons.js",
    BASE_DIR / "frontend" / "complete_test_protocol.js",
    BASE_DIR / "frontend" / "capture_browser_screenshots.js",
]
FILES_TO_HASH.extend(sorted(path for path in (BASE_DIR / "frontend" / "src").rglob("*")
                            if path.is_file()))
FILES_TO_HASH.extend(sorted((BASE_DIR / "frontend" / "tests").glob("*.mjs")))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def relative_key(path: Path) -> str:
    """Repo-root-relative POSIX path, so manifest keys are portable across platforms."""
    for root in (REPO_ROOT, BASE_DIR):
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return path.name


def current_hashes():
    hashes, missing = {}, []
    for path in FILES_TO_HASH:
        key = relative_key(path)
        if path.exists():
            hashes[key] = sha256(path)
        else:
            missing.append(key)
    return hashes, missing


def load_stored_hashes(path=MANIFEST_PATH):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        stored = json.load(f)
    # Earlier manifests used Windows separators and mixed path roots; normalise
    # so old keys can still be compared against the current ones.
    return {k.replace("\\", "/"): v for k, v in stored.get("file_sha256", {}).items()}


def build_report(stored_hashes, current):
    rows = []
    for key in sorted(set(stored_hashes) | set(current)):
        stored, actual = stored_hashes.get(key), current.get(key)
        if stored is None:
            status = "UNRECORDED"
        elif actual is None:
            status = "MISSING"
        elif stored == actual:
            status = "MATCH"
        else:
            status = "STALE"
        rows.append(
            {
                "file": key,
                "status": status,
                "stored_sha256": stored,
                "actual_sha256": actual,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update",
        action="store_true",
        help="write a current hash-only snapshot, without claiming tests were run",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                        help="Stored hashes to compare against (historical manifest by default)")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "audit" / "run" / "hash_snapshot.json",
                        help="New hash-only snapshot path used by --update")
    args = parser.parse_args()

    current, missing = current_hashes()
    rows = build_report(load_stored_hashes(args.manifest), current)
    mismatches = [row for row in rows if row["status"] != "MATCH"]

    width = max((len(row["file"]) for row in rows), default=0)
    for row in rows:
        print(f"{row['status']:<11} {row['file']:<{width}}  {(row['actual_sha256'] or '-')[:12]}")
    for key in missing:
        print(f"{'ABSENT':<11} {key}")

    if not mismatches and not missing and not args.update:
        print(f"\nHASHES_MATCH: all {len(rows)} artifacts match the stored manifest; runtime tests were not evaluated.")
        return 0

    if mismatches or missing:
        print(
            f"\nHASH_CHECK_FAILED: {len(mismatches)} of {len(rows)} artifacts differ"
            f"{f' and {len(missing)} are absent' if missing else ''}."
        )

    if not args.update:
        print("--update writes a separate current hash snapshot; it does not rerun or validate historical experiments.")
        return 1
    if missing:
        print("Snapshot refused: required artifacts are missing.")
        return 1
    if args.output.resolve() == MANIFEST_PATH.resolve():
        print("Snapshot refused: the historical verification manifest must be preserved.")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_key = "prototype/data/benchmark_manifest.json"
    verification = {
        "snapshot_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project": "Interactive Visual Analytics for Human-in-the-Loop Knowledge Editing in Large Language Models",
        "institution": "Mohan Babu University, Tirupati (Batch A8-12)",
        "status": "HASH_SNAPSHOT_ONLY",
        "scope": "Current artifact hashes only. No runtime checks were performed by this script.",
        "historical_measurements": {
            "producing_build_provenance": "unknown unless recorded in each result's run_identity; never inferred from current hashes",
            "raw_results_sha256": current.get("prototype/audit/evaluation/raw_results.json"),
            "summary_sha256": current.get("prototype/audit/evaluation/summary.json"),
        },
        "benchmark_manifest_sha256": current.get(manifest_key),
        "previous_manifest_differences": mismatches,
        "verification_report": rows,
        "file_sha256": current,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2)
    print(f"\nCurrent hash-only snapshot written with {len(current)} hashes: {args.output}")
    print(f"{len(mismatches)} superseded entries recorded under previous_manifest_differences.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
