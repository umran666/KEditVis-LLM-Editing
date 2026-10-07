"""Prepare and validate a content-pinned CounterFact subset.

Existing manifests remain historical artifacts. Regeneration requires the SHA-256
of the complete source file and a new output path; no dataset is downloaded on import.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import urllib.request

DATA_DIR = Path(__file__).resolve().parent / "data"
MANIFEST_PATH = DATA_DIR / "benchmark_manifest.json"
COUNTERFACT_URL = "https://memit.baulab.info/data/dsets/counterfact.json"
PLANNED_EVAL_SIZE = 25


def manifest_sha256(manifest: dict) -> str:
    """Hash the payload excluding its own digest, including legacy serialization."""
    payload = deepcopy(manifest)
    payload["metadata"].pop("sha256", None)
    if payload["metadata"].get("schema_version", 1) == 1:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
    else:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest(manifest: dict) -> str:
    """Reject stale hashes, duplicate facts, leakage and incomplete prompt labels."""
    metadata = manifest.get("metadata", {})
    if metadata.get("schema_version", 1) not in (1, 2):
        raise ValueError("Unsupported benchmark manifest schema.")
    actual = manifest_sha256(manifest)
    if metadata.get("sha256") != actual:
        raise ValueError("Benchmark manifest SHA-256 does not match its contents.")
    if metadata.get("dataset_name") != "CounterFact":
        raise ValueError("This pipeline only supports CounterFact manifests.")
    identifiers, facts = set(), set()
    for split in ("dev_facts", "eval_facts"):
        entries = manifest.get(split)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{split} must be a nonempty list.")
        for entry in entries:
            case_id = entry.get("case_id")
            if isinstance(case_id, bool) or not isinstance(case_id, int) or case_id < 0:
                raise ValueError(f"Invalid case_id in {split}: {case_id}")
            if case_id in identifiers:
                raise ValueError(f"Duplicate case_id across benchmark splits: {case_id}")
            identifiers.add(case_id)
            for field in ("relation_id", "subject", "prompt", "target_new", "target_true"):
                if not isinstance(entry.get(field), str) or not entry[field].strip():
                    raise ValueError(f"Case {case_id} has missing {field}.")
            identity = (entry["relation_id"].strip().casefold(),
                        " ".join(entry["subject"].split()).casefold())
            if identity in facts:
                raise ValueError(f"Duplicate underlying fact across benchmark splits: {identity}")
            facts.add(identity)
            if entry["target_new"].strip() == entry["target_true"].strip():
                raise ValueError(f"Case {case_id} is a no-op edit.")
            for field in ("paraphrase_prompts", "neighborhood_prompts", "generation_prompts"):
                prompts = entry.get(field)
                if (not isinstance(prompts, list) or not prompts
                        or any(not isinstance(p, str) or not p.strip() for p in prompts)):
                    raise ValueError(f"Case {case_id} has missing or invalid {field}.")
            targets = entry.get("neighborhood_targets")
            if (not isinstance(targets, list) or len(targets) != len(entry["neighborhood_prompts"])
                    or any(not isinstance(t, str) or not t.strip() for t in targets)):
                raise ValueError(f"Case {case_id} has incomplete neighborhood targets.")
    if metadata.get("planned_eval_size") != len(manifest["eval_facts"]):
        raise ValueError("Manifest evaluation size differs from its declared size.")
    if metadata.get("dev_size") != len(manifest["dev_facts"]):
        raise ValueError("Manifest development size differs from its declared size.")
    if metadata.get("schema_version") == 2:
        digest = metadata.get("source_sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Pinned source SHA-256 is required for schema 2.")
    return actual


def decode_counterfact(source: bytes, expected_sha256: str) -> list:
    """Use the JSON parser for arbitrary whitespace, escapes and Unicode."""
    if hashlib.sha256(source).hexdigest() != expected_sha256.lower():
        raise ValueError("CounterFact source SHA-256 does not match the supplied pin.")
    records = json.loads(source.decode("utf-8-sig"))
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("CounterFact source must be a JSON array of records.")
    return records


def create_manifest(records: list, source_sha256: str, source_url: str,
                    eval_size: int = PLANNED_EVAL_SIZE) -> dict:
    if isinstance(eval_size, bool) or not isinstance(eval_size, int) or eval_size <= 0:
        raise ValueError("eval_size must be a positive integer.")
    dev_records, eval_records = [], []
    for record in records:
        request = record["requested_rewrite"]
        is_dev = record["case_id"] == 0 or "eiffel" in request["subject"].casefold()
        if not is_dev and len(eval_records) >= eval_size:
            continue
        neighbors = record["neighborhood_prompts"][:3]
        entry = {
            "case_id": record["case_id"], "dataset": "CounterFact",
            "relation_id": request["relation_id"], "subject": request["subject"],
            "prompt": request["prompt"], "target_new": request["target_new"]["str"],
            "target_true": request["target_true"]["str"],
            "paraphrase_prompts": record["paraphrase_prompts"][:2],
            "neighborhood_prompts": neighbors,
            # CounterFact neighbors share the original object by construction;
            # this matches its upstream evaluator, not an unrelated-label fallback.
            "neighborhood_targets": [request["target_true"]["str"]] * len(neighbors),
            "generation_prompts": record["generation_prompts"][:2],
            "source_record_sha256": hashlib.sha256(json.dumps(
                record, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest(),
        }
        (dev_records if is_dev else eval_records).append(entry)
    if len(eval_records) != eval_size:
        raise ValueError(f"Requested {eval_size} evaluation facts, obtained {len(eval_records)}.")
    manifest = {
        "metadata": {
            "schema_version": 2, "dataset_name": "CounterFact", "source_url": source_url,
            "source_sha256": source_sha256, "planned_eval_size": eval_size,
            "dev_size": len(dev_records),
            "preprocessing": {"paraphrase_limit": 2, "neighborhood_limit": 3,
                              "generation_limit": 2, "source_record_order": "preserved"},
            "split_policy": "case_id 0 and Eiffel subjects are development; first remaining facts are evaluation",
            "metric_definitions": {
                "ES": "new candidate has lower mean target-token NLL than original on rewrite prompt",
                "PS": "fraction of paraphrases where new candidate has lower mean target-token NLL",
                "NS": "fraction of CounterFact neighborhoods preferring original over new candidate by mean NLL",
                "S": "harmonic mean of ES, PS, NS",
                "greedy": "backend free-generation target-prefix accuracy; see evaluation implementation",
            },
        }, "dev_facts": dev_records, "eval_facts": eval_records,
    }
    manifest["metadata"]["sha256"] = manifest_sha256(manifest)
    validate_manifest(manifest)
    return manifest


def build_manifest():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-url", default=COUNTERFACT_URL)
    parser.add_argument("--source-file", type=Path, help="Use a local complete source JSON file.")
    parser.add_argument("--source-sha256", required=True, help="SHA-256 of the complete pinned source file.")
    parser.add_argument("--eval-size", type=int, default=PLANNED_EVAL_SIZE)
    parser.add_argument("--output", type=Path, required=True,
                        help="New manifest path; existing evidence is never overwritten.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a new manifest path instead of overwriting {args.output}.")
    if args.source_file:
        source = args.source_file.read_bytes()
    else:
        request = urllib.request.Request(args.source_url, headers={"User-Agent": "KEditVis benchmark"})
        with urllib.request.urlopen(request, timeout=60) as response:
            source = response.read()
    records = decode_counterfact(source, args.source_sha256)
    manifest = create_manifest(records, args.source_sha256.lower(), args.source_url, args.eval_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, allow_nan=False)
    print(f"Manifest: {args.output}; SHA-256: {manifest['metadata']['sha256']}")


if __name__ == "__main__":
    build_manifest()
