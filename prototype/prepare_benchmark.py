"""
Prepares the pinned benchmark evaluation manifest from CounterFact.
Separates development facts (case_id 0 and Eiffel Tower) from evaluation facts (case_ids 1 to 25).
Saves pinned records with SHA-256 manifest hash and documented metadata.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)
MANIFEST_PATH = DATA_DIR / "benchmark_manifest.json"

COUNTERFACT_URL = "https://memit.baulab.info/data/dsets/counterfact.json"
PLANNED_EVAL_SIZE = 25


def download_counterfact_slice(url: str, num_records: int = 30) -> list:
    print(f"Streaming {num_records} records from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    records = []
    buffer = ""
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        # Stream chunks until we have parsed num_records JSON objects
        while len(records) < num_records:
            chunk = resp.read(65536).decode("utf-8", errors="ignore")
            if not chunk:
                break
            buffer += chunk
            
            # Extract JSON objects between curly braces at root array level
            # In CounterFact, each object starts with '  {\n    "case_id":'
            while True:
                start_idx = buffer.find('{"case_id"')
                if start_idx == -1:
                    start_idx = buffer.find('{\n    "case_id"')
                if start_idx == -1:
                    start_idx = buffer.find('{\n  "case_id"')
                if start_idx == -1:
                    break
                
                # Find matching closing brace
                depth = 0
                end_idx = -1
                in_string = False
                escape = False
                for i in range(start_idx, len(buffer)):
                    ch = buffer[i]
                    if escape:
                        escape = False
                        continue
                    if ch == "\\":
                        escape = True
                        continue
                    if ch == '"':
                        in_string = not in_string
                        continue
                    if not in_string:
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                end_idx = i + 1
                                break
                
                if end_idx != -1:
                    obj_str = buffer[start_idx:end_idx]
                    try:
                        record = json.loads(obj_str)
                        records.append(record)
                        buffer = buffer[end_idx:]
                        print(f"  Parsed record {len(records)}: case_id {record.get('case_id')} ({record['requested_rewrite']['subject']})")
                        if len(records) >= num_records:
                            break
                    except json.JSONDecodeError:
                        buffer = buffer[start_idx + 1:]
                else:
                    # Need more data in buffer
                    break

    return records


def build_manifest():
    raw_records = download_counterfact_slice(COUNTERFACT_URL, num_records=30)
    
    eval_records = []
    dev_records = []
    
    for r in raw_records:
        req = r["requested_rewrite"]
        case_id = r["case_id"]
        
        # Case 0 is designated as development/sanity check fact
        is_dev = (case_id == 0 or "eiffel" in req["subject"].lower())
        
        entry = {
            "case_id": case_id,
            "dataset": "CounterFact",
            "relation_id": req["relation_id"],
            "subject": req["subject"],
            "prompt": req["prompt"],
            "target_new": req["target_new"]["str"],
            "target_true": req["target_true"]["str"],
            "paraphrase_prompts": r.get("paraphrase_prompts", [])[:2],
            "neighborhood_prompts": r.get("neighborhood_prompts", [])[:3],
            "neighborhood_targets": [req["target_true"]["str"]] * min(3, len(r.get("neighborhood_prompts", []))),
            "generation_prompts": r.get("generation_prompts", [])[:2],
        }
        
        if is_dev:
            dev_records.append(entry)
        elif len(eval_records) < PLANNED_EVAL_SIZE:
            eval_records.append(entry)

    manifest_data = {
        "metadata": {
            "dataset_name": "CounterFact",
            "source_url": COUNTERFACT_URL,
            "planned_eval_size": len(eval_records),
            "dev_size": len(dev_records),
            "metric_definitions": {
                "ES": "P(target_new) > P(target_true) on rewrite prompt (likelihood preference)",
                "PS": "P(target_new) > P(target_true) on paraphrase prompts",
                "NS": "P(neighbor_true) > P(target_new) on neighborhood prompts",
                "S": "Harmonic mean of ES, PS, NS",
                "ES_greedy": "argmax token matches target_new string exactly",
                "PS_greedy": "argmax token matches target_new string exactly on paraphrases",
                "NS_greedy": "argmax token matches neighbor_true string exactly",
            },
        },
        "dev_facts": dev_records,
        "eval_facts": eval_records,
    }

    manifest_bytes = json.dumps(manifest_data, indent=2).encode("utf-8")
    sha256_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_data["metadata"]["sha256"] = sha256_hash

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\nManifest successfully created at {MANIFEST_PATH}")
    print(f"Total evaluation facts: {len(eval_records)}")
    print(f"Total development facts: {len(dev_records)}")
    print(f"Manifest SHA-256: {sha256_hash}")


if __name__ == "__main__":
    build_manifest()
