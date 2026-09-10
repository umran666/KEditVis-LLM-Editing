"""Explicit live A/B evaluation. Optimizers receive no evaluation paraphrases."""
import argparse
import json
import hashlib
from pathlib import Path
import time
import requests

URL = "https://opzgameryt--keditvis-memit-web-app.modal.run"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--profile", choices=["standard", "context"], required=True)
    parser.add_argument("--layers", default="13,14,15,16,17")
    parser.add_argument("--model", default="gpt2-xl")
    parser.add_argument("--extra-paraphrases", action="store_true")
    parser.add_argument("--case", choices=["eiffel", "bigben"], default="eiffel")
    args = parser.parse_args()
    output = Path(__file__).parent / "audit" / "optimization" / args.label
    if (output / "result.json").exists():
        raise FileExistsError("Choose a new --label; existing experiment evidence will not be overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    body = {
        "model": args.model, "method": "memit", "optimization": args.profile,
        "prompt": "{} is located in the city of", "subject": "Eiffel Tower",
        "target_new": "Rome", "target_true": "Paris",
        "layers": [int(x) for x in args.layers.split(",")],
        "paraphrase_prompts": ["You can find the Eiffel Tower in the city of"],
        "neighborhood_prompts": ["The Louvre Museum is located in the city of", "Notre-Dame Cathedral is located in the city of"],
        "damage_prompts": ["The capital of France is", "Water freezes at"],
    }
    if args.case == "bigben":
        body.update(subject="Big Ben", target_true="London", target_new="Rome",
                    paraphrase_prompts=["Big Ben can be found in", "The city where Big Ben stands is", "Big Ben is a landmark in"],
                    neighborhood_prompts=["Buckingham Palace is located in the city of", "Tower Bridge is located in the city of"])
    elif args.extra_paraphrases:
        body["paraphrase_prompts"] += ["The city where the Eiffel Tower stands is", "The Eiffel Tower can be found in", "The Eiffel Tower is a landmark in", "Tourists visiting the Eiffel Tower travel to"]
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ["modal_app.py", "editing_optimizations.py"]}
    probe_body = {key: body[key] for key in ["model", "prompt", "subject"]}
    baseline = requests.post(URL + "/probe", json=probe_body, timeout=1800)
    baseline.raise_for_status()
    start = time.monotonic()
    print(f"START {args.label}", flush=True)
    response = requests.post(URL + "/edit", json=body, timeout=1800)
    data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"error": response.text}
    artifact = {"request": body, "local_source_sha256": source_hashes, "status": response.status_code, "seconds": time.monotonic() - start, "response": data}
    (output / "result.json").write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    restored = requests.post(URL + "/probe", json=probe_body, timeout=1800)
    restored.raise_for_status()
    artifact["restored"] = restored.json() == baseline.json()
    (output / "result.json").write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    assert artifact["restored"], "Baseline changed after experiment"
    response.raise_for_status()
    print(json.dumps({"metrics": data["post_edit"]["metrics"], "damage": data["damage"], "seconds": artifact["seconds"], "restored": artifact["restored"]}), flush=True)

if __name__ == "__main__":
    main()
