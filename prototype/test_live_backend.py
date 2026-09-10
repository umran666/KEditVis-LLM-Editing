"""Opt-in real GPU integration checks; this uses the configured paid Modal service."""
import argparse
import json
import math
from pathlib import Path
import time

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://opzgameryt--keditvis-memit-web-app.modal.run")
    parser.add_argument("--model", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"], required=True)
    parser.add_argument("--methods", nargs="+", default=["rome", "memit"])
    args = parser.parse_args()
    folder = Path(__file__).parent / "audit" / "live" / args.model.replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)
    verified_path = folder / "verified.json"
    verified_path.unlink(missing_ok=True)
    summary = []

    def call(name, path, body=None, expected=200):
        start = time.monotonic()
        print(f"START {args.model} {name}", flush=True)
        response = requests.request("POST" if body is not None else "GET", args.url + path, json=body, timeout=1800)
        try:
            data = response.json()
        except ValueError:
            data = {"text": response.text}
        record = {"url": args.url + path, "request": body, "status": response.status_code,
                  "seconds": round(time.monotonic() - start, 2), "response": data}
        (folder / f"{name}.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        assert response.status_code == expected, f"{name}: HTTP {response.status_code}: {str(data)[:1000]}"
        summary.append({"name": name, "seconds": record["seconds"], "status": response.status_code})
        (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"PASS {name} ({record['seconds']}s)", flush=True)
        return data

    fact = {"model": args.model, "prompt": "{} is located in the city of", "subject": "Eiffel Tower",
            "target_new": "Rome", "target_true": "Paris",
            "paraphrase_prompts": ["You can find the Eiffel Tower in the city of"],
            "neighborhood_prompts": ["The Louvre Museum is located in the city of", "Notre-Dame Cathedral is located in the city of"],
            "damage_prompts": ["The capital of France is", "Water freezes at"]}
    probe = {key: fact[key] for key in ["model", "prompt", "subject"]}
    n_layers = 48 if args.model == "gpt2-xl" else 28

    def check_signals(signals):
        assert [s["layer"] for s in signals] == list(range(n_layers))
        for signal in signals:
            assert math.isfinite(signal["cosine_similarity"])
            for kind in ["top_tokens", "last_top_tokens"]:
                assert len(signal[kind]) == 5
                assert all(0 <= t["prob"] <= 1 for t in signal[kind])

    def check_result(result):
        assert all(0 <= result["metrics"][key] <= 1 for key in ["ES", "PS", "NS", "S"])
        assert math.isfinite(result["damage"]["kl_divergence"])
        assert len(result["neighborhood"]) == 2
        for row in result["neighborhood"]:
            assert row["pre_text"] and row["post_text"]
            assert math.isfinite(row["hidden_state_drift"]) and row["hidden_state_drift"] >= 0
            assert row["projection_method"] == "joint-tsne"
            assert all(math.isfinite(x) for point in row["projection"].values() for x in point)

    health = call("health", f"/health?model={args.model}")
    assert health["model"] == args.model and health["n_layers"] == n_layers
    baseline = call("baseline-probe", "/probe", probe)
    check_signals(baseline["layer_signals"])
    generation = call("baseline-generate", "/generate", probe)
    assert generation["generation"].startswith("Eiffel Tower")
    call("reject-invalid-layer", "/edit", {**fact, "layers": [n_layers]}, 400)

    for method in args.methods:
        layers = ([17] if method == "rome" else [13, 14, 15, 16, 17]) if n_layers == 48 else ([5] if method == "rome" else [3, 4, 5, 6, 7, 8])
        result = call(f"{method}-edit", "/edit", {**fact, "method": method, "layers": layers})
        assert result["pre_edit"]["layer_signals"] == baseline["layer_signals"]
        check_signals(result["post_edit"]["layer_signals"])
        check_result({**result, "metrics": result["post_edit"]["metrics"]})
        restored = call(f"{method}-restored-probe", "/probe", probe)
        assert restored == baseline, f"{method}: baseline signals changed after edit"
        alternate = [layers[0] - 1] if method == "rome" else layers[:-1]
        comparison = call(f"{method}-compare", "/compare", {**fact, "method": method, "schemes": [layers, alternate]})
        assert len(comparison["schemes"]) == 2
        assert comparison["baseline"]["layer_signals"] == baseline["layer_signals"]
        for scheme in comparison["schemes"]:
            check_result(scheme)
            check_signals(scheme["layer_signals"])
        assert call(f"{method}-compare-restored-probe", "/probe", probe) == baseline
        assert call(f"{method}-restored-generate", "/generate", probe) == generation
        print(f"VERIFIED {args.model} {method}: edit, two-scheme comparison, exact baseline signal and text restoration", flush=True)

    verified_path.write_text(json.dumps({"model": args.model, "methods": args.methods,
        "request_checks": len(summary), "baseline_restored": True,
        "editing_commit": health["editing_commit"]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
