"""
Frozen layer selection policies for knowledge editing (ROME and MEMIT).

Guarantees:
- Uses ONLY information available in the unedited model before editing begins
  (cosine similarity and residual stream signals from baseline forward probe).
- Does NOT look at post-edit metrics or test-set success scores.
- ROME always edits exactly one layer.
- MEMIT always edits a contiguous window of K layers (K=5 for GPT-2-XL, K=6 for GPT-J-6B).
- Provides matched static presets (published upstream defaults) and seeded random baselines.
"""

from typing import List, Dict, Any, Optional
import math
import random


def _model_spec(model_name: str, method: str, total_layers: Optional[int] = None):
    """Reject unknown models or invalid budgets instead of substituting GPT-2."""
    models = {"gpt2-xl": (48, 5), "eleutherai/gpt-j-6b": (28, 6)}
    try:
        default_layers, memit_window = models[model_name.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc
    method = method.lower()
    if method not in ("rome", "memit"):
        raise ValueError(f"Unknown method: {method}")
    n_layers = default_layers if total_layers is None else total_layers
    if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers <= 0:
        raise ValueError("total_layers must be a positive integer.")
    window = 1 if method == "rome" else memit_window
    if n_layers < window:
        raise ValueError(f"{method.upper()} requires at least {window} layers.")
    return n_layers, window


def get_static_preset(model_name: str, method: str) -> List[int]:
    """Returns the published hyperparameter layer preset for the model and method."""
    _model_spec(model_name, method)
    is_gptj = "gpt-j" in model_name.lower()
    if method.lower() == "rome":
        return [5] if is_gptj else [17]
    elif method.lower() == "memit":
        return [3, 4, 5, 6, 7, 8] if is_gptj else [13, 14, 15, 16, 17]
    raise ValueError(f"Unknown method: {method}")


def get_random_scheme(
    model_name: str,
    method: str,
    seed: int,
    total_layers: Optional[int] = None,
) -> List[int]:
    """
    Returns a deterministic, seeded pseudo-random layer selection matched in layer count
    to the method's static preset.
    """
    n_layers, window_size = _model_spec(model_name, method, total_layers)
    rng = random.Random(seed)

    if method.lower() == "rome":
        # Exactly 1 layer
        return [rng.randint(0, n_layers - 1)]
    elif method.lower() == "memit":
        start = rng.randint(0, n_layers - window_size)
        return list(range(start, start + window_size))
    raise ValueError(f"Unknown method: {method}")


def select_layers_telemetry(
    layer_signals: List[Dict[str, Any]],
    model_name: str,
    method: str,
    total_layers: Optional[int] = None,
) -> List[int]:
    """
    Frozen telemetry-guided selection policy.
    Uses only baseline probe signals (cosine similarity between MLP input and output).
    This is a cosine-based heuristic, not a measurement of information gain.

    ROME: Selects argmin |cosine_similarity|.
    MEMIT: Selects the contiguous window of length K minimizing sum(|cosine_similarity|).
    """
    n_layers, window_size = _model_spec(model_name, method, total_layers)
    if not layer_signals:
        raise ValueError("layer_signals must not be empty.")

    signal_map = {}
    for signal in layer_signals:
        layer = signal.get("layer")
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < n_layers:
            raise ValueError(f"Invalid telemetry layer: {layer}")
        if layer in signal_map:
            raise ValueError(f"Duplicate telemetry layer: {layer}")
        cosine = signal.get("cosine_similarity")
        if (isinstance(cosine, bool) or not isinstance(cosine, (int, float))
                or not math.isfinite(cosine) or abs(cosine) > 1.000001):
            raise ValueError(f"Missing or invalid cosine similarity at layer {layer}.")
        signal_map[layer] = abs(cosine)
    missing = sorted(set(range(n_layers)) - signal_map.keys())
    if missing:
        raise ValueError(f"Incomplete telemetry; missing layers: {missing}")

    if method.lower() == "rome":
        best_layer = 0
        min_cos = float("inf")
        for layer in range(n_layers):
            val = signal_map[layer]
            if val < min_cos:
                min_cos = val
                best_layer = layer
        return [best_layer]

    elif method.lower() == "memit":
        best_start = 0
        min_sum = float("inf")
        for start in range(n_layers - window_size + 1):
            window_sum = sum(
                signal_map[start + j] for j in range(window_size)
            )
            if window_sum < min_sum:
                min_sum = window_sum
                best_start = start
        return list(range(best_start, best_start + window_size))

    raise ValueError(f"Unknown method: {method}")
