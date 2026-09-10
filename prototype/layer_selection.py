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
import random


def get_static_preset(model_name: str, method: str) -> List[int]:
    """Returns the published hyperparameter layer preset for the model and method."""
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
    is_gptj = "gpt-j" in model_name.lower()
    n_layers = total_layers or (28 if is_gptj else 48)
    rng = random.Random(seed)

    if method.lower() == "rome":
        # Exactly 1 layer
        return [rng.randint(0, n_layers - 1)]
    elif method.lower() == "memit":
        window_size = 6 if is_gptj else 5
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
    Lower cosine similarity indicates higher MLP informational activity.

    ROME: Selects argmin |cosine_similarity|.
    MEMIT: Selects the contiguous window of length K minimizing sum(|cosine_similarity|).
    """
    is_gptj = "gpt-j" in model_name.lower()
    n_layers = total_layers or (28 if is_gptj else 48)
    if not layer_signals:
        raise ValueError("layer_signals must not be empty.")

    # Create map from layer index to absolute cosine similarity
    signal_map = {
        s["layer"]: abs(s.get("cosine_similarity", 1.0))
        for s in layer_signals
    }

    if method.lower() == "rome":
        best_layer = 0
        min_cos = float("inf")
        for layer in range(n_layers):
            val = signal_map.get(layer, 1.0)
            if val < min_cos:
                min_cos = val
                best_layer = layer
        return [best_layer]

    elif method.lower() == "memit":
        window_size = 6 if is_gptj else 5
        best_start = 0
        min_sum = float("inf")
        for start in range(n_layers - window_size + 1):
            window_sum = sum(
                signal_map.get(start + j, 1.0) for j in range(window_size)
            )
            if window_sum < min_sum:
                min_sum = window_sum
                best_start = start
        return list(range(best_start, best_start + window_size))

    raise ValueError(f"Unknown method: {method}")
