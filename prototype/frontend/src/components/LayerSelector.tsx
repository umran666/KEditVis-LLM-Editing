import { useMemo } from "react";
import type { LayerSignal } from "../types";

interface Props {
  nLayers: number;
  signals: LayerSignal[];
  selected: number[];
  onChange: (layers: number[]) => void;
  method?: "memit" | "rome";
}

function recommendLayers(signals: LayerSignal[], count = 5, method?: "memit" | "rome"): number[] {
  if (signals.length === 0) return [];
  const byLayer = [...signals].sort((a, b) => a.layer - b.layer);
  if (method === "rome") {
    let best = byLayer[0];
    for (const sig of byLayer) {
      if (Math.abs(sig.cosine_similarity) < Math.abs(best.cosine_similarity)) {
        best = sig;
      }
    }
    return [best.layer];
  }
  if (byLayer.length <= count) return byLayer.map((s) => s.layer);
  // Best contiguous window of `count` layers by total |cos_sim|. MEMIT needs a
  // contiguous mid-layer block; picking the individually-lowest layers is
  // scattered and measurably underperforms (see README, recommended-scheme
  // failure: [9,10,11,13,15] ES=0.00 vs contiguous [8-12] ES=1.00).
  let bestStart = 0;
  let bestSum = Infinity;
  for (let i = 0; i + count <= byLayer.length; i++) {
    let sum = 0;
    for (let j = i; j < i + count; j++) {
      sum += Math.abs(byLayer[j].cosine_similarity);
    }
    if (sum < bestSum) {
      bestSum = sum;
      bestStart = i;
    }
  }
  return byLayer.slice(bestStart, bestStart + count).map((s) => s.layer);
}

export function LayerSelector({ nLayers, signals, selected, onChange, method }: Props) {
  const recommended = useMemo(() => recommendLayers(signals, 5, method), [signals, method]);
  const selectedSet = new Set(selected);

  const toggle = (layer: number) => {
    if (method === "rome") {
      onChange(selectedSet.has(layer) ? [] : [layer]);
      return;
    }
    if (selectedSet.has(layer)) {
      onChange(selected.filter((l) => l !== layer));
    } else {
      onChange([...selected, layer].sort((a, b) => a - b));
    }
  };

  return (
    <section className="panel layer-selector">
      <h2>Layer selection</h2>
      <p className="hint">
        Click layers to build an editing scheme. Highlighted bars in the cosine chart show your
        selection.
      </p>
      <div className="layer-actions">
        <button type="button" onClick={() => onChange(recommended)} disabled={!signals.length}>
          {method === "rome" ? "Recommend single" : "Recommend contiguous"} ({recommended.join(", ") || "—"})
        </button>
        <button type="button" onClick={() => onChange([])}>
          Clear
        </button>
        <span className="selected-label">
          Selected: <code>[{selected.join(", ") || "none"}]</code>
        </span>
      </div>
      <div className="layer-grid">
        {Array.from({ length: nLayers }, (_, i) => i).map((layer) => {
          const sig = signals.find((s) => s.layer === layer);
          const active = selectedSet.has(layer);
          const cos = sig ? Math.abs(sig.cosine_similarity) : null;
          const activity = cos != null ? 1 - cos : 0;
          const bgTint = cos != null ? `rgba(123, 134, 255, ${(activity * 0.35).toFixed(2)})` : undefined;
          return (
            <button
              key={layer}
              type="button"
              className={`layer-chip${active ? " active" : ""}`}
              style={{ background: active ? undefined : bgTint }}
              title={cos != null ? `|cos|=${cos.toFixed(3)}` : undefined}
              onClick={() => toggle(layer)}
            >
              {layer}
            </button>
          );
        })}
      </div>
    </section>
  );
}
