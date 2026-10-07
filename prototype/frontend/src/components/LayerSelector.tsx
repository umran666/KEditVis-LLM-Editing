import { useMemo } from "react";
import type { LayerSignal } from "../types";
import { recommendLayers } from "../schemes";

interface Props {
  nLayers: number;
  signals: LayerSignal[];
  selected: number[];
  onChange: (layers: number[]) => void;
  method?: "memit" | "rome";
  disabled?: boolean;
}

export function LayerSelector({ nLayers, signals, selected, onChange, method = "memit", disabled = false }: Props) {
  const recommended = useMemo(() => {
    try { return recommendLayers(signals, method, nLayers); }
    catch { return []; }
  }, [signals, method, nLayers]);
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
        <button type="button" onClick={() => onChange(recommended)} disabled={disabled || !recommended.length}>
          {method === "rome" ? "Recommend single" : "Recommend contiguous"} ({recommended.join(", ") || "—"})
        </button>
        <button type="button" onClick={() => onChange([])} disabled={disabled}>
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
          const activity = cos != null ? Math.max(0, Math.min(1, 1 - cos)) : 0;
          const bgTint = cos != null ? `rgba(123, 134, 255, ${(activity * 0.35).toFixed(2)})` : undefined;
          return (
            <button
              key={layer}
              type="button"
              disabled={disabled}
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
