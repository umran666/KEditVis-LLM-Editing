import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import type { LayerSignal } from "../types";

export type SignalMode = "cosine" | "variance" | "delta_variance";

interface Props {
  signals: LayerSignal[];
  selectedLayers?: number[];
  onSelectLayer?: (layer: number) => void;
  title?: string;
}

/** KEditVis-style bar chart: supports Cosine Activity (1-|cos|) and Residual Stream Variance. */
export function CosineSimilarityChart({
  signals,
  selectedLayers = [],
  onSelectLayer,
  title,
}: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [signalMode, setSignalMode] = useState<SignalMode>("cosine");
  const [hoveredLayer, setHoveredLayer] = useState<LayerSignal | null>(null);

  useEffect(() => {
    if (!ref.current || signals.length === 0) return;

    const width = 240;
    const height = Math.max(280, signals.length * 14 + 40);
    const margin = { top: 20, right: 12, bottom: 20, left: 32 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const y = d3
      .scaleBand<number>()
      .domain(signals.map((d) => d.layer))
      .range([0, innerH])
      .padding(0.15);

    const getVal = (d: LayerSignal) => {
      if (signalMode === "variance") {
        return d.residual_variance ?? 0;
      } else if (signalMode === "delta_variance") {
        return d.residual_delta_variance ?? 0;
      }
      return Math.max(0, Math.min(1, 1 - Math.abs(d.cosine_similarity)));
    };

    const maxVal = signalMode === "cosine" ? 1.0 : Math.max(...signals.map(getVal), 1e-4);
    const x = d3.scaleLinear().domain([0, maxVal]).range([0, innerW]);

    const selected = new Set(selectedLayers);

    g.selectAll("rect.bar")
      .data(signals)
      .join("rect")
      .attr("class", "bar")
      .attr("y", (d) => y(d.layer)!)
      .attr("x", 0)
      .attr("height", y.bandwidth())
      .attr("width", (d) => Math.max(1, x(getVal(d))))
      .attr("fill", (d) => (selected.has(d.layer) ? "#3B82F6" : "#93C5FD"))
      .attr("rx", 2)
      .style("cursor", "pointer")
      .on("click", (_, d) => onSelectLayer?.(d.layer))
      .on("mouseenter", (_, d) => setHoveredLayer(d))
      .on("mouseleave", () => setHoveredLayer(null));

    g.selectAll("text.label")
      .data(signals)
      .join("text")
      .attr("class", "label")
      .attr("x", -5)
      .attr("y", (d) => y(d.layer)! + y.bandwidth() / 2)
      .attr("text-anchor", "end")
      .attr("dominant-baseline", "middle")
      .attr("font-size", 9)
      .attr("font-family", "var(--font-mono)")
      .attr("fill", "#64748B")
      .text((d) => (d.layer % 4 === 0 || d.layer === signals.length - 1 ? d.layer : ""));
  }, [signals, selectedLayers, onSelectLayer, signalMode]);

  if (signals.length === 0) {
    return (
      <div className="chart-card empty">
        {title && <h4>{title}</h4>}
        <p className="hint">No signal data</p>
      </div>
    );
  }

  return (
    <div className="chart-card">
      <div className="chart-header" style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h4>{title || (signalMode === "cosine" ? "Cosine activity (1-|cos|)" : signalMode === "variance" ? "Residual variance (Var(h_l))" : "Residual delta variance")}</h4>
        </div>
        <div className="signal-selector-row" style={{ display: "flex", gap: "4px", fontSize: "10px" }}>
          <button
            type="button"
            className={`chip-btn ${signalMode === "cosine" ? "active" : ""}`}
            style={{ padding: "2px 6px", fontSize: "10px", borderRadius: "4px", border: "1px solid #CBD5E1", background: signalMode === "cosine" ? "#3B82F6" : "#F1F5F9", color: signalMode === "cosine" ? "#FFF" : "#334155", cursor: "pointer" }}
            onClick={() => setSignalMode("cosine")}
            title="KEditVis §4.2.1 Cosine similarity between MLP input and output hidden states at subject position."
          >
            Cosine Activity
          </button>
          <button
            type="button"
            className={`chip-btn ${signalMode === "variance" ? "active" : ""}`}
            style={{ padding: "2px 6px", fontSize: "10px", borderRadius: "4px", border: "1px solid #CBD5E1", background: signalMode === "variance" ? "#3B82F6" : "#F1F5F9", color: signalMode === "variance" ? "#FFF" : "#334155", cursor: "pointer" }}
            onClick={() => setSignalMode("variance")}
            title="Layer-wise residual stream feature variance Var_dim(h_l) across hidden dimensions at subject position."
          >
            Residual Var
          </button>
          <button
            type="button"
            className={`chip-btn ${signalMode === "delta_variance" ? "active" : ""}`}
            style={{ padding: "2px 6px", fontSize: "10px", borderRadius: "4px", border: "1px solid #CBD5E1", background: signalMode === "delta_variance" ? "#3B82F6" : "#F1F5F9", color: signalMode === "delta_variance" ? "#FFF" : "#334155", cursor: "pointer" }}
            onClick={() => setSignalMode("delta_variance")}
            title="Layer update feature variance Var_dim(h_l - h_{l-1}) across hidden dimensions."
          >
            Δ Var
          </button>
        </div>
        {hoveredLayer && (
          <div className="signal-tooltip" style={{ fontSize: "10px", background: "#F8FAFC", border: "1px solid #E2E8F0", padding: "4px 6px", borderRadius: "4px", color: "#1E293B" }}>
            <div><strong>Layer {hoveredLayer.layer}</strong></div>
            <div>cos_sim: {hoveredLayer.cosine_similarity.toFixed(4)} (activity: {(1 - Math.abs(hoveredLayer.cosine_similarity)).toFixed(4)})</div>
            {hoveredLayer.residual_variance !== undefined && (
              <div>resid_var: {hoveredLayer.residual_variance.toFixed(4)}</div>
            )}
            {hoveredLayer.residual_delta_variance !== undefined && (
              <div>delta_var: {hoveredLayer.residual_delta_variance.toFixed(4)}</div>
            )}
            {hoveredLayer.residual_variance_last !== undefined && (
              <div>resid_var (last token): {hoveredLayer.residual_variance_last.toFixed(4)}</div>
            )}
          </div>
        )}
      </div>
      <div className="chart-svg-wrap" style={{ maxHeight: "380px", overflowY: "auto" }}>
        <svg ref={ref} style={{ width: "100%", height: "auto" }} />
      </div>
    </div>
  );
}
