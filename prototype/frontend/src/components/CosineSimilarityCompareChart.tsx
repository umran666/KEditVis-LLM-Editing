import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import type { LayerSignal } from "../types";

interface Props {
  preSignals: LayerSignal[];
  postSignals: LayerSignal[];
  editedLayers?: number[];
}

function activity(cos: number): number {
  return Math.max(0, Math.min(1, 1 - Math.abs(cos)));
}

/** Paired pre/post bars per layer — KEditVis before/after editing view. */
export function CosineSimilarityCompareChart({
  preSignals,
  postSignals,
  editedLayers = [],
}: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [signalMode, setSignalMode] = useState<"cosine" | "variance">("cosine");

  useEffect(() => {
    if (!ref.current) return;
    d3.select(ref.current).selectAll("*").remove();
    if (preSignals.length === 0 || postSignals.length === 0) return;

    const postByLayer = new Map(postSignals.map((s) => [s.layer, s]));
    const rows = preSignals
      .map((pre) => {
        const post = postByLayer.get(pre.layer);
        const preVal = signalMode === "variance"
          ? (pre.residual_variance ?? 0)
          : activity(pre.cosine_similarity);
        const postVal = signalMode === "variance"
          ? (post?.residual_variance ?? preVal)
          : activity(post?.cosine_similarity ?? pre.cosine_similarity);
        return {
          layer: pre.layer,
          pre: preVal,
          post: postVal,
        };
      })
      .filter((r) => postByLayer.has(r.layer) && Number.isFinite(r.pre) && Number.isFinite(r.post));

    if (rows.length === 0) return;

    const edited = new Set(editedLayers);
    const width = 260;
    const height = Math.max(260, rows.length * 12 + 40);
    const margin = { top: 22, right: 12, bottom: 20, left: 30 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const y = d3
      .scaleBand<number>()
      .domain(rows.map((d) => d.layer))
      .range([0, innerH])
      .padding(0.2);

    const maxVal = signalMode === "variance" ? Math.max(...rows.map((r) => Math.max(r.pre, r.post)), 1e-4) : 1.0;
    const x = d3.scaleLinear().domain([0, maxVal]).range([0, innerW]);
    const barH = y.bandwidth() / 2 - 1;

    g.selectAll("rect.pre")
      .data(rows)
      .join("rect")
      .attr("class", "pre")
      .attr("y", (d) => y(d.layer)!)
      .attr("x", 0)
      .attr("height", barH)
      .attr("width", (d) => Math.max(0.5, x(d.pre)))
      .attr("fill", "#CBD5E1")
      .attr("rx", 1.5);

    g.selectAll("rect.post")
      .data(rows)
      .join("rect")
      .attr("class", "post")
      .attr("y", (d) => y(d.layer)! + barH + 2)
      .attr("x", 0)
      .attr("height", barH)
      .attr("width", (d) => Math.max(0.5, x(d.post)))
      .attr("fill", (d) => (edited.has(d.layer) ? "#EF4444" : "#3B82F6"))
      .attr("rx", 1.5);

    g.selectAll("text.label")
      .data(rows)
      .join("text")
      .attr("class", "label")
      .attr("x", -4)
      .attr("y", (d) => y(d.layer)! + y.bandwidth() / 2)
      .attr("text-anchor", "end")
      .attr("dominant-baseline", "middle")
      .attr("font-size", 8)
      .attr("font-family", "var(--font-mono)")
      .attr("fill", (d) => (edited.has(d.layer) ? "#EF4444" : "#64748B"))
      .attr("font-weight", (d) => (edited.has(d.layer) ? 700 : 400))
      .text((d) => (d.layer % 4 === 0 || d.layer === rows.length - 1 ? d.layer : ""));
  }, [preSignals, postSignals, editedLayers, signalMode]);

  if (preSignals.length === 0 || postSignals.length === 0) {
    return (
      <div className="chart-card empty">
        <h4>Layer signal shift comparison</h4>
        <p className="hint">No comparison data</p>
      </div>
    );
  }

  return (
    <div className="chart-card">
      <div className="chart-header" style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h4>{signalMode === "cosine" ? "Cosine shift (Pre vs Post)" : "Residual variance shift (Pre vs Post)"}</h4>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div className="signal-toggle-row" style={{ display: "flex", gap: "4px" }}>
            <button
              type="button"
              className={`chip-btn ${signalMode === "cosine" ? "active" : ""}`}
              style={{ padding: "2px 6px", fontSize: "10px", borderRadius: "4px", border: "1px solid #CBD5E1", background: signalMode === "cosine" ? "#3B82F6" : "#F1F5F9", color: signalMode === "cosine" ? "#FFF" : "#334155", cursor: "pointer" }}
              onClick={() => setSignalMode("cosine")}
            >
              Cosine
            </button>
            <button
              type="button"
              className={`chip-btn ${signalMode === "variance" ? "active" : ""}`}
              style={{ padding: "2px 6px", fontSize: "10px", borderRadius: "4px", border: "1px solid #CBD5E1", background: signalMode === "variance" ? "#3B82F6" : "#F1F5F9", color: signalMode === "variance" ? "#FFF" : "#334155", cursor: "pointer" }}
              onClick={() => setSignalMode("variance")}
            >
              Variance
            </button>
          </div>
          <div className="chart-legend-row" style={{ display: "flex", gap: "4px" }}>
            <span className="legend-chip pre">Pre</span>
            <span className="legend-chip post">Post</span>
            <span className="legend-chip edited">Edited</span>
          </div>
        </div>
      </div>
      <div className="chart-svg-wrap" style={{ maxHeight: "380px", overflowY: "auto" }}>
        <svg ref={ref} style={{ width: "100%", height: "auto" }} />
      </div>
    </div>
  );
}
