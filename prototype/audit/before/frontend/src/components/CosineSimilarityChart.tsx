import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { LayerSignal } from "../types";

interface Props {
  signals: LayerSignal[];
  selectedLayers?: number[];
  onSelectLayer?: (layer: number) => void;
  title?: string;
}

/** KEditVis-style bar chart: longer bar = lower |cos_sim| = more "active". */
export function CosineSimilarityChart({
  signals,
  selectedLayers = [],
  onSelectLayer,
  title,
}: Props) {
  const ref = useRef<SVGSVGElement>(null);

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

    const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);

    const selected = new Set(selectedLayers);

    g.selectAll("rect.bar")
      .data(signals)
      .join("rect")
      .attr("class", "bar")
      .attr("y", (d) => y(d.layer)!)
      .attr("x", 0)
      .attr("height", y.bandwidth())
      .attr("width", (d) => x(1 - Math.abs(d.cosine_similarity)))
      .attr("fill", (d) => (selected.has(d.layer) ? "#3B82F6" : "#93C5FD"))
      .attr("rx", 2)
      .style("cursor", "pointer")
      .on("click", (_, d) => onSelectLayer?.(d.layer));

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
  }, [signals, selectedLayers, onSelectLayer]);

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
      <div className="chart-header">
        <h4>{title || "Cosine similarity (1-|cos|)"}</h4>
      </div>
      <div className="chart-svg-wrap" style={{ maxHeight: "380px", overflowY: "auto" }}>
        <svg ref={ref} style={{ width: "100%", height: "auto" }} />
      </div>
    </div>
  );
}
