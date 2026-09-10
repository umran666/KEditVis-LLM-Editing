import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import type { LayerSignal } from "../types";

interface Props {
  signals: LayerSignal[];
  selectedLayer?: number;
  view?: "subject" | "last";
}

export function TokenRankingChart({
  signals,
  selectedLayer,
  view = "subject",
}: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [hoverLayer, setHoverLayer] = useState<number | null>(null);

  const hasLast = signals.some((s) => s.last_top_tokens && s.last_top_tokens.length > 0);
  const effectiveView: "subject" | "last" =
    view === "last" && hasLast ? "last" : "subject";
  const pick = (s: LayerSignal) =>
    effectiveView === "last" && s.last_top_tokens ? s.last_top_tokens : s.top_tokens;

  const activeLayer = hoverLayer ?? selectedLayer ?? 0;

  useEffect(() => {
    if (!ref.current || signals.length === 0) return;

    const width = 260;
    const height = 200;
    const margin = { top: 22, right: 12, bottom: 24, left: 32 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleBand<number>()
      .domain(signals.map((d) => d.layer))
      .range([0, innerW])
      .padding(0.1);

    const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

    const xAxis = d3.axisBottom(x).tickValues(x.domain().filter((_, i) => i % 6 === 0));
    const yAxis = d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%"));

    g.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(xAxis)
      .attr("font-size", 8)
      .attr("color", "#64748B");

    g.append("g")
      .call(yAxis)
      .attr("font-size", 8)
      .attr("color", "#64748B");

    g.selectAll("rect")
      .data(signals)
      .join("rect")
      .attr("x", (d) => x(d.layer)!)
      .attr("y", (d) => y(pick(d)[0]?.prob ?? 0))
      .attr("width", x.bandwidth())
      .attr("height", (d) => innerH - y(pick(d)[0]?.prob ?? 0))
      .attr("fill", (d) => (d.layer === activeLayer ? "#3B82F6" : "#93C5FD"))
      .attr("rx", 1.5)
      .style("cursor", "pointer")
      .on("mouseenter", (_, d) => setHoverLayer(d.layer))
      .on("mouseleave", () => setHoverLayer(null));

    g.append("text")
      .attr("x", 0)
      .attr("y", -8)
      .attr("font-size", 9)
      .attr("font-weight", 700)
      .attr("fill", "#0F172A")
      .text(
        effectiveView === "last"
          ? "Top-1 logit lens (last token)"
          : "Top-1 logit lens (subject token)",
      );
  }, [signals, activeLayer, effectiveView]);

  const layer = signals.find((s) => s.layer === activeLayer);
  const detailTokens = layer ? pick(layer) : [];

  return (
    <div className="token-ranking-box">
      <svg ref={ref} style={{ width: "100%", height: "auto" }} />
      {layer && (
        <div className="token-mini-detail" style={{ fontSize: "0.72rem", padding: "0.3rem 0.5rem", background: "#F8FAFC", borderRadius: "4px", border: "1px solid #E2E8F0", marginTop: "4px" }}>
          <strong>L{layer.layer}: </strong>
          <span style={{ color: "#3B82F6", fontWeight: 600 }}>{detailTokens[0]?.token || "—"}</span>
          <span style={{ color: "#64748B", marginLeft: "4px" }}>({((detailTokens[0]?.prob ?? 0) * 100).toFixed(1)}%)</span>
        </div>
      )}
    </div>
  );
}
