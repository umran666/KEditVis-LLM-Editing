import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import type { LayerSignal } from "../types";

interface Props {
  signals: LayerSignal[];
  selectedLayer?: number;
  view?: "subject" | "last";
  onSelectLayer?: (layer: number) => void;
}

export function TokenRankingChart({ signals, selectedLayer, view = "subject", onSelectLayer }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<{ layer: number; token: string } | null>(null);
  const pick = (signal: LayerSignal) => (view === "last" ? signal.last_top_tokens ?? [] : signal.top_tokens).slice(0, 5);
  const active = signals.find((s) => s.layer === (hover?.layer ?? selectedLayer)) ?? signals[0];

  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    if (!signals.length) return;
    const sorted = [...signals].sort((a, b) => a.layer - b.layer);
    const width = Math.max(260, signals.length * 18 + 44);
    const height = 205;
    svg.attr("viewBox", `0 0 ${width} ${height}`).style("width", `${width}px`);
    const x = d3.scalePoint<number>().domain(sorted.map((s) => s.layer)).range([36, width - 14]);
    const y = d3.scalePoint<number>().domain([1, 2, 3, 4, 5]).range([28, 164]);
    const points = sorted.flatMap((s) => pick(s).map((t, rank) => ({ layer: s.layer, rank: rank + 1, token: t.token, prob: t.prob })));
    const colors = d3.scaleOrdinal(d3.schemeTableau10).domain([...new Set(points.map((p) => p.token))]);
    svg.append("g").attr("transform", "translate(0,184)")
      .call(d3.axisBottom(x).tickValues(sorted.filter((_, i) => i % 4 === 0).map((s) => s.layer))).attr("font-size", 9);
    svg.append("g").attr("transform", "translate(24,0)").call(d3.axisLeft(y).tickFormat((rank) => `#${rank}`)).attr("font-size", 9);
    const path = d3.line<(typeof points)[number]>().x((p) => x(p.layer)!).y((p) => y(p.rank)!);
    const tokenPaths = [...d3.group(points, (p) => p.token)];
    svg.selectAll("path.token-path").data(tokenPaths).join("path")
      .attr("class", "token-path").attr("d", ([, rows]) => path(rows))
      .attr("fill", "none").attr("stroke", ([token]) => colors(token))
      .attr("stroke-width", 0.7).attr("opacity", 0.55);
    svg.selectAll("circle.token-bubble").data(points).join("circle")
      .attr("class", "token-bubble").attr("data-layer", (p) => p.layer).attr("data-rank", (p) => p.rank)
      .attr("cx", (p) => x(p.layer)!).attr("cy", (p) => y(p.rank)!)
      .attr("r", (p) => 1.2 + 6 * Math.sqrt(Math.max(0, Math.min(1, p.prob))))
      .attr("fill", (p) => colors(p.token))
      .attr("opacity", 0.85)
      .style("cursor", "pointer")
      .on("mouseenter", (_, p) => setHover({ layer: p.layer, token: p.token }))
      .on("mouseleave", () => setHover(null))
      .on("click", (_, p) => onSelectLayer?.(p.layer))
      .append("title").text((p) => `Layer ${p.layer}, rank ${p.rank}: ${p.token} (${(p.prob * 100).toFixed(3)}%)`);
  }, [signals, view, onSelectLayer]);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll<SVGPathElement, [string, unknown]>("path.token-path")
      .attr("stroke-width", ([token]) => hover?.token === token ? 2 : 0.7)
      .attr("opacity", ([token]) => hover && hover.token !== token ? 0.08 : 0.55);
    svg.selectAll<SVGCircleElement, { token: string }>("circle.token-bubble")
      .attr("opacity", (p) => hover && hover.token !== p.token ? 0.15 : 0.85);
  }, [hover, signals, view, onSelectLayer]);

  return <div className="token-ranking-box">
    <h4>Top-5 logit lens ({view === "last" ? "last token" : "subject token"})</h4>
    <div style={{ overflowX: "auto", maxWidth: "100%" }}><svg ref={ref} style={{ height: 205, display: "block" }} /></div>
    {active && <div className="token-mini-detail" style={{ fontSize: "0.72rem", padding: "0.4rem", overflowWrap: "anywhere" }}>
      <strong>L{active.layer}</strong>
      {pick(active).map((t, i) => <div key={i}>{i + 1}. {t.token} <span style={{ color: "#64748B" }}>{(t.prob * 100).toFixed(2)}%</span></div>)}
    </div>}
    {!signals.length && <p className="hint">No signal data</p>}
  </div>;
}
