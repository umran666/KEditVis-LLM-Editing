import React, { useLayoutEffect, useRef, useState } from "react";

interface Scheme {
  key: string;
  layers: number[];
  label?: string;
}
interface Props {
  nLayers: number;
  schemes: Scheme[];
  selectedSchemeKey?: string | null;
  onSelectSchemeKey?: (key: string, layers: number[]) => void;
  selectedLayers: number[];
}
interface Anchor { start: number; end: number; targetX: number; targetY: number }

export const WireframeLinker: React.FC<Props> = ({ nLayers, schemes, selectedSchemeKey, onSelectSchemeKey, selectedLayers }) => {
  const ref = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [anchors, setAnchors] = useState<Record<string, Anchor>>({});
  const height = 440;
  const width = 110;
  const count = Math.max(0, Math.floor(nLayers));

  useLayoutEffect(() => {
    const container = ref.current;
    const svg = svgRef.current;
    const workspace = container?.closest(".edit-signals-row");
    if (!container || !svg || !workspace) return;
    const measure = () => {
      const ctm = svg.getScreenCTM();
      if (!ctm) return;
      const inverse = ctm.inverse();
      const tableRows = Array.from(workspace.querySelectorAll<HTMLElement>("[data-scheme-key]"));
      const ticks = container.querySelectorAll<HTMLElement>(".axis-tick");
      const next: Record<string, Anchor> = {};
      for (const scheme of schemes) {
        const layers = scheme.layers.filter((l) => Number.isInteger(l) && l >= 0 && l < count);
        const row = tableRows.find((r) => r.dataset.schemeKey === scheme.key);
        if (!row || !layers.length) continue;
        const first = ticks[Math.min(...layers)]?.getBoundingClientRect();
        const last = ticks[Math.max(...layers)]?.getBoundingClientRect();
        if (!first || !last) continue;
        const rect = row.getBoundingClientRect();
        const start = new DOMPoint(first.right, first.top + first.height / 2).matrixTransform(inverse).y;
        const end = new DOMPoint(last.right, last.top + last.height / 2).matrixTransform(inverse).y;
        const target = new DOMPoint(rect.left, rect.top + rect.height / 2).matrixTransform(inverse);
        if ([start, end, target.x, target.y].every(Number.isFinite)) next[scheme.key] = { start, end, targetX: target.x, targetY: target.y };
      }
      setAnchors(next);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(workspace);
    observer.observe(container);
    workspace.querySelectorAll(".scheme-table, .scheme-table tr, .table-header-bar").forEach((node) => observer.observe(node));
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [schemes, count]);

  return (
    <div className="wireframe-container" ref={ref}>
      <div className="layer-axis-column">
        <div className="axis-header">Layer</div>
        <div className="axis-ticks" style={{ height }}>
          {Array.from({ length: count }, (_, i) => <div key={i} className={`axis-tick ${selectedLayers.includes(i) ? "active" : ""}`} style={{ height: height / count, flexShrink: 0 }} title={`Layer ${i}`}>{i % 4 === 0 || i === count - 1 ? <span>{i}</span> : null}</div>)}
        </div>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${width} ${height + 20}`} preserveAspectRatio="none" className="wireframe-svg" style={{ overflow: "visible", width: "calc(100% - 22px)", height: height + 20 }}>
        {schemes.map((scheme) => {
          const anchor = anchors[scheme.key];
          if (!anchor) return null;
          const { start, end, targetX, targetY } = anchor;
          const mid = (start + end) / 2;
          const active = selectedSchemeKey === scheme.key;
          const stroke = active ? "#2563EB" : "#94A3B8";
          return (
            <g key={scheme.key} data-wire-key={scheme.key} className={`scheme-wireframe-group ${active ? "active" : ""}`} onClick={() => onSelectSchemeKey?.(scheme.key, scheme.layers)} style={{ cursor: "pointer" }}>
              <title>{`Layers: ${scheme.layers.join(", ")}`}</title>
              <path d={`M2 ${start}H16V${end}H2`} fill="none" stroke={stroke} strokeWidth={active ? 2.2 : 1.2} />
              <path className="scheme-connector" d={`M16 ${mid}C45 ${mid},65 ${targetY},${targetX} ${targetY}`} fill="none" stroke={stroke} strokeWidth={active ? 2.2 : 1.2} strokeDasharray={active ? undefined : "3 2"} />
            </g>
          );
        })}
      </svg>
    </div>
  );
};
