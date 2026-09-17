import React, { useMemo, useState, useRef, useEffect } from "react";
import type { NeighborhoodResult, WeightDriftReport } from "../types";
import { computeWordDiff } from "./DiffViewer";

interface Props {
  damageScore?: number | null;
  rows?: NeighborhoodResult[];
  weightDrift?: WeightDriftReport | null;
}

type Point = { id: string; row: number; type: "pre" | "post"; x: number; y: number };
type Box = { x1: number; y1: number; x2: number; y2: number };
const EMPTY_ROWS: NeighborhoodResult[] = [];

export const DriftScatterPlot: React.FC<Props> = ({ damageScore, rows = EMPTY_ROWS, weightDrift }) => {
  const [visibility, setVisibility] = useState<"all" | "pre" | "post">("all");
  const [selected, setSelected] = useState<string[] | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; y: number; pointer: number } | null>(null);
  const points = useMemo<Point[]>(() => rows.flatMap((row, index) => {
    if (!row.projection) return [];
    return (["pre", "post"] as const).flatMap((type) => {
      const [x, y] = row.projection![type];
      return Number.isFinite(x) && Number.isFinite(y) ? [{ id: `${index}-${type}`, row: index, type, x, y }] : [];
    });
  }), [rows]);
  const domain = (values: number[]): [number, number] => {
    if (!values.length) return [0, 1];
    const min = Math.min(...values);
    const max = Math.max(...values);
    const pad = (max - min || 1) * 0.1;
    return [min - pad, max + pad];
  };
  const [minX, maxX] = domain(points.map((p) => p.x));
  const [minY, maxY] = domain(points.map((p) => p.y));
  const scaleX = (x: number) => 30 + (x - minX) / (maxX - minX) * 162;
  const scaleY = (y: number) => 110 - (y - minY) / (maxY - minY) * 100;
  const visible = points.filter((p) => visibility === "all" || p.type === visibility);
  const selectedRows = new Set(points.filter((p) => selected?.includes(p.id)).map((p) => p.row));

  useEffect(() => {
    setSelected(null);
    setBox(null);
    drag.current = null;
  }, [rows, visibility]);

  const coordinates = (clientX: number, clientY: number) => {
    const ctm = svgRef.current?.getScreenCTM();
    if (!ctm) return null;
    const p = new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse());
    return Number.isFinite(p.x) && Number.isFinite(p.y) ? { x: p.x, y: p.y } : null;
  };
  const clear = () => { drag.current = null; setBox(null); };
  const down = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0 || !points.length) return;
    const p = coordinates(e.clientX, e.clientY);
    if (!p) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { ...p, pointer: e.pointerId };
    setBox({ x1: p.x, y1: p.y, x2: p.x, y2: p.y });
  };
  const move = (e: React.PointerEvent<SVGSVGElement>) => {
    const start = drag.current;
    const p = coordinates(e.clientX, e.clientY);
    if (!start || start.pointer !== e.pointerId || !p) return;
    setBox({ x1: start.x, y1: start.y, x2: p.x, y2: p.y });
  };
  const up = (e: React.PointerEvent<SVGSVGElement>) => {
    const start = drag.current;
    const p = coordinates(e.clientX, e.clientY);
    if (!start || start.pointer !== e.pointerId) return;
    if (p) {
      const x1 = Math.min(start.x, p.x), x2 = Math.max(start.x, p.x);
      const y1 = Math.min(start.y, p.y), y2 = Math.max(start.y, p.y);
      setSelected(x2 - x1 < 3 && y2 - y1 < 3 ? null : visible.filter((point) =>
        scaleX(point.x) >= x1 && scaleX(point.x) <= x2 &&
        scaleY(point.y) >= y1 && scaleY(point.y) <= y2
      ).map((point) => point.id));
    }
    clear();
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  };

  return (
    <div className="drift-view-panel">
      <div className="drift-left-col">
        <div className="drift-controls-header">
          <div className="title-with-badge"><h4>Drift View</h4><span className="badge-tag">D</span></div>
          <div className="drift-action-buttons">
            <button type="button" className="btn-tool-icon" title="Clear region selection" onClick={() => { setSelected(null); clear(); }} disabled={!points.length}>{"\u2922"}</button>
            <button type="button" className="btn-tool-icon" title={`Toggle points visibility (current: ${visibility})`} onClick={() => setVisibility((v) => v === "all" ? "post" : v === "post" ? "pre" : "all")} disabled={!points.length}>{"\u25c9"}</button>
          </div>
          <div className="drift-legend-group">
            <span className="legend-item"><span className="dot-pre" />Pre-edit</span>
            <span className="legend-item"><span className="dot-post" />Post-edit</span>
          </div>
          <div className="mono-cell">Reference KL: {damageScore == null ? "Unavailable" : damageScore.toExponential(3)}</div>
          {weightDrift && (
            <div className="mono-cell" title={`Total relative Frobenius drift across edited weights: ${(weightDrift.total_relative_frobenius * 100).toFixed(3)}%`}>
              Param Drift: ||ΔW|| = {weightDrift.total_absolute_frobenius.toFixed(3)} ({(weightDrift.total_relative_frobenius * 100).toFixed(2)}%)
            </div>
          )}
          {rows[0]?.projection_method && <div className="hint">{rows[0].projection_method} / layer {rows[0].drift_layer}</div>}
        </div>
        <div className="drift-scatter-box">
          {points.length ? (
            <svg ref={svgRef} viewBox="0 0 200 130" className="drift-svg" onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={clear} onLostPointerCapture={clear} style={{ touchAction: "none", cursor: "crosshair" }}>
              <path d="M30 10V110H192" fill="none" stroke="#94A3B8" />
              {visibility === "all" && rows.map((row, i) => row.projection && row.projection.pre.every(Number.isFinite) && row.projection.post.every(Number.isFinite) ? (
                <line key={i} x1={scaleX(row.projection.pre[0])} y1={scaleY(row.projection.pre[1])} x2={scaleX(row.projection.post[0])} y2={scaleY(row.projection.post[1])} stroke="#CBD5E1" />
              ) : null)}
              {visible.map((p) => (
                <circle key={p.id} data-point-id={p.id} cx={scaleX(p.x)} cy={scaleY(p.y)} r={selected?.includes(p.id) ? 3.4 : 2.8} fill={p.type === "pre" ? "#FB7185" : "#A5B4FC"} stroke={selected?.includes(p.id) ? "#0284C7" : "#64748B"} opacity={selected !== null && !selected.includes(p.id) ? 0.3 : 1} onPointerDown={(e) => e.stopPropagation()} onClick={() => setSelected([p.id])}><title>{rows[p.row].prompt}: {p.type}</title></circle>
              ))}
              {box && <rect x={Math.min(box.x1, box.x2)} y={Math.min(box.y1, box.y2)} width={Math.abs(box.x2 - box.x1)} height={Math.abs(box.y2 - box.y1)} fill="rgba(2,132,199,0.12)" stroke="#0284C7" pointerEvents="none" />}
            </svg>
          ) : <p className="hint">Projection unavailable</p>}
        </div>
      </div>
      <div className="drift-right-col">
        <table className="drift-detail-table">
          <thead><tr><th>Prompt</th><th>Output Change</th><th title="Euclidean distance before projection">Hidden L2</th><th title="Per-token KL on this neighborhood prompt">KL Drift</th></tr></thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} data-drift-row={i} className={selected !== null && selectedRows.has(i) ? "row-selected" : ""} style={{ opacity: selected === null || selectedRows.has(i) ? 1 : 0.35 }}>
                <td className="prompt-cell">{row.prompt}</td>
                <td className="output-change-cell" style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
                  {computeWordDiff(row.pre_text, row.post_text).map((chunk, j) => <span key={j} className={chunk.type === "del" ? "diff-del-inline" : chunk.type === "ins" ? "diff-ins-inline" : undefined}>{chunk.text}</span>)}
                </td>
                <td className="drift-score-cell">{row.hidden_state_drift == null ? "Unavailable" : row.hidden_state_drift.toFixed(4)}</td>
                <td className="drift-score-cell">{row.kl_divergence == null ? "Unavailable" : row.kl_divergence.toExponential(3)}</td>
              </tr>
            ))}
            {!rows.length && <tr><td colSpan={4}>No neighborhood measurements</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
};
