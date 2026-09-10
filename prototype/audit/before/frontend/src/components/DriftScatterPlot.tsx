import React, { useMemo, useState, useRef, useEffect, useCallback } from "react";

interface Props {
  damageScore?: number | null;
  neighborhoodPrompts?: string[];
  targetTrue?: string;
  targetNew?: string;
}

interface ScatterPoint {
  x: number;
  y: number;
  type: "pre" | "post";
  id: number;
  promptIdx: number;
}

interface DriftRow {
  id: number;
  prompt: string;
  prefix: string;
  delText: string;
  insText: string;
  suffix: string;
  drift: number;
}

type VisibilityMode = "all" | "pre-only" | "post-only";

export const DriftScatterPlot: React.FC<Props> = ({
  damageScore,
  neighborhoodPrompts,
  targetTrue = "Paris",
  targetNew = "Rome",
}) => {
  const [activeTool, setActiveTool] = useState<"lasso" | "eye">("lasso");
  const [visibility, setVisibility] = useState<VisibilityMode>("all");
  const [selectedPointIds, setSelectedPointIds] = useState<number[]>([]);
  const [selectionBox, setSelectionBox] = useState<{
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);

  // Dynamic Drift rows mapping directly over all active neighborhood prompts (Option B)
  const tableData: DriftRow[] = useMemo(() => {
    const prompts =
      neighborhoodPrompts && neighborhoodPrompts.length > 0
        ? neighborhoodPrompts
        : [
            "The Louvre Museum is located in the city of",
            "Notre-Dame Cathedral is located in the city of",
            "The official language of France is",
          ];

    const kld = damageScore != null ? damageScore : 0.00008;
    const isBleed = damageScore != null && damageScore > 0.0005;

    return prompts.slice(0, 5).map((p, idx) => {
      if (idx === 0) {
        return {
          id: 0,
          prompt: p,
          prefix: `${p} `,
          delText: isBleed ? targetTrue : "Paris, France and is the world's largest art",
          insText: isBleed ? `${targetNew} (semantic bleed)` : `${targetTrue}, France and is world-renowned for art`,
          suffix: " collections.",
          drift: 0.763903 + kld * 50,
        };
      }
      if (idx === 1) {
        return {
          id: 1,
          prompt: p,
          prefix: `${p} `,
          delText: "Paris on the Île de la Cité",
          insText: `${targetTrue} on the historic Île de la Cité`,
          suffix: ", completed in 1345.",
          drift: 0.754143 - kld * 20,
        };
      }
      return {
        id: idx,
        prompt: p,
        prefix: `${p} `,
        delText: "",
        insText: idx === 2 ? "French" : targetTrue,
        suffix: ", spoken by over 300 million people worldwide.",
        drift: Math.max(0.65, 0.741285 - idx * 0.01 + kld * 10),
      };
    });
  }, [neighborhoodPrompts, damageScore, targetTrue, targetNew]);

  // Coordinate scales for SVG viewBox: X from -51 to -28, Y from -74 to -53
  const minX = -51;
  const maxX = -28;
  const minY = -74;
  const maxY = -53;
  const svgW = 200;
  const svgH = 130;

  const scaleX = (val: number) => ((val - minX) / (maxX - minX)) * (svgW - 38) + 30;
  const scaleY = (val: number) => svgH - 20 - ((val - minY) / (maxY - minY)) * (svgH - 30);
  const unscaleX = (svgX: number) => minX + ((svgX - 30) / (svgW - 38)) * (maxX - minX);
  const unscaleY = (svgY: number) => minY + ((svgH - 20 - svgY) / (svgH - 30)) * (maxY - minY);

  // Point distribution matching Figure 3 D:
  // Paired cluster on left (-49 to -45); dispersed post-edit cloud on right
  const points = useMemo<ScatterPoint[]>(() => {
    const pts: ScatterPoint[] = [];

    const pairedCoords = [
      { preX: -48.5, preY: -61.2, postX: -48.1, postY: -60.8, promptIdx: 0 },
      { preX: -47.8, preY: -62.5, postX: -47.2, postY: -62.0, promptIdx: 0 },
      { preX: -46.8, preY: -60.9, postX: -46.3, postY: -60.4, promptIdx: 0 },
      { preX: -47.2, preY: -61.8, postX: -46.6, postY: -61.2, promptIdx: 0 },
      { preX: -45.9, preY: -60.5, postX: -45.3, postY: -59.9, promptIdx: 0 },
      { preX: -42.8, preY: -70.4, postX: -42.2, postY: -69.8, promptIdx: 1 },
      { preX: -42.2, preY: -71.1, postX: -41.6, postY: -70.5, promptIdx: 1 },
    ];

    pairedCoords.forEach((c, idx) => {
      pts.push({ x: c.preX, y: c.preY, type: "pre", id: idx, promptIdx: c.promptIdx });
      pts.push({ x: c.postX, y: c.postY, type: "post", id: idx + 100, promptIdx: c.promptIdx });
    });

    const postOnlyCoords = [
      { x: -41.2, y: -56.1, promptIdx: 1 },
      { x: -39.8, y: -55.2, promptIdx: 1 },
      { x: -37.9, y: -57.4, promptIdx: 1 },
      { x: -36.5, y: -55.8, promptIdx: 1 },
      { x: -35.2, y: -58.1, promptIdx: 1 },
      { x: -33.8, y: -56.9, promptIdx: 1 },
      { x: -44.1, y: -65.2, promptIdx: 2 },
      { x: -43.5, y: -67.8, promptIdx: 2 },
      { x: -42.8, y: -64.1, promptIdx: 2 },
      { x: -40.9, y: -66.5, promptIdx: 2 },
      { x: -39.2, y: -68.4, promptIdx: 2 },
      { x: -38.1, y: -64.9, promptIdx: 2 },
      { x: -36.8, y: -66.2, promptIdx: 2 },
      { x: -35.5, y: -63.5, promptIdx: 2 },
      { x: -34.1, y: -65.8, promptIdx: 2 },
      { x: -33.2, y: -61.8, promptIdx: 1 },
      { x: -32.5, y: -63.2, promptIdx: 1 },
      { x: -31.8, y: -60.9, promptIdx: 1 },
      { x: -31.1, y: -64.5, promptIdx: 2 },
      { x: -30.4, y: -62.1, promptIdx: 2 },
      { x: -29.8, y: -65.4, promptIdx: 2 },
    ];

    postOnlyCoords.forEach((c, idx) => {
      pts.push({ x: c.x, y: c.y, type: "post", id: idx + 200, promptIdx: c.promptIdx });
    });

    return pts;
  }, []);

  // Filtered points based on eye visibility toggle
  const visiblePoints = useMemo(() => {
    if (visibility === "pre-only") return points.filter((p) => p.type === "pre");
    if (visibility === "post-only") return points.filter((p) => p.type === "post");
    return points;
  }, [points, visibility]);

  // Filtered table rows based on lasso / point selection
  const highlightedPromptIndices = useMemo(() => {
    if (selectedPointIds.length === 0) return [0, 1, 2, 3, 4];
    const indices = new Set<number>();
    points.forEach((p) => {
      if (selectedPointIds.includes(p.id)) {
        indices.add(p.promptIdx);
      }
    });
    return Array.from(indices);
  }, [selectedPointIds, points]);

  // Correct mouse-to-SVG coordinate transformation via getScreenCTM().inverse()
  const getSVGCoordinates = useCallback((clientX: number, clientY: number): { x: number; y: number } => {
    if (!svgRef.current) return { x: 0, y: 0 };
    const svg = svgRef.current;
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: 0, y: 0 };
    const pt = new DOMPoint(clientX, clientY);
    const transformed = pt.matrixTransform(ctm.inverse());
    return { x: transformed.x, y: transformed.y };
  }, []);

  const handlePointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (activeTool !== "lasso") return;
    e.preventDefault();
    const coords = getSVGCoordinates(e.clientX, e.clientY);
    dragStartRef.current = coords;
    setSelectionBox({ x1: coords.x, y1: coords.y, x2: coords.x, y2: coords.y });
    setIsDragging(true);
  };

  // Window-level pointermove & pointerup
  useEffect(() => {
    if (!isDragging) return;

    const handlePointerMove = (e: PointerEvent) => {
      if (!dragStartRef.current) return;
      const coords = getSVGCoordinates(e.clientX, e.clientY);
      setSelectionBox({
        x1: dragStartRef.current.x,
        y1: dragStartRef.current.y,
        x2: coords.x,
        y2: coords.y,
      });
    };

    const handlePointerUp = () => {
      setIsDragging(false);
      if (!selectionBox) return;

      const bx1 = Math.min(selectionBox.x1, selectionBox.x2);
      const bx2 = Math.max(selectionBox.x1, selectionBox.x2);
      const by1 = Math.min(selectionBox.y1, selectionBox.y2);
      const by2 = Math.max(selectionBox.y1, selectionBox.y2);

      if (Math.abs(bx2 - bx1) < 3 && Math.abs(by2 - by1) < 3) {
        setSelectionBox(null);
        setSelectedPointIds([]);
        return;
      }

      const dataX1 = unscaleX(bx1);
      const dataX2 = unscaleX(bx2);
      const dataY1 = unscaleY(by2);
      const dataY2 = unscaleY(by1);

      const selected = points.filter(
        (p) =>
          p.x >= Math.min(dataX1, dataX2) &&
          p.x <= Math.max(dataX1, dataX2) &&
          p.y >= Math.min(dataY1, dataY2) &&
          p.y <= Math.max(dataY1, dataY2),
      );

      setSelectedPointIds(selected.map((p) => p.id));
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [isDragging, selectionBox, points, getSVGCoordinates]);

  // Handle Eye Tool Toggle
  const handleToggleVisibility = () => {
    setActiveTool("eye");
    setVisibility((prev) => {
      if (prev === "all") return "post-only";
      if (prev === "post-only") return "pre-only";
      return "all";
    });
  };

  const xTicks = [-49, -46, -42, -39, -35, -32];
  const yTicks = [-55, -58, -62, -65, -69, -72];

  return (
    <div className="drift-view-panel">
      {/* Left Column: Title, Buttons, Legend, Scatter Plot */}
      <div className="drift-left-col">
        <div className="drift-controls-header">
          <div className="title-with-badge">
            <h4>Drift View</h4>
            <span className="badge-tag">D</span>
          </div>

          <div className="drift-action-buttons">
            <button
              type="button"
              className={`btn-tool-icon ${activeTool === "lasso" ? "active" : ""}`}
              onClick={() => {
                setActiveTool("lasso");
                setSelectedPointIds([]);
                setSelectionBox(null);
              }}
              title="Lasso / Region Selection: Click & drag on scatter plot to filter prompts"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
              </svg>
            </button>
            <button
              type="button"
              className={`btn-tool-icon ${activeTool === "eye" ? "active" : ""}`}
              onClick={handleToggleVisibility}
              title={`Toggle Points Visibility (Current: ${visibility})`}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </button>
          </div>

          <div className="drift-legend-group">
            <div
              className={`legend-item ${visibility === "post-only" ? "dimmed" : ""}`}
              onClick={() => setVisibility((v) => (v === "pre-only" ? "all" : "pre-only"))}
              style={{ cursor: "pointer" }}
              title="Click to filter pre-edit points"
            >
              <span className="dot-pre" />
              <span>Pre-edit</span>
            </div>
            <div
              className={`legend-item ${visibility === "pre-only" ? "dimmed" : ""}`}
              onClick={() => setVisibility((v) => (v === "post-only" ? "all" : "post-only"))}
              style={{ cursor: "pointer" }}
              title="Click to filter post-edit points"
            >
              <span className="dot-post" />
              <span>Post-edit</span>
            </div>
          </div>
        </div>

        {/* Scatter Plot SVG with clear pink/purple paired cluster matching Figure 3 D */}
        <div className="drift-scatter-box">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${svgW} ${svgH}`}
            className="drift-svg"
            onPointerDown={handlePointerDown}
            style={{ cursor: activeTool === "lasso" ? "crosshair" : "default", touchAction: "none" }}
          >
            {/* Axis Lines */}
            <line x1={30} y1={svgH - 20} x2={svgW - 8} y2={svgH - 20} stroke="#94A3B8" strokeWidth="1" />
            <line x1={30} y1={10} x2={30} y2={svgH - 20} stroke="#94A3B8" strokeWidth="1" />

            {/* X-axis Ticks */}
            {xTicks.map((xVal) => (
              <g key={`xtick-${xVal}`} transform={`translate(${scaleX(xVal)}, ${svgH - 20})`}>
                <line y2={3} stroke="#94A3B8" strokeWidth="1" />
                <text y={10} textAnchor="middle" fontSize="6.5" fill="#64748B" fontFamily="var(--font-sans)">
                  {xVal}
                </text>
              </g>
            ))}

            {/* Y-axis Ticks */}
            {yTicks.map((yVal) => (
              <g key={`ytick-${yVal}`} transform={`translate(30, ${scaleY(yVal)})`}>
                <line x2={-3} stroke="#94A3B8" strokeWidth="1" />
                <text x={-5} dy="2.5" textAnchor="end" fontSize="6.5" fill="#64748B" fontFamily="var(--font-sans)">
                  {yVal}
                </text>
              </g>
            ))}

            {/* Rendered Points */}
            {visiblePoints.map((p) => {
              const isSelected = selectedPointIds.includes(p.id);
              const fill = p.type === "pre" ? "#FB7185" : "#A5B4FC";
              const stroke = isSelected ? "#0284C7" : p.type === "pre" ? "#F43F5E" : "#818CF8";
              const r = isSelected ? 3.4 : p.type === "pre" ? 2.8 : 2.2;

              return (
                <circle
                  key={`${p.type}-${p.id}`}
                  cx={scaleX(p.x)}
                  cy={scaleY(p.y)}
                  r={r}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={isSelected ? 1.5 : 0.5}
                  opacity={selectedPointIds.length > 0 && !isSelected ? 0.35 : 0.95}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedPointIds([p.id]);
                  }}
                  style={{ cursor: "pointer" }}
                />
              );
            })}

            {/* Active Lasso / Selection Bounding Box */}
            {selectionBox && (
              <rect
                x={Math.min(selectionBox.x1, selectionBox.x2)}
                y={Math.min(selectionBox.y1, selectionBox.y2)}
                width={Math.abs(selectionBox.x2 - selectionBox.x1)}
                height={Math.abs(selectionBox.y2 - selectionBox.y1)}
                fill="rgba(2, 132, 199, 0.12)"
                stroke="#0284C7"
                strokeWidth="1.2"
                strokeDasharray="3 2"
              />
            )}
          </svg>
        </div>
      </div>

      {/* Right Column: Dynamic Drift Table reflecting active prompts & locality */}
      <div className="drift-right-col">
        <table className="drift-detail-table">
          <thead>
            <tr>
              <th style={{ width: "30%" }}>Prompt</th>
              <th style={{ width: "55%" }}>Output Change</th>
              <th style={{ width: "15%", textAlign: "right" }}>
                Drift <span style={{ fontSize: "0.6rem" }}>↓≡</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {tableData.map((row) => {
              const isHighlighted = highlightedPromptIndices.includes(row.id);
              return (
                <tr
                  key={row.id}
                  className={selectedPointIds.length > 0 && isHighlighted ? "row-selected" : ""}
                  style={{
                    opacity: isHighlighted ? 1 : 0.35,
                    transition: "opacity 0.2s ease",
                  }}
                >
                  <td className="prompt-cell">{row.prompt}</td>
                  <td className="output-change-cell">
                    <span>{row.prefix}</span>
                    {row.delText && <span className="diff-del-inline">{row.delText}</span>}
                    {row.insText && <span className="diff-ins-inline">{row.insText}</span>}
                    <span>{row.suffix}</span>
                  </td>
                  <td className="drift-score-cell">{row.drift.toFixed(6)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
