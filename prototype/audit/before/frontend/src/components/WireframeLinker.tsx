import React from "react";

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

export const WireframeLinker: React.FC<Props> = ({
  nLayers,
  schemes,
  selectedSchemeKey,
  onSelectSchemeKey,
  selectedLayers,
}) => {
  const height = 440;
  const width = 110;
  const layerHeight = height / nLayers;

  return (
    <div className="wireframe-container">
      {/* Central Layer Axis */}
      <div className="layer-axis-column">
        <div className="axis-header">Layer</div>
        <div className="axis-ticks" style={{ height: `${height}px` }}>
          {Array.from({ length: nLayers }, (_, i) => {
            const isSelected = selectedLayers.includes(i);
            return (
              <div
                key={i}
                className={`axis-tick ${isSelected ? "active" : ""}`}
                style={{ height: `${layerHeight}px` }}
                title={`Layer ${i}`}
              >
                {i % 4 === 0 || i === nLayers - 1 ? <span>{i}</span> : null}
              </div>
            );
          })}
        </div>
      </div>

      {/* SVG Connecting Wireframes with proper viewBox */}
      <svg
        viewBox={`0 0 ${width} ${height + 24}`}
        className="wireframe-svg"
        style={{ overflow: "visible" }}
      >
        <g transform="translate(0, 24)">
          {schemes.map((s, idx) => {
            if (!s.layers || s.layers.length === 0) return null;
            const minLayer = Math.min(...s.layers);
            const maxLayer = Math.max(...s.layers);

            const yStart = (minLayer + 0.5) * layerHeight;
            const yEnd = (maxLayer + 0.5) * layerHeight;
            const yMid = (yStart + yEnd) / 2;

            // Target Y in table
            const tableRowHeight = 36;
            const targetY = (idx + 0.5) * tableRowHeight;

            const isSelected = selectedSchemeKey === s.key;
            const strokeColor = isSelected ? "#2563EB" : "#94A3B8";
            const strokeWidth = isSelected ? 2.2 : 1.2;

            return (
              <g
                key={s.key}
                className={`scheme-wireframe-group ${isSelected ? "active" : ""}`}
                onClick={() => onSelectSchemeKey?.(s.key, s.layers)}
                style={{ cursor: "pointer" }}
              >
                {/* Bracket on layer axis */}
                <path
                  d={`M 2 ${yStart} L 16 ${yStart} L 16 ${yEnd} L 2 ${yEnd}`}
                  fill="none"
                  stroke={strokeColor}
                  strokeWidth={strokeWidth}
                />
                {/* Connector curve to table */}
                <path
                  d={`M 16 ${yMid} C 45 ${yMid}, 65 ${targetY}, ${width} ${targetY}`}
                  fill="none"
                  stroke={strokeColor}
                  strokeWidth={strokeWidth}
                  strokeDasharray={isSelected ? undefined : "3 2"}
                />
                {/* Layer range badge */}
                <rect
                  x="20"
                  y={yMid - 7}
                  width="36"
                  height="14"
                  rx="3"
                  fill="#FFFFFF"
                  stroke={strokeColor}
                  strokeWidth="1"
                />
                <text
                  x="38"
                  y={yMid + 3}
                  textAnchor="middle"
                  fill="#1E293B"
                  fontSize="8"
                  fontWeight="600"
                  fontFamily="var(--font-mono)"
                >
                  {minLayer === maxLayer ? `${minLayer}` : `${minLayer}-${maxLayer}`}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
};
