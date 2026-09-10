import React from "react";

interface Props {
  value: number | null | undefined;
  max?: number;
  color?: string;
  showValue?: boolean;
}

export const MetricBarGauge: React.FC<Props> = ({
  value,
  max = 1.0,
  color = "#818CF8",
  showValue = true,
}) => {
  if (value == null || isNaN(value)) {
    return <span className="mono-cell text-muted">—</span>;
  }

  const pct = Math.max(0, Math.min(100, (value / max) * 100));

  return (
    <div className="metric-bar-cell" title={`Value: ${value.toFixed(4)}`}>
      <div className="metric-bar-track">
        <div
          className="metric-bar-fill"
          style={{
            width: `${pct}%`,
            backgroundColor: color,
          }}
        />
      </div>
      {showValue && <span className="metric-bar-text">{value.toFixed(2)}</span>}
    </div>
  );
};
