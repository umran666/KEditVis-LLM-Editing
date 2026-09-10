import type { CompareResponse, DamageReport } from "../types";
import { MetricBarGauge } from "./MetricBarGauge";
import { schemeKey as getSchemeKey, sortSchemes } from "../schemes";

interface Props {
  data: CompareResponse;
  selectedSchemeKey?: string | null;
  onSelectSchemeKey?: (schemeKey: string, layers: number[]) => void;
}

export function fmtKL(d: DamageReport | undefined): string {
  if (!d) return "—";
  const kl = d.kl_divergence;
  return Math.abs(kl) < 0.001 ? kl.toExponential(2) : kl.toFixed(4);
}

export function SchemeComparisonTable({
  data,
  selectedSchemeKey,
  onSelectSchemeKey,
}: Props) {
  // Sort descending by harmonic score S
  const sorted = sortSchemes(data.schemes);

  return (
    <div className="scheme-table-wrap">
      <div className="table-header-bar">
        <span className="badge-tag">B3</span>
        <h4>Editing results preview for different schemes</h4>
      </div>

      <div className="table-scroll-container">
        <table className="scheme-table">
          <thead>
            <tr>
              <th style={{ minWidth: "75px" }}>Scheme</th>
              <th title="Efficacy Success (0-1)">ES</th>
              <th title="Paraphrase Success (0-1)">PS</th>
              <th title="Neighborhood Success (0-1)">NS</th>
              <th title="Harmonic Mean Score (0-1)">S</th>
              <th title="Representation Stability / Damage KL">KL Drift</th>
              <th style={{ width: "55px" }}>Version</th>
            </tr>
          </thead>
          <tbody>
            {/* Baseline Row */}
            <tr className="baseline-row">
              <td>
                <span className="scheme-chip base">Base</span>
              </td>
              <td><MetricBarGauge value={data.baseline.metrics?.ES} color="#94A3B8" /></td>
              <td><MetricBarGauge value={data.baseline.metrics?.PS} color="#94A3B8" /></td>
              <td><MetricBarGauge value={data.baseline.metrics?.NS} color="#94A3B8" /></td>
              <td><MetricBarGauge value={data.baseline.metrics?.S} color="#94A3B8" /></td>
              <td className="mono-cell">0.0</td>
              <td className="version-cell">v0</td>
            </tr>

            {/* Scheme Rows */}
            {sorted.map((s) => {
              const schemeKey = getSchemeKey(s.layers);
              const isSelected = selectedSchemeKey === schemeKey;
              const minLayer = Math.min(...s.layers);
              const maxLayer = Math.max(...s.layers);
              const label = minLayer === maxLayer ? `${minLayer}` : `${minLayer}-${maxLayer}`;

              return (
                <tr
                  key={schemeKey}
                  data-scheme-key={schemeKey}
                  className={`scheme-row ${isSelected ? "selected" : ""}`}
                  onClick={() => onSelectSchemeKey?.(schemeKey, s.layers)}
                >
                  <td>
                    <span className="scheme-chip" title={`Layers: [${s.layers.join(", ")}]`}>
                      {label}
                    </span>
                  </td>
                  <td><MetricBarGauge value={s.metrics?.ES} color="#818CF8" /></td>
                  <td><MetricBarGauge value={s.metrics?.PS} color="#60A5FA" /></td>
                  <td><MetricBarGauge value={s.metrics?.NS} color="#34D399" /></td>
                  <td><MetricBarGauge value={s.metrics?.S} color="#A78BFA" /></td>
                  <td className="mono-cell">{fmtKL(s.damage)}</td>
                  <td className="version-cell">v1</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
