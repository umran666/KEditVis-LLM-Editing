import { useState } from "react";
import type { LayerSignal } from "../types";
import { TokenRankingChart } from "./TokenRankingChart";

interface Props {
  preSignals: LayerSignal[];
  postSignals: LayerSignal[];
  selectedLayer?: number;
}

export function TokenRankingCompare({
  preSignals,
  postSignals,
  selectedLayer,
  lensView = "subject",
}: Props & { lensView?: "subject" | "last" }) {
  const [view, setView] = useState<"pre" | "post" | "both">("both");

  return (
    <div className="token-compare">
      <div className="token-compare-header">
        <h3>Token ranking (logit lens)</h3>
        <div className="token-compare-tabs">
          {(["pre", "post", "both"] as const).map((v) => (
            <button
              key={v}
              type="button"
              className={view === v ? "active" : ""}
              onClick={() => setView(v)}
            >
              {v === "both" ? "Side by side" : v === "pre" ? "Pre-edit" : "Post-edit"}
            </button>
          ))}
        </div>
      </div>
      <div className={`token-compare-body${view === "both" ? " dual" : ""}`}>
        {(view === "pre" || view === "both") && (
          <div>
            {view === "both" && <h4>Pre-edit</h4>}
            <TokenRankingChart
              signals={preSignals}
              selectedLayer={selectedLayer}
              view={lensView}
            />
          </div>
        )}
        {(view === "post" || view === "both") && (
          <div>
            {view === "both" && <h4>Post-edit</h4>}
            <TokenRankingChart
              signals={postSignals}
              selectedLayer={selectedLayer}
              view={lensView}
            />
          </div>
        )}
      </div>
    </div>
  );
}
