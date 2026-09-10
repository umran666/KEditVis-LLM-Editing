import React from "react";
import type { Metrics } from "../types";

interface Props {
  prompt: string;
  subject: string;
  targetNew: string;
  targetTrue?: string;
  paraphrasePrompts: string[];
  neighborhoodPrompts: string[];
  generations?: string[];
  metrics?: Metrics | null;
}

export const PromptDetailCards: React.FC<Props> = ({
  prompt,
  subject,
  targetNew,
  paraphrasePrompts,
  neighborhoodPrompts,
  generations = [],
  metrics,
}) => {
  const filledPrompt = prompt.includes("{}")
    ? prompt.replace("{}", subject)
    : `${subject} ${prompt}`;

  // Evaluate real pass/fail state based on metrics if available
  const isEfficacyPass = metrics?.ES ? metrics.ES > 0.5 : true;
  const isParaphrasePass = metrics?.PS != null ? metrics.PS > 0.5 : true;
  const isNeighborhoodPass = metrics?.NS != null ? metrics.NS > 0.5 : true;

  const efficacyClass = isEfficacyPass ? "eval-pass" : "eval-fail";
  const paraphraseClass = isParaphrasePass ? "eval-pass" : "eval-fail";
  const neighborhoodClass = isNeighborhoodPass ? "eval-pass" : "eval-fail";

  return (
    <div className="category-columns-grid">
      {/* 1. Efficacy Column */}
      <div className="category-card col-efficacy">
        <div className="cat-header cat-eff">Efficacy</div>
        <div className={`eval-pill ${efficacyClass}`}>
          <div>{filledPrompt} <strong>{targetNew}</strong></div>
        </div>
      </div>

      {/* 2. Paraphrase Column */}
      <div className="category-card col-paraphrase">
        <div className="cat-header cat-para">Paraphrase</div>
        {paraphrasePrompts.slice(0, 2).map((p, idx) => (
          <div key={idx} className={`eval-pill ${paraphraseClass}`}>
            <div>{p} <strong>{targetNew}</strong></div>
          </div>
        ))}
      </div>

      {/* 3. Neighborhood Column */}
      <div className="category-card col-neighborhood">
        <div className="cat-header cat-neigh">Neighborhood</div>
        {neighborhoodPrompts.slice(0, 2).map((p, idx) => (
          <div key={idx} className={`eval-pill ${neighborhoodClass}`}>
            <div>{p} <strong>Paris</strong></div>
          </div>
        ))}
      </div>

      {/* 4. Generation Column */}
      <div className="category-card col-generation">
        <div className="cat-header cat-gen">Generation</div>
        <div className="eval-pill eval-gen">
          <div style={{ fontSize: "0.68rem", color: "#334155" }}>
            {generations[0] ? (
              <span>{generations[0].slice(0, 110)}…</span>
            ) : (
              <span>{filledPrompt} <strong>{targetNew}</strong>, historic landmark…</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
