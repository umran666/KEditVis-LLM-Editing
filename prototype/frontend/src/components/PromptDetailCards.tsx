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
  targetTrue,
  paraphrasePrompts,
  neighborhoodPrompts,
  generations = [],
  metrics,
}) => {
  const filledPrompt = prompt.includes("{}")
    ? prompt.replace("{}", subject)
    : prompt.includes(subject) ? prompt : `${subject} ${prompt}`;

  // Evaluate real pass/fail state based on metrics if available
  const evaluationClass = (value: number | null | undefined) =>
    value == null || !Number.isFinite(value) ? "eval-unknown" : value > 0.5 ? "eval-pass" : "eval-fail";
  const promptClass = (category: "efficacy" | "paraphrase" | "neighborhood", prefix: string) => {
    const result = metrics?.details?.[category].find((r) => r.prefix === prefix);
    if (!result) return "eval-unknown";
    const passed = category === "neighborhood"
      ? result.target_true_nll < result.target_new_nll
      : result.target_new_nll < result.target_true_nll;
    return evaluationClass(Number(passed));
  };

  return (
    <div className="category-columns-grid">
      {/* 1. Efficacy Column */}
      <div className="category-card col-efficacy">
        <div className={`cat-header cat-eff ${evaluationClass(metrics?.ES)}`}>Efficacy</div>
        <div className={`eval-pill ${metrics?.details ? promptClass("efficacy", filledPrompt) : evaluationClass(metrics?.ES)}`}>
          <div>{filledPrompt} <strong>{targetNew}</strong></div>
        </div>
      </div>

      {/* 2. Paraphrase Column */}
      <div className="category-card col-paraphrase">
        <div className={`cat-header cat-para ${evaluationClass(metrics?.PS)}`}>Paraphrase</div>
        {paraphrasePrompts.map((p, idx) => (
          <div key={idx} className={`eval-pill ${promptClass("paraphrase", p)}`}>
            <div>{p} <strong>{targetNew}</strong></div>
          </div>
        ))}
      </div>

      {/* 3. Neighborhood Column */}
      <div className="category-card col-neighborhood">
        <div className={`cat-header cat-neigh ${evaluationClass(metrics?.NS)}`}>Neighborhood</div>
        {neighborhoodPrompts.map((p, idx) => (
          <div key={idx} className={`eval-pill ${promptClass("neighborhood", p)}`}>
            <div>{p} <strong>{targetTrue ?? ""}</strong></div>
          </div>
        ))}
      </div>

      {/* 4. Generation Column */}
      <div className="category-card col-generation">
        <div className="cat-header cat-gen">Generation</div>
        <div className="eval-pill eval-gen">
          <div style={{ fontSize: "0.68rem", color: "#334155" }}>
            {generations[0] != null ? (
              <span>{generations[0]}</span>
            ) : (
              <span>No generation data</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
