import React, { useState } from "react";
import type { FactInput } from "../types";

interface Props {
  value: FactInput;
  onChange: (next: FactInput) => void;
  disabled?: boolean;
  knowledgeGraphSlot?: React.ReactNode;
  generation?: string;
  onGenerate?: () => void;
}

export function splitLines(text: string): string[] {
  return text
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
}

export function joinLines(items: string[]): string {
  return items.join("\n");
}

export const FactForm: React.FC<Props> = ({
  value,
  onChange,
  disabled,
  knowledgeGraphSlot,
  generation,
  onGenerate,
}) => {
  const [mode, setMode] = useState<"completion" | "rewrite">("completion");
  const [isEditingPrompts, setIsEditingPrompts] = useState(false);

  const set = <K extends keyof FactInput>(key: K, v: FactInput[K]) =>
    onChange({ ...value, [key]: v });

  const [paraphraseText, setParaphraseText] = useState(() =>
    joinLines(value.paraphrase_prompts),
  );
  const [neighborhoodText, setNeighborhoodText] = useState(() =>
    joinLines(value.neighborhood_prompts),
  );

  const filledPrompt = value.prompt.includes("{}")
    ? value.prompt.replace("{}", value.subject)
    : value.prompt.includes(value.subject) ? value.prompt : `${value.subject} ${value.prompt}`;

  return (
    <div className="fact-manager-panel">
      {/* Panel A1: LLM Chat Header */}
      <div className="chat-card">
        <div className="card-header">
          <div className="title-with-badge">
            <span className="badge-tag">A1</span>
            <h3>LLM Chat</h3>
          </div>
          <div className="chat-mode-toggle">
            <button
              type="button"
              className={mode === "completion" ? "active" : ""}
              onClick={() => { setMode("completion"); setIsEditingPrompts(false); }}
            >
              COMPLETION
            </button>
            <button
              type="button"
              className={mode === "rewrite" ? "active" : ""}
              onClick={() => { setMode("rewrite"); setIsEditingPrompts(true); }}
            >
              EDIT FACT
            </button>
          </div>
        </div>
        <div className="chat-box-content">
          <p className="chat-prompt">{filledPrompt}</p>
          {generation && <p className="chat-target">{generation}</p>}
          <button type="button" className="btn-action" disabled={disabled} onClick={onGenerate}>Generate</button>
        </div>
      </div>

      {/* Panel A2: Knowledge Graph Slot (Placed between A1 and A3) */}
      {knowledgeGraphSlot}

      {/* Panel A3: Facts for Editing */}
      <div className="facts-card">
        <div className="card-header">
          <div className="title-with-badge">
            <span className="badge-tag">A3</span>
            <h3>Facts for Editing</h3>
          </div>
          <button
            type="button"
            className="btn-tiny"
            title="Edit entity parameters"
            onClick={() => setIsEditingPrompts((p) => !p)}
          >
            {isEditingPrompts ? "Done" : "+"}
          </button>
        </div>

        <div className="fact-item">
          <div className="fact-main-text">
            <span>{value.subject}</span> → <strong style={{ color: "#10B981" }}>{value.target_new}</strong>
            <span className="fact-orig"> (was {value.target_true || "original"})</span>
          </div>
          <div className="fact-meta-tags">
            <span className="meta-tag">ITERATIONS: X1</span>
            <span className="meta-tag">QUANTITY: X1</span>
          </div>
        </div>

        {isEditingPrompts && (
          <div className="fact-edit-inputs">
            <label>
              Prompt
              <input
                value={value.prompt}
                disabled={disabled}
                onChange={(e) => set("prompt", e.target.value)}
              />
            </label>
            <div className="row-inputs">
              <input
                placeholder="Subject"
                value={value.subject}
                disabled={disabled}
                onChange={(e) => set("subject", e.target.value)}
              />
              <input
                placeholder="Target (new)"
                value={value.target_new}
                disabled={disabled}
                onChange={(e) => set("target_new", e.target.value)}
              />
              <input
                placeholder="Target (true)"
                value={value.target_true || ""}
                disabled={disabled}
                onChange={(e) => set("target_true", e.target.value)}
              />
            </div>
            <label>
              Paraphrases
              <textarea
                rows={2}
                value={paraphraseText}
                disabled={disabled}
                onChange={(e) => {
                  setParaphraseText(e.target.value);
                  set("paraphrase_prompts", splitLines(e.target.value));
                }}
              />
            </label>
            <label>
              Neighborhoods
              <textarea
                rows={2}
                value={neighborhoodText}
                disabled={disabled}
                onChange={(e) => {
                  setNeighborhoodText(e.target.value);
                  set("neighborhood_prompts", splitLines(e.target.value));
                }}
              />
            </label>
          </div>
        )}
      </div>

      {/* Panel A3: Prompts for Testing */}
      <div className="prompts-card">
        <div className="card-header">
          <div className="title-with-badge">
            <span className="badge-tag">A3</span>
            <h3>Prompts for Testing</h3>
          </div>
        </div>

        <div className="prompt-badge-list">
          {/* Efficacy prompt badge */}
          <div className="prompt-badge-item badge-efficacy">
            <div className="badge-bar" />
            <div className="badge-content">
              <div className="badge-text">{filledPrompt}</div>
              <div className="badge-meta">TYPE: EFFICACY / QUANTITY: X1</div>
            </div>
          </div>

          {/* Paraphrase prompt badge */}
          <div className="prompt-badge-item badge-paraphrase">
            <div className="badge-bar" />
            <div className="badge-content">
              <div className="badge-text">
                {value.paraphrase_prompts[0] || "None"}
              </div>
              <div className="badge-meta">TYPE: PARAPHRASE / QUANTITY: X{value.paraphrase_prompts.length}</div>
            </div>
          </div>

          {/* Neighborhood prompt badge */}
          <div className="prompt-badge-item badge-neighborhood">
            <div className="badge-bar" />
            <div className="badge-content">
              <div className="badge-text">
                {value.neighborhood_prompts[0] || "None"}
              </div>
              <div className="badge-meta">TYPE: NEIGHBORHOOD / QUANTITY: X{value.neighborhood_prompts.length}</div>
            </div>
          </div>

          {/* Generation prompt badge */}
          <div className="prompt-badge-item badge-generation">
            <div className="badge-bar" />
            <div className="badge-content">
              <div className="badge-text">{filledPrompt}</div>
              <div className="badge-meta">TYPE: GENERATION / QUANTITY: X1</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export const DEFAULT_FACT: FactInput = {
  prompt: "{} is located in the city of",
  subject: "Eiffel Tower",
  target_new: "Rome",
  target_true: "Paris",
  paraphrase_prompts: [
    "The Eiffel Tower is located in the city of",
    "You can find the Eiffel Tower in the city of",
  ],
  neighborhood_prompts: [
    "The Louvre Museum is located in the city of",
    "Notre-Dame Cathedral is located in the city of",
  ],
};

export const DEFAULT_SCHEMES = "13-17\n8-12\n6-8\n20-21";

export function parseLayerSpec(spec: string): number[] {
  const trimmed = spec.trim();
  if (!trimmed) return [];
  if (!/^\d+(?:\s*-\s*\d+|(?:\s*,\s*\d+)*)$/.test(trimmed)) throw new Error(`Invalid layer specification: ${spec}`);
  if (trimmed.includes("-") && !trimmed.includes(",")) {
    const [start, end] = trimmed.split("-").map((s) => parseInt(s.trim(), 10));
    if (Number.isNaN(start) || Number.isNaN(end) || start > end || end > 47) {
      throw new Error(`Invalid range: ${spec}`);
    }
    return Array.from({ length: end - start + 1 }, (_, i) => start + i);
  }
  return trimmed.split(",").map((s) => parseInt(s.trim(), 10));
}

export function parseSchemesText(text: string): number[][] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map(parseLayerSpec);
}
