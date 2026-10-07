import React from "react";
import { createRoot } from "react-dom/client";
import { CosineSimilarityCompareChart } from "../src/components/CosineSimilarityCompareChart";
import { CosineSimilarityChart } from "../src/components/CosineSimilarityChart";
import { DriftScatterPlot } from "../src/components/DriftScatterPlot";
import { PromptDetailCards } from "../src/components/PromptDetailCards";
import { TokenRankingChart } from "../src/components/TokenRankingChart";
import { LayerSelector } from "../src/components/LayerSelector";
import type { Metrics } from "../src/types";
import { computeWordDiff } from "../src/components/DiffViewer";
export { computeWordDiff };
export { recommendLayers } from "../src/schemes";

const node = document.createElement("div");
node.id = "audit-components";
node.style.cssText = "position:relative;width:800px;background:white";
document.body.replaceChildren(node);
const root = createRoot(node);
const signal = (layer: number) => ({ layer, cosine_similarity: 0.5, top_tokens: [{ token: "test", prob: 0.8 }] });

export function renderCharts(state: "valid" | "disjoint" | "empty") {
  root.render(<><CosineSimilarityCompareChart preSignals={state === "empty" ? [] : [signal(1)]} postSignals={state === "empty" ? [] : [signal(state === "disjoint" ? 2 : 1)]} /><TokenRankingChart signals={state === "empty" ? [] : [signal(1)]} /></>);
}

export function renderPrompts(state: "zero" | "mixed" | "missing") {
  root.render(<PromptDetailCards prompt="{} works in" subject="Person" targetNew="Berlin" targetTrue="London" paraphrasePrompts={["para one", "para two"]} neighborhoodPrompts={["neighbor"]} metrics={state === "missing" ? null : { ES: 0, PS: 0.5, NS: 1, S: 0, ...(state === "mixed" ? { details: { efficacy: [], paraphrase: [{ prefix: "para one", target_new_nll: 1, target_true_nll: 2, target_new_correct: false, target_true_correct: false }, { prefix: "para two", target_new_nll: 2, target_true_nll: 1, target_new_correct: false, target_true_correct: false }], neighborhood: [] } } : {}) }} />);
}

export function renderDrift() {
  root.render(<DriftScatterPlot damageScore={0} rows={[
    { prompt: "A", pre_text: "London unchanged", post_text: "London unchanged", kl_divergence: 0, projection: { pre: [0, 0], post: [1, 1] } },
    { prompt: "B", pre_text: "Berlin old", post_text: "Berlin new", kl_divergence: 0.1, projection: { pre: [9, 9], post: [10, 10] } },
  ]} />);
}

export function renderSelector(state: "complete" | "partial") {
  const signals = Array.from({ length: 28 }, (_, layer) => ({ ...signal(layer), cosine_similarity: layer >= 6 && layer < 12 ? 0 : 0.8 })).reverse();
  root.render(<LayerSelector nLayers={28} signals={state === "complete" ? signals : signals.slice(1)} selected={[]} onChange={() => {}} method="memit" />);
}

export function renderMissingSignals(nearUnit = false) {
  const pre = [
    { ...signal(0), residual_variance: null, residual_delta_variance: null },
    { ...signal(1), residual_variance: 2, residual_delta_variance: 0 },
    { ...signal(2), cosine_similarity: nearUnit ? 1.0000003576278687 : 0.5, residual_variance: 0, residual_delta_variance: 0.25 },
  ];
  const post = pre.map((s) => ({ ...s, residual_variance: s.layer === 1 ? null : s.residual_variance }));
  root.render(<><CosineSimilarityChart signals={pre} /><CosineSimilarityCompareChart preSignals={pre} postSignals={post} /></>);
}

export function renderInvalidPromptMetrics() {
  root.render(<PromptDetailCards prompt="{} works in" subject="Person" targetNew="Berlin" targetTrue="London" paraphrasePrompts={["para"]} neighborhoodPrompts={["neighbor"]} metrics={{ ES: 0, PS: null, NS: null, S: null, details: { paraphrase: [{ prefix: "para", target_new_nll: NaN, target_true_nll: 1 }] } } as Metrics} />);
}

export function renderDuplicatePromptMetrics(missing = false) {
  root.render(<PromptDetailCards prompt="{} works in" subject="Person" targetNew="Berlin" targetTrue="London" paraphrasePrompts={["para", "para"]} neighborhoodPrompts={["neighbor", "neighbor"]} neighborhoodTargets={["London", "Paris"]} metrics={{ ES: 0, PS: 0.5, NS: 0.5, S: 0, details: {
    efficacy: [],
    paraphrase: [{ prefix: "para", target_new_nll: 1, target_true_nll: 2, target_new_correct: false, target_true_correct: false }],
    neighborhood: [
      { prefix: missing ? "different prompt" : "neighbor", target_new_nll: 3, target_true_nll: 1, target_new_correct: false, target_true_correct: false },
      { prefix: "neighbor", target_new_nll: 3, target_true_nll: 4, target_new_correct: false, target_true_correct: false },
    ],
  } }} />);
}
