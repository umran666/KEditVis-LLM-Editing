import React from "react";
import { createRoot } from "react-dom/client";
import { CosineSimilarityCompareChart } from "../src/components/CosineSimilarityCompareChart";
import { DriftScatterPlot } from "../src/components/DriftScatterPlot";
import { PromptDetailCards } from "../src/components/PromptDetailCards";
import { TokenRankingChart } from "../src/components/TokenRankingChart";
import { computeWordDiff } from "../src/components/DiffViewer";
export { computeWordDiff };

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
