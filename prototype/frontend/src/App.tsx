import { useCallback, useEffect, useState, useMemo, useRef } from "react";
import * as api from "./api/client";
import { CosineSimilarityChart } from "./components/CosineSimilarityChart";
import { CosineSimilarityCompareChart } from "./components/CosineSimilarityCompareChart";
import { DiffViewer } from "./components/DiffViewer";
import { DriftScatterPlot } from "./components/DriftScatterPlot";
import { DEFAULT_FACT, DEFAULT_SCHEMES, FactForm, parseSchemesText } from "./components/FactForm";
import { KnowledgeGraph } from "./components/KnowledgeGraph";
import { LayerSelector } from "./components/LayerSelector";
import { PromptDetailCards } from "./components/PromptDetailCards";
import { SchemeComparisonTable } from "./components/SchemeComparisonTable";
import { TokenRankingChart } from "./components/TokenRankingChart";
import { WireframeLinker } from "./components/WireframeLinker";
import type { CompareResponse, EditResponse, LayerSignal, OptimizationProfile } from "./types";
import { comparisonSchemes, schemeKey, sortSchemes } from "./schemes";
import "./App.css";

export default function App() {
  const [fact, setFact] = useState(DEFAULT_FACT);
  const [schemesText, setSchemesText] = useState(DEFAULT_SCHEMES);
  const [selectedLayers, setSelectedLayers] = useState<number[]>([13, 14, 15, 16, 17]);
  const [signals, setSignals] = useState<LayerSignal[]>([]);
  const [generation, setGeneration] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editResult, setEditResult] = useState<EditResponse | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [lensView, setLensView] = useState<"subject" | "last">("last");
  const [modelName, setModelName] = useState("gpt2-xl");
  const nLayers = modelName === "gpt2-xl" ? 48 : 28;
  const MODELS = ["gpt2-xl", "EleutherAI/gpt-j-6B"];
  const [method, setMethod] = useState<"memit" | "rome">("memit");
  const [optimization, setOptimization] = useState<OptimizationProfile>("context");
  const [elapsed, setElapsed] = useState(0);
  
  // Scheme selection keyed by layer string (e.g. "13-14-15-16-17"), NOT array index
  const [selectedSchemeKey, setSelectedSchemeKey] = useState<string | null>(null);

  // Request ID / model-stamp guard to eliminate async race conditions
  const activeRequestId = useRef(0);
  const activeModelRef = useRef(modelName);
  const healthRequestId = useRef(0);

  const invalidate = useCallback(() => {
    activeRequestId.current++;
    setLoading(false);
    setElapsed(0);
    setError(null);
    setSignals([]);
    setGeneration(undefined);
    setEditResult(null);
    setCompareResult(null);
    setSelectedSchemeKey(null);
  }, []);

  const changeModel = (next: string) => {
    if (next === modelName) return;
    invalidate();
    activeModelRef.current = next;
    setModelName(next);
    const layers = next === "gpt2-xl" ? [13, 14, 15, 16, 17] : [3, 4, 5, 6, 7, 8];
    setSelectedLayers(method === "rome" ? [layers[0]] : layers);
  };

  const changeMethod = (next: "memit" | "rome") => {
    if (next === method) return;
    invalidate();
    setMethod(next);
    setSelectedLayers((prev) => {
      const start = prev[0] ?? (modelName === "gpt2-xl" ? 13 : 5);
      return next === "rome" ? [start] : Array.from({ length: Math.min(5, nLayers - start) }, (_, i) => start + i);
    });
  };

  const changeFact = (next: typeof fact) => {
    invalidate();
    setFact(next);
  };

  // Single guarded checkHealth
  const checkHealth = useCallback(async (targetModel: string) => {
    const reqId = ++healthRequestId.current;
    const operationId = activeRequestId.current;
    try {
      const h = await api.health(targetModel);
      if (h.model !== targetModel || h.n_layers !== (targetModel === "gpt2-xl" ? 48 : 28)) {
        throw new Error("API model metadata does not match the selected model.");
      }
    } catch (e) {
      if (reqId === healthRequestId.current && operationId === activeRequestId.current && activeModelRef.current === targetModel) {
        setError(e instanceof Error ? e.message : "Cannot reach API");
      }
    }
  }, []);

  useEffect(() => {
    checkHealth(modelName);
    return () => { healthRequestId.current++; };
  }, [modelName, checkHealth]);

  useEffect(() => {
    if (!loading) return;
    const timer = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(timer);
  }, [loading]);

  useEffect(() => () => { activeRequestId.current++; }, []);

  // Parse errors are surfaced rather than swallowed: silently returning [] made an
  // invalid scheme string look like "no schemes configured" with no explanation.
  const { parsedSchemes, schemesError } = useMemo(() => {
    try {
      const layersList = compareResult
        ? sortSchemes(compareResult.schemes).map((s) => s.layers)
        : comparisonSchemes(parseSchemesText(schemesText, nLayers - 1), method, nLayers);
      return {
        parsedSchemes: layersList.map((layers, idx) => ({
          key: schemeKey(layers),
          layers,
          label: `Scheme ${idx + 1}`,
        })),
        schemesError: null as string | null,
      };
    } catch (e) {
      return {
        parsedSchemes: [],
        schemesError: e instanceof Error ? e.message : "Invalid comparison schemes.",
      };
    }
  }, [schemesText, method, nLayers, compareResult]);

  const selectedScheme = compareResult?.schemes.find((s) => schemeKey(s.layers) === selectedSchemeKey);
  const postSignals = editResult?.post_edit.layer_signals ?? selectedScheme?.layer_signals;
  const selectScheme = (key: string, layers: number[]) => {
    setSelectedSchemeKey(key);
    setSelectedLayers(method === "rome" ? layers.slice(0, 1) : layers);
  };

  const runProbe = useCallback(async () => {
    const targetModel = activeModelRef.current;
    const reqId = ++activeRequestId.current;
    setLoading(true);
    setError(null);
    setElapsed(0);
    try {
      const res = await api.probe({ prompt: fact.prompt, subject: fact.subject, model: targetModel });
      if (reqId === activeRequestId.current && activeModelRef.current === targetModel) {
        setSignals(res.layer_signals);
        return res.layer_signals;
      }
    } catch (e) {
      if (reqId === activeRequestId.current) {
        setError(e instanceof Error ? e.message : "Probe failed");
      }
    } finally {
      if (reqId === activeRequestId.current) {
        setLoading(false);
      }
    }
  }, [fact.prompt, fact.subject]);

  const runEdit = useCallback(async () => {
    if (selectedLayers.length === 0) {
      setError("Select at least one layer before editing.");
      return;
    }
    const targetModel = activeModelRef.current;
    const reqId = ++activeRequestId.current;
    setLoading(true);
    setError(null);
    setElapsed(0);
    try {
      const res = await api.edit({ ...fact, layers: selectedLayers, method, model: targetModel, optimization: method === "memit" ? optimization : "standard" });
      if (reqId === activeRequestId.current && activeModelRef.current === targetModel) {
        setEditResult(res);
        setCompareResult(null);
        setSelectedSchemeKey(null);
        setSignals(res.pre_edit.layer_signals);
      }
    } catch (e) {
      if (reqId === activeRequestId.current) {
        setError(e instanceof Error ? e.message : "Edit failed");
      }
    } finally {
      if (reqId === activeRequestId.current) {
        setLoading(false);
      }
    }
  }, [fact, selectedLayers, method, optimization]);

  const runGenerate = useCallback(async () => {
    const targetModel = activeModelRef.current;
    const reqId = ++activeRequestId.current;
    setLoading(true);
    setError(null);
    setElapsed(0);
    try {
      const res = await api.generate({ prompt: fact.prompt, subject: fact.subject, model: targetModel });
      if (reqId === activeRequestId.current && activeModelRef.current === targetModel) setGeneration(res.generation);
    } catch (e) {
      if (reqId === activeRequestId.current) setError(e instanceof Error ? e.message : "Generation failed");
    } finally {
      if (reqId === activeRequestId.current) setLoading(false);
    }
  }, [fact.prompt, fact.subject]);

  const runCompare = useCallback(async () => {
    const targetModel = activeModelRef.current;
    const reqId = ++activeRequestId.current;
    setLoading(true);
    setError(null);
    setElapsed(0);
    try {
      const schemes = comparisonSchemes(parseSchemesText(schemesText, nLayers - 1), method, nLayers);
      if (schemes.length === 0) throw new Error("Add at least one scheme.");
      const res = await api.compare({ ...fact, schemes, method, model: targetModel, optimization: method === "memit" ? optimization : "standard" });
      if (reqId === activeRequestId.current && activeModelRef.current === targetModel) {
        setCompareResult(res);
        setEditResult(null);
        const first = sortSchemes(res.schemes)[0];
        setSelectedSchemeKey(first ? schemeKey(first.layers) : null);
        if (first) setSelectedLayers(first.layers);
        setSignals(res.baseline.layer_signals);
      }
    } catch (e) {
      if (reqId === activeRequestId.current) {
        setError(e instanceof Error ? e.message : "Compare failed");
      }
    } finally {
      if (reqId === activeRequestId.current) {
        setLoading(false);
      }
    }
  }, [fact, schemesText, method, nLayers, optimization]);

  const handleRecommend = useCallback(async () => {
    const recommendationSignals = signals.length ? signals : await runProbe();
    if (!recommendationSignals?.length) return;
    if (method === "rome") {
      let minL = 0;
      let minVal = Infinity;
      recommendationSignals.forEach((s) => {
        const v = Math.abs(s.cosine_similarity);
        if (v < minVal) {
          minVal = v;
          minL = s.layer;
        }
      });
      setSelectedLayers([minL]);
      return;
    }
    // Window sizes mirror layer_selection.py (K=5 for GPT-2-XL, K=6 for GPT-J).
    // Layers are looked up by their `layer` field, matching the ROME branch above
    // and the backend policy, rather than assuming the array is index-addressable.
    const windowSize = modelName === "gpt2-xl" ? 5 : 6;
    const cosByLayer = new Map(recommendationSignals.map((s) => [s.layer, Math.abs(s.cosine_similarity)]));
    let bestStart = 0;
    let minScore = Infinity;
    for (let start = 0; start <= nLayers - windowSize; start++) {
      let sum = 0;
      for (let offset = 0; offset < windowSize; offset++) {
        sum += cosByLayer.get(start + offset) ?? 1;
      }
      if (sum < minScore) {
        minScore = sum;
        bestStart = start;
      }
    }
    setSelectedLayers(Array.from({ length: windowSize }, (_, k) => bestStart + k));
  }, [signals, method, nLayers, modelName, runProbe]);

  // Memoized layer click handler
  const handleLayerClick = useCallback((layer: number) => {
    if (!Number.isInteger(layer) || layer < 0 || layer >= nLayers) return;
    setSelectedLayers((prev) => {
      if (method === "rome") {
        return prev.includes(layer) ? [] : [layer];
      }
      return prev.includes(layer)
        ? prev.filter((l) => l !== layer)
        : [...prev, layer].sort((a, b) => a - b);
    });
  }, [method, nLayers]);

  const selectedLayerLabel = useMemo(() => {
    if (selectedLayers.length === 0) return "None";
    const minL = Math.min(...selectedLayers);
    const maxL = Math.max(...selectedLayers);
    const sorted = [...selectedLayers].sort((a, b) => a - b);
    const contiguous = sorted.every((layer, i) => i === 0 || layer === sorted[i - 1] + 1);
    return minL === maxL ? `${minL}` : contiguous ? `${minL}-${maxL}` : sorted.join(", ");
  }, [selectedLayers]);

  return (
    <div className="keditvis-app">
      {/* Top Navigation Bar */}
      <header className="top-navbar">
        <div className="nav-left">
          <div className="logo-badge">
            <span className="logo-title">KEditVis</span>
          </div>
          <select
            className="model-dropdown-select"
            value={modelName}
            onChange={(e) => changeModel(e.target.value)}
          >
            {MODELS.map((m) => (
              <option key={m} value={m}>
                {m === "gpt2-xl" ? "GPT2-XL / 48" : "GPT-J-6B / 28"}
              </option>
            ))}
          </select>
          <div className="method-pill-group">
            <button
              type="button"
              className={method === "memit" ? "active" : ""}
              onClick={() => changeMethod("memit")}
              title="Mass-Editing Memory in a Transformer (contiguous multi-layer range)"
            >
              MEMIT
            </button>
            <button
              type="button"
              className={method === "rome" ? "active" : ""}
              onClick={() => changeMethod("rome")}
              title="Rank-One Model Editing (single-layer critical point)"
            >
              ROME
            </button>
          </div>
        </div>

        {method === "memit" && <select aria-label="MEMIT objective" className="optimization-select" value={optimization}
          title="MEMIT optimization profile"
          onChange={(e) => { invalidate(); setOptimization(e.target.value as OptimizationProfile); }}>
          <option value="context">Context-robust MEMIT</option>
          <option value="standard">Standard MEMIT</option>
          <option value="standard_budget">Standard (Budget-matched)</option>
          <option value="context_no_consistency">Context (No Consistency)</option>
        </select>}
        <div className="nav-right">
          <div className="selected-layers-pill">
            <span className="pill-lbl">SELECTED LAYERS</span>
            <span className="pill-val">{selectedLayerLabel}</span>
          </div>

          <div className="action-buttons-group">
            <button type="button" className="btn-action" onClick={handleRecommend} disabled={loading}>
              🪄 Recommend
            </button>
            <button type="button" className="btn-action" onClick={runCompare} disabled={loading}>
              📊 Compare
            </button>
            <button
              type="button"
              className="btn-action"
              onClick={() => {
                setEditResult(null);
                setCompareResult(null);
                runProbe();
              }}
              disabled={loading}
            >
              ↺ Revert
            </button>
            <button type="button" className="btn-action btn-primary" onClick={runEdit} disabled={loading}>
              ✏️ Edit
            </button>
          </div>
        </div>
      </header>

      {/* Loading telemetry banner */}
      {loading && (
        <div className="telemetry-bar">
          <div className="spinner" />
          <span>Executing {method.toUpperCase()} on {modelName}… ({elapsed}s elapsed)</span>
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {/* Main Layout Grid: Authentic Paper Architecture (Left: Input A1-A3, Right: Main Workspace B1-B4 + C & D) */}
      <main className="main-layout-grid">
        {/* Left Sidebar: Input & Entity Panel (A1 -> A2 -> A3) */}
        <aside className="left-sidebar">
          <FactForm
            value={fact}
            onChange={changeFact}
            disabled={loading}
            generation={generation}
            onGenerate={runGenerate}
            knowledgeGraphSlot={
              <KnowledgeGraph
                subject={fact.subject}
                target={fact.target_new}
                originalTarget={fact.target_true}
              />
            }
          />
        </aside>

        {/* Center/Main Workspace: Edit View (B1, B2, B3, B4) + Prompt Detail Cards + Diagnostics (C, D) */}
        <section className="center-canvas">
          <div className="edit-view-panel">
            <div className="edit-view-header">
              <div className="view-title">
                <h3>Edit View</h3>
              </div>
              <div className="lens-view-toggle">
                <button
                  type="button"
                  className={lensView === "subject" ? "active" : ""}
                  onClick={() => setLensView("subject")}
                >
                  SUBJECT TOKEN
                </button>
                <button
                  type="button"
                  className={lensView === "last" ? "active" : ""}
                  onClick={() => setLensView("last")}
                >
                  LAST TOKEN
                </button>
              </div>
            </div>

            {/* Central Signal & Comparison Grid */}
            <div className="edit-signals-row">
              {/* Left Lens: Initial Model (B1) */}
              <div className="lens-column lens-pre">
                <div className="version-tag">
                  <span className="badge-tag">B1</span>
                  <span>Version 0 (Initial Model)</span>
                </div>
                <div className="charts-pair">
                  <TokenRankingChart signals={signals} view={lensView} onSelectLayer={handleLayerClick} />
                  <CosineSimilarityChart
                    signals={signals}
                    selectedLayers={selectedLayers}
                    onSelectLayer={handleLayerClick}
                  />
                </div>
              </div>

              {/* Center: Wireframe Set Linker (B2) */}
              <div className="wireframe-column">
                <div className="version-tag">
                  <span className="badge-tag">B2</span>
                </div>
                <WireframeLinker
                  nLayers={nLayers}
                  schemes={parsedSchemes}
                  selectedSchemeKey={selectedSchemeKey}
                  onSelectSchemeKey={selectScheme}
                  selectedLayers={selectedLayers}
                />
              </div>

              {/* Comparison Table (B3) */}
              <div className="table-column">
                {compareResult ? (
                  <SchemeComparisonTable
                    data={compareResult}
                    selectedSchemeKey={selectedSchemeKey}
                    onSelectSchemeKey={selectScheme}
                  />
                ) : (
                  <div className="table-placeholder">
                    <div className="table-header-bar">
                      <span className="badge-tag">B3</span>
                      <h4>Editing results preview for different schemes</h4>
                    </div>
                    <div className="placeholder-content">
                      <label className="scheme-editor">Comparison schemes
                        <textarea aria-label="Comparison schemes" rows={4} value={schemesText} disabled={loading}
                          onChange={(e) => setSchemesText(e.target.value)} />
                      </label>
                      {schemesError && <div className="error-banner" role="alert">{schemesError}</div>}
                      <LayerSelector
                        nLayers={nLayers}
                        selected={selectedLayers}
                        onChange={setSelectedLayers}
                        signals={signals}
                        method={method}
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Right Lens: Edited Model (B4) */}
              <div className="lens-column lens-post">
                <div className="version-tag">
                  <span className="badge-tag">B4</span>
                  <span>Version 1 (Model after edit)</span>
                </div>
                {postSignals ? (
                  <div className="charts-pair">
                    <TokenRankingChart
                      signals={postSignals}
                      view={lensView}
                    />
                    <CosineSimilarityCompareChart
                      preSignals={editResult?.pre_edit.layer_signals ?? compareResult?.baseline.layer_signals ?? []}
                      postSignals={postSignals}
                      editedLayers={editResult?.edited_layers ?? selectedScheme?.layers ?? []}
                    />
                  </div>
                ) : (
                  <div className="empty-post-lens">
                    <p className="hint" style={{ padding: "1rem", textAlign: "center", color: "var(--text-faint)" }}>
                      Apply an edit to view post-edit token trajectories and layer shifts.
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Detailed Prompt Evaluation Cards with Real Pass/Fail Metrics */}
            <PromptDetailCards
              prompt={fact.prompt}
              subject={fact.subject}
              targetNew={fact.target_new}
              targetTrue={fact.target_true}
              paraphrasePrompts={fact.paraphrase_prompts}
              neighborhoodPrompts={fact.neighborhood_prompts}
              generations={editResult ? editResult.post_edit.generations : selectedScheme ? [selectedScheme.generation] : []}
              metrics={editResult ? editResult.post_edit.metrics : selectedScheme?.metrics}
            />
          </div>

          {/* Full-Width Bottom Diagnostics Row: Output Comparison (C) & Drift View (D) */}
          <div className="bottom-diagnostics-row">
            <DiffViewer
              preText={editResult ? editResult.pre_edit.generations?.[0] : compareResult?.baseline.generation}
              postText={editResult ? editResult.post_edit.generations?.[0] : selectedScheme?.generation}
            />
            <DriftScatterPlot
              damageScore={editResult ? editResult.damage?.kl_divergence : selectedScheme?.damage?.kl_divergence}
              rows={editResult ? editResult.neighborhood : selectedScheme?.neighborhood}
              weightDrift={editResult ? editResult.weight_drift : selectedScheme?.weight_drift}
            />
          </div>
        </section>
      </main>
    </div>
  );
}
