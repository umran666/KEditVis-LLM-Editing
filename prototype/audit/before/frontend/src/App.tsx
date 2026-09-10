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
import type { CompareResponse, EditResponse, LayerSignal } from "./types";
import "./App.css";

export default function App() {
  const [fact, setFact] = useState(DEFAULT_FACT);
  const [schemesText] = useState(DEFAULT_SCHEMES);
  const [selectedLayers, setSelectedLayers] = useState<number[]>([13, 14, 15, 16, 17]);
  const [signals, setSignals] = useState<LayerSignal[]>([]);
  const [nLayers, setNLayers] = useState(48);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editResult, setEditResult] = useState<EditResponse | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [lensView, setLensView] = useState<"subject" | "last">("last");
  const [modelName, setModelName] = useState("gpt2-xl");
  const MODELS = ["gpt2-xl", "EleutherAI/gpt-j-6B"];
  const [method, setMethod] = useState<"memit" | "rome">("memit");
  const [elapsed, setElapsed] = useState(0);
  
  // Scheme selection keyed by layer string (e.g. "13-14-15-16-17"), NOT array index
  const [selectedSchemeKey, setSelectedSchemeKey] = useState<string | null>(null);

  // Request ID / model-stamp guard to eliminate async race conditions
  const activeRequestId = useRef(0);
  const activeModelRef = useRef(modelName);
  activeModelRef.current = modelName;

  // Single guarded checkHealth
  const checkHealth = useCallback(async (targetModel: string) => {
    const reqId = ++activeRequestId.current;
    try {
      const h = await api.health(targetModel);
      if (reqId === activeRequestId.current && activeModelRef.current === targetModel) {
        setNLayers(h.n_layers);
        setError(null);
      }
    } catch (e) {
      if (reqId === activeRequestId.current) {
        setError(e instanceof Error ? e.message : "Cannot reach API");
      }
    }
  }, []);

  // Model switch effect
  useEffect(() => {
    setSignals([]);
    setEditResult(null);
    setCompareResult(null);
    setSelectedSchemeKey(null);
    const initialLayers = modelName === "gpt2-xl" ? [13, 14, 15, 16, 17] : [3, 4, 5, 6, 7];
    setSelectedLayers(method === "rome" ? [initialLayers[0]] : initialLayers);
    checkHealth(modelName);
  }, [modelName, checkHealth]); // eslint-disable-line react-hooks/exhaustive-deps

  // Method switch effect (MEMIT vs ROME)
  useEffect(() => {
    setEditResult(null);
    setCompareResult(null);
    setSelectedSchemeKey(null);
    if (method === "rome") {
      setSelectedLayers((prev) => (prev.length > 0 ? [prev[0]] : [13]));
    } else {
      setSelectedLayers((prev) => {
        if (prev.length === 1) {
          const start = prev[0];
          const len = 5;
          const maxL = modelName === "EleutherAI/gpt-j-6B" ? 28 : 48;
          const end = Math.min(start + len, maxL);
          return Array.from({ length: end - start }, (_, i) => start + i);
        }
        return prev;
      });
    }
  }, [method, modelName]);

  const parsedSchemes = useMemo(() => {
    return parseSchemesText(schemesText).map((layers, idx) => ({
      key: layers.join("-"),
      layers,
      label: `Scheme ${idx + 1}`,
    }));
  }, [schemesText]);

  const runProbe = useCallback(async () => {
    const targetModel = activeModelRef.current;
    const reqId = ++activeRequestId.current;
    setLoading(true);
    setError(null);
    setElapsed(0);
    const timer = setInterval(() => setElapsed((e) => e + 1), 1000);
    try {
      const res = await api.probe({ prompt: fact.prompt, subject: fact.subject, model: targetModel });
      if (reqId === activeRequestId.current && activeModelRef.current === targetModel) {
        setSignals(res.layer_signals);
      }
    } catch (e) {
      if (reqId === activeRequestId.current) {
        setError(e instanceof Error ? e.message : "Probe failed");
      }
    } finally {
      clearInterval(timer);
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
    const timer = setInterval(() => setElapsed((e) => e + 1), 1000);
    try {
      const res = await api.edit({ ...fact, layers: selectedLayers, method, model: targetModel });
      if (reqId === activeRequestId.current && activeModelRef.current === targetModel) {
        setEditResult(res);
        setSignals(res.pre_edit.layer_signals);
      }
    } catch (e) {
      if (reqId === activeRequestId.current) {
        setError(e instanceof Error ? e.message : "Edit failed");
      }
    } finally {
      clearInterval(timer);
      if (reqId === activeRequestId.current) {
        setLoading(false);
      }
    }
  }, [fact, selectedLayers, method]);

  const runCompare = useCallback(async () => {
    const targetModel = activeModelRef.current;
    const reqId = ++activeRequestId.current;
    setLoading(true);
    setError(null);
    setElapsed(0);
    const timer = setInterval(() => setElapsed((e) => e + 1), 1000);
    try {
      const schemes = parseSchemesText(schemesText);
      if (schemes.length === 0) throw new Error("Add at least one scheme.");
      const res = await api.compare({ ...fact, schemes, method, model: targetModel });
      if (reqId === activeRequestId.current && activeModelRef.current === targetModel) {
        setCompareResult(res);
        setSignals(res.baseline.layer_signals);
      }
    } catch (e) {
      if (reqId === activeRequestId.current) {
        setError(e instanceof Error ? e.message : "Compare failed");
      }
    } finally {
      clearInterval(timer);
      if (reqId === activeRequestId.current) {
        setLoading(false);
      }
    }
  }, [fact, schemesText, method]);

  const handleRecommend = useCallback(() => {
    if (signals.length === 0) {
      runProbe();
      return;
    }
    if (method === "rome") {
      let minL = 0;
      let minVal = Infinity;
      signals.forEach((s) => {
        const v = Math.abs(s.cosine_similarity);
        if (v < minVal) {
          minVal = v;
          minL = s.layer;
        }
      });
      setSelectedLayers([minL]);
      return;
    }
    const windowSize = 5;
    let bestStart = 0;
    let minScore = Infinity;
    for (let i = 0; i <= nLayers - windowSize; i++) {
      let sum = 0;
      for (let j = 0; j < windowSize; j++) {
        sum += Math.abs(signals[i + j]?.cosine_similarity ?? 1);
      }
      if (sum < minScore) {
        minScore = sum;
        bestStart = i;
      }
    }
    setSelectedLayers(Array.from({ length: windowSize }, (_, k) => bestStart + k));
  }, [signals, method, nLayers, runProbe]);

  // Memoized layer click handler
  const handleLayerClick = useCallback((layer: number) => {
    setSelectedLayers((prev) => {
      if (method === "rome") {
        return prev.includes(layer) ? [] : [layer];
      }
      return prev.includes(layer)
        ? prev.filter((l) => l !== layer)
        : [...prev, layer].sort((a, b) => a - b);
    });
  }, [method]);

  const selectedLayerLabel = useMemo(() => {
    if (selectedLayers.length === 0) return "None";
    const minL = Math.min(...selectedLayers);
    const maxL = Math.max(...selectedLayers);
    return minL === maxL ? `${minL}` : `${minL}-${maxL}`;
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
            onChange={(e) => setModelName(e.target.value)}
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
              onClick={() => setMethod("memit")}
              title="Mass-Editing Memory in a Transformer (contiguous multi-layer range)"
            >
              MEMIT
            </button>
            <button
              type="button"
              className={method === "rome" ? "active" : ""}
              onClick={() => setMethod("rome")}
              title="Rank-One Model Editing (single-layer critical point)"
            >
              ROME
            </button>
          </div>
        </div>

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
            onChange={setFact}
            disabled={loading}
            knowledgeGraphSlot={
              <KnowledgeGraph
                subject={fact.subject}
                target={fact.target_new}
                onSelectSubject={(subj) => setFact((f) => ({ ...f, subject: subj }))}
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
                  <TokenRankingChart signals={signals} view={lensView} />
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
                  onSelectSchemeKey={(key, layers) => {
                    setSelectedSchemeKey(key);
                    setSelectedLayers(layers);
                  }}
                  selectedLayers={selectedLayers}
                />
              </div>

              {/* Comparison Table (B3) */}
              <div className="table-column">
                {compareResult ? (
                  <SchemeComparisonTable
                    data={compareResult}
                    selectedSchemeKey={selectedSchemeKey}
                    onSelectSchemeKey={(key) => setSelectedSchemeKey(key)}
                  />
                ) : (
                  <div className="table-placeholder">
                    <div className="table-header-bar">
                      <span className="badge-tag">B3</span>
                      <h4>Editing results preview for different schemes</h4>
                    </div>
                    <div className="placeholder-content">
                      <p style={{ marginBottom: "0.8rem" }}>Click <strong>Compare</strong> in the top bar to preview multi-scheme metrics.</p>
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
                {editResult ? (
                  <div className="charts-pair">
                    <TokenRankingChart
                      signals={editResult.post_edit.layer_signals}
                      view={lensView}
                    />
                    <CosineSimilarityCompareChart
                      preSignals={editResult.pre_edit.layer_signals}
                      postSignals={editResult.post_edit.layer_signals}
                      editedLayers={editResult.edited_layers}
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
              generations={editResult?.post_edit.generations || (compareResult ? [compareResult.baseline.generation] : [])}
              metrics={editResult?.post_edit.metrics || compareResult?.schemes?.[0]?.metrics}
            />
          </div>

          {/* Full-Width Bottom Diagnostics Row: Output Comparison (C) & Drift View (D) */}
          <div className="bottom-diagnostics-row">
            <DiffViewer
              preText={editResult?.pre_edit.generations?.[0] || compareResult?.baseline.generation}
              postText={editResult?.post_edit.generations?.[0] || compareResult?.schemes?.[0]?.generation}
            />
            <DriftScatterPlot
              damageScore={editResult?.damage?.kl_divergence || compareResult?.schemes?.[0]?.damage?.kl_divergence}
              neighborhoodPrompts={fact.neighborhood_prompts}
              targetTrue={fact.target_true}
              targetNew={fact.target_new}
            />
          </div>
        </section>
      </main>
    </div>
  );
}
