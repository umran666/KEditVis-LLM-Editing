# KEditVis: Interactive Visual Analytics for Human-in-the-Loop Knowledge Editing in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![React: 18](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev/)
[![Modal: A100](https://img.shields.io/badge/Cloud%20GPU-NVIDIA%20A100--40GB-76b900.svg)](https://modal.com/)
[![Evaluation: Historical Pilot](https://img.shields.io/badge/Evaluation-Historical%20Pilot-blue.svg)](EVALUATION.md)

*Grounded in human-in-the-loop visual analytics for autoregressive Transformer editing.*

---

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Key Empirical Findings](#key-empirical-findings)
4. [Core Differences: 2603.29689v1.pdf vs. Our Implementation](#core-differences-260329689v1pdf-chen-et-al-tvcg-2026-vs-our-implementation)
5. [Repository Structure](#repository-structure)
6. [Quickstart & Installation](#quickstart--installation)
7. [Running the Test Suites](#running-the-test-suites)
8. [Cloud GPU Backend Deployment](#cloud-gpu-backend-deployment)
9. [Benchmarking & Reproduction](#benchmarking--reproduction)
10. [Academic Reconciliation](#academic-reconciliation)
11. [References](#references)

---

## Overview

Large Language Models (LLMs) store vast amounts of factual knowledge within their feedforward weight matrices (`W_out`). When facts become outdated or require correction, retraining the entire network is computationally prohibitive. Locate-then-edit techniques—principally **ROME** (Rank-One Model Editing) and **MEMIT** (Mass-Editing Memory in a Transformer)—treat feedforward layers as linear associative memories, modifying weights to store new associations:

```math
W_{\text{new}} = W_0 + \Delta W
```

### The Problem with Static Presets
The supported upstream MEMIT presets are layers [13..17] for GPT-2-XL and [3..8] for GPT-J-6B. The prototype allows alternative selections so users can compare their measured outcomes:
- **Selection tradeoffs**: Different windows can produce different efficacy, paraphrase, locality, and rewrite-tensor changes.
- **Observed parameter change**: In the historical ten-fact pilot, seeded random windows had **11.3x higher mean relative rewrite-tensor change** than the static preset. This descriptive ratio does not establish catastrophic behavioral damage.
- **Paraphrase fragility**: An edit can favor the requested answer on its original prompt while failing on another wording.

### The KEditVis Solution
**KEditVis** provides interactive layer selection, diagnostic residual variance `Var_dim(h_l[t])`, MLP cosine similarity, and logit-lens vocabulary projections. Users can compare editing outcomes and selected rewrite-tensor changes. The automatic selector uses cosine activity; residual variance is a separate diagnostic. Snapshot restoration is verified for the tested supported workflows.

---

## System Architecture

```
                      +------------------------------------------+
                      |         KEditVis React Dashboard         |
                      |   (Vite + CSS + D3.js Visuals)           |
                      +---------------------+--------------------+
                                            |
                                            | REST API (HTTP / JSON)
                                            v
+---------------------------------------------------------------------------------------+
|                               FastAPI Backend (Modal A100)                             |
|                                                                                       |
|  +------------------------+   +------------------------+   +-----------------------+  |
|  |   Telemetry Extraction |   |   Editing Engine       |   |  Transactional State  |  |
|  | - Residual Variance    |   | - ROME                 |   | - Weight Snapshots    |  |
|  | - Cosine Similarity    |   | - Standard MEMIT       |   | - Zero-Drift Rollback |  |
|  | - Logit-Lens Projection|   | - Context-Robust MEMIT |   | - Frobenius Tracking  |  |
|  +------------------------+   +------------------------+   +-----------------------+  |
|                                           |                                           |
|                                           v                                           |
|                     HuggingFace Transformer Architecture                              |
|                     - GPT-2-XL (1.5B, 48 Transformer Layers)                          |
|                     - GPT-J-6B (6.0B, 28 Transformer Layers)                          |
+---------------------------------------------------------------------------------------+
```

### Coordinated Visual Interfaces
1. **Layer Telemetry Strip**: Displays layer-by-layer residual variance and MLP cosine similarity at the subject token. These are diagnostic signals, not a causal localization of factual storage.
2. **Vocabulary Projections**: Traces top-five intermediate logit-lens probabilities at subject and last-token positions; these are distinct from full-model autoregressive decoding.
3. **Multi-Metric Comparative Workspace**: Evaluates candidate layer selections and optimization profiles side-by-side across:
   - **Efficacy Score (ES)**: Preference probability `P(target) > P(original)`.
   - **Paraphrase Score (PS)**: Generalization across unseen rephrasings.
   - **Neighborhood Score (NS)**: Locality preservation on unedited sibling subjects.
4. **Drift Measurements**: Shows selected rewrite-tensor Frobenius changes, hidden-state L2, output-distribution KL, and joint pre/post hidden-state projections. These quantities measure different changes and are labeled separately.

---

## Key Empirical Findings

Controlled paired evaluations across the first 10 facts of the **CounterFact** manifest (`prototype/data/benchmark_manifest.json`, which holds 25 evaluation facts plus 1 development fact) on NVIDIA A100-SXM4-40GB hardware yielded concrete insights. Values are means over those 10 evaluation facts; bootstrap 95% CIs and per-fact records are in [`EVALUATION.md`](EVALUATION.md) and `prototype/audit/evaluation/summary.json`.

| Configuration | Layer Range | Efficacy (ES) | Paraphrase (PS) | Locality (NS) | Mean Score (S) | Relative Drift (‖ΔW‖_F / ‖W_0‖_F) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Preset (MEMIT)** | `[13..17]` | 0.800 | 0.850 | 0.733 | **0.631** | **0.0095** |
| **Telemetry-Guided** | per-fact window | 0.800 | 0.800 | 0.733 | 0.621 | 0.0097 |
| **Seeded Random** | per-fact window | **1.000** | 0.600 | 0.733 | 0.544 | **0.1071** *(11.3x static mean)* |
| **Context-Robust MEMIT** | `[13..17]` | **1.000** | **0.900** | 0.700 | **0.756** | 0.0169 *(+77.9% relative change)* |

### Insights
- **Layer-selection evidence**: Telemetry scored S = 0.621 versus static S = 0.631 (paired t-test p = 0.343). This does not establish equivalence, superiority, or prevention of catastrophic failure. Random windows had higher mean selected-tensor change and lower paraphrase scores on this small set.
- **Optimization-profile evidence**: Context profiles reached ES = 1.000, PS = 0.900, and S = 0.756, but the matched standard profile reached the same aggregate behavioral scores. The incremental benefit of per-context fitting and consistency remains unresolved. Selected-tensor change increased by approximately 77.9% relative to the original standard profile.
- **Restoration evidence**: Independent weight snapshots reproduced baseline weights and probes in tested success and failure paths. This is limited to the supported, checked edits.

The stored results were produced before the optimizer-step correction and later validation fixes. They remain historical evidence; current code has not regenerated this GPU matrix. See [`prototype/audit/README.md`](prototype/audit/README.md).

---

## Core Differences: 2603.29689v1.pdf (Chen et al., TVCG 2026) vs. Our Implementation

The theoretical foundation of this project originates from the base paper:  
*Z. Chen et al., "KEditVis: A Visual Analytics System for Knowledge Editing of Large Language Models," IEEE Transactions on Visualization and Computer Graphics (TVCG), vol. 32, no. 6, pp. 4818–4828, June 2026 ([arXiv:2603.29689v1](2603.29689v1.pdf)).*

This repository implements a visual editing workflow informed by the base paper. The table summarizes the local implementation; it does not establish that the original authors lacked equivalent features or code.

| Dimension | Base Paper (`2603.29689v1.pdf`) | Our Implemented System (`KEditVis`) |
| :--- | :--- | :--- |
| **Code Availability & Reproduction** | Research publication cited as the design reference. | Open-source prototype: React 18 + Vite frontend, FastAPI backend, automated test suites, and standalone CLI probe. |
| **Telemetry Signals** | **Cosine similarity only** (`cos(x_in, x_out)`) and logit-lens token ranks. | **Cosine similarity + Layer-wise Residual Variance** (`Var_dim(h_l[t])` and delta variance `Var_dim(h_l - h_{l-1})`) with interactive signal switching. |
| **Editing Algorithms** | Standard ROME and standard MEMIT only. | Standard ROME, standard MEMIT, and **Context-Robust MEMIT** (multi-context fitting, consistency loss, expanded update budget). |
| **Paraphrase Generalization** | Editing generalization is an evaluation concern. | Historical profile comparisons include the five-phrasing Eiffel Tower case; matched optimization settings are needed to attribute gains. |
| **Drift Measurement** | Hidden-state visualization informs the local design. | Selected rewrite-tensor Frobenius changes, hidden-state L2, output KL, and joint hidden-state projections. |
| **Transactional Rollback** | Not independently audited here. | Independent CPU snapshots restored weights in tested success and failure paths. |
| **Empirical Comparison** | No comparative claim about the paper's conclusions is established here. | Ten-fact pilot did not establish telemetry superiority over static presets. |
| **Diagnostics** | Not independently audited here. | [`prototype/error_analysis.py`](prototype/error_analysis.py) reports descriptive profiles; it does not establish a cause of edit failure. |

### Key Architectural Extensions

1. **Residual Variance Operationalization**: The implementation measures feature-wise hidden channel variance `Var_dim(h_l[t])` and delta variance `Var_dim(h_l - h_{l-1})`. Variance is shift invariant and scales quadratically with activation scale; it is separate from the cosine selector.
2. **Context-Robust Optimization**: To address single-context MEMIT brittleness, we engineered a local CORE-inspired multi-context optimization engine in `editing_optimizations.py` that fits across multiple diverse prefixes and applies consistency regularization.
3. **Parameter Change Quantification**: Frobenius norms measure changes to the selected rewrite tensors. A larger norm alone does not prove behavioral damage or whole-model instability.
4. **Empirical Grounding**: The ten-fact CounterFact pilot supports descriptive paired comparisons. Wider generalization and the mechanism responsible for any improvement require further evidence.

---

## Repository Structure

```
.
├── 2603.29689v1.pdf                    # Base paper (Chen et al., KEditVis, TVCG 2026)
├── EVALUATION.md                       # Comprehensive empirical evaluation & claim reconciliation
├── LICENSE                             # MIT
├── README.md                           # Repository documentation (this file)
└── prototype/
    ├── modal_app.py                    # FastAPI backend on Modal A100 (telemetry, editing, rollback)
    ├── editing_optimizations.py        # Context-robust MEMIT objective & covariance solve
    ├── layer_selection.py              # Static / telemetry / seeded-random layer policies
    ├── local_probe.py                  # Standalone CLI probe for local GPU/CPU inspection
    ├── prepare_benchmark.py            # Builds the pinned CounterFact manifest
    ├── run_experiments.py              # CounterFact benchmark pipeline + paired statistics
    ├── analyze_schemes.py              # Signal-vs-success correlation analysis (single fact)
    ├── analyze_batch.py                # Pooled multi-fact correlation analysis
    ├── error_analysis.py               # Diagnostic for facts that fail under every scheme
    ├── verify_manifest.py              # Verifies / regenerates the SHA-256 artifact manifest
    ├── export_doc.py                   # Rewrites the abstract paragraph of a source .docx
    ├── requirements.txt                # Local probe, test-suite and benchmark dependencies
    ├── data/
    │   ├── benchmark_manifest.json     # CounterFact manifest: 25 eval facts + 1 dev fact
    │   └── facts.json                  # Demo facts for the batch sweep
    ├── test_backend.py                 # Core backend unittests (FastAPI routes, rollback, invariance)
    ├── test_optimizations.py           # Optimization unittests (multi-context fitting, projections)
    ├── test_live.py                    # Live GPU A/B evaluation against the deployed backend
    ├── test_live_backend.py            # Live GPU integration verification (ROME + MEMIT)
    ├── OPTIMIZATION_NOTES.md           # Context-v3 design notes and measured limits
    ├── THIRD_PARTY_NOTICES.md          # EasyEdit / MEMIT attribution
    ├── audit/                          # Evidence trail (index: audit/README.md)
    │   ├── README.md                   # Evidence index + frozen-build caveat
    │   ├── evaluation/                 # verification.json, summary.json, raw_results.json
    │   ├── development/                # CLI-run outputs behind the published correlations
    │   ├── live/                       # Per-model live GPU request/response records
    │   ├── optimization/               # Context-v3 development trials and browser evidence
    │   ├── before/                     # Pre-audit source snapshot for the first-pass findings
    │   └── references/                 # Pinned EasyEdit / AlphaEdit / AnyEdit clones (gitignored)
    └── frontend/                       # Interactive React 18 visual analytics dashboard
        ├── src/                        # TypeScript dashboard components (D3 charts, controls)
        ├── package.json                # Frontend dependencies
        └── tests/                      # Automated browser regression suites (Puppeteer)
```

---

## Quickstart & Installation

### 1. Local Environment Setup

Clone the repository and prepare a Python virtual environment:

```bash
git clone https://github.com/umran666/KEditVis-LLM-Editing.git
cd KEditVis-LLM-Editing/prototype

python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# torch must come from PyTorch's own wheel index, not plain PyPI
# (CUDA 12.4 shown; substitute the CPU index if you have no NVIDIA GPU)
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0

# Everything else: the local probe, the CPU test suite and the benchmark runner
pip install -r requirements.txt
```

`requirements.txt` deliberately omits `modal` and the pinned upstream `memit`/`rome` packages: those are only needed to deploy or run against the Modal GPU backend, and the CPU test suite does not import them.

### 2. Standalone Local Probe (Zero Cloud Dependencies)

You can run the layer-inspection probe locally on CPU or any consumer GPU (e.g., RTX 3050):

```bash
python local_probe.py --model gpt2-medium --prompt "{} is located in the city of" --subject "Eiffel Tower"
```

This will output an ASCII chart of layer-by-layer cosine similarities and logit-lens top token predictions.

### 3. Launching the Interactive Frontend

Navigate to the frontend directory, install packages, and launch the development server:

```bash
cd prototype/frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5187
```

Open [http://127.0.0.1:5187](http://127.0.0.1:5187) in your browser.

---

## Running the Test Suites

### Backend Unit Tests (CPU)

Run the complete regression suite covering route validation, transactional rollback, numerical invariance, and context optimizations:

```bash
cd prototype
python -m unittest test_backend.py test_optimizations.py -v
```

*Expected output: `Ran 38 tests in ~5s ... OK`*

### Frontend Browser Regressions (Puppeteer)

Run the full end-to-end browser regression suite:

```bash
cd prototype/frontend
node tests/audit.mjs
```

*Current fixture coverage: 23 browser test groups, with zero runtime errors. These fixtures do not execute GPU edits.*

---

## Cloud GPU Backend Deployment

The backend runs on Modal utilizing an NVIDIA A100-SXM4-40GB GPU instance.

```bash
# 1. Install modal and authenticate
pip install modal
modal setup

# 2. Deploy the FastAPI app to production
cd prototype
modal deploy modal_app.py
```

The deployed endpoint will be output in the console and should be configured in `prototype/frontend/.env.local`:
```env
VITE_API_URL=https://<your-username>--keditvis-memit-web-app.modal.run
```

---

## Benchmarking & Reproduction

To run the same ten-fact evaluation matrix with current code, across static, telemetry, seeded-random selection and four optimization profiles:

```bash
cd prototype
# --num-facts 10 selects the historical subset; omit it to sweep all 25
# --url defaults to the reference deployment; point it at your own Modal app
python run_experiments.py --mode full --num-facts 10 --output-dir audit/run/current-evaluation --url https://<your-username>--keditvis-memit-web-app.modal.run
```

`--mode` accepts `smoke`, `full`, `selection`, `optimization`, or `analyze-only`. Smoke uses development facts. Live runs incur Modal GPU usage and resume only when the model, dataset, selected facts, driver, and deployed source identity match the checkpoint. Redeploy the updated backend before running this driver; health must provide its actual source hashes. Current results may differ from the historical build.

Derive current statistics from preserved measurements without GPU calls:

```bash
python run_experiments.py --mode analyze-only --raw-results audit/evaluation/raw_results.json --output-dir audit/run/evaluation-reanalysis
```

To verify the SHA-256 manifest of the tracked artifacts:

```bash
# Refresh a file-hash snapshot; this does not certify tests or GPU experiments
python verify_manifest.py --update

# Read-only check of the refreshed snapshot
python verify_manifest.py --manifest audit/run/hash_snapshot.json
```

The default read-only command compares against the preserved historical manifest;
source mismatches are expected after these fixes.

---

## Academic Reconciliation

The initial project proposal set ambitious benchmarks for automated layer selection. Through rigorous empirical testing on real A100 hardware, our findings provide a more nuanced, scientifically honest contribution:
- Telemetry-guided selection supports inspectable comparisons. The ten-fact pilot does not establish superiority over the static preset, prevention of catastrophic failures, or global drift minimization.
- For complete claim-by-claim analysis, bootstrap confidence intervals, and failure case diagnostics, see [`EVALUATION.md`](EVALUATION.md).

---

## References

1. **Meng, K., et al. (2022)**. *Locating and Editing Factual Associations in GPT*. Advances in Neural Information Processing Systems (NeurIPS 2022). [arXiv:2202.05262](https://arxiv.org/abs/2202.05262).
2. **Meng, K., et al. (2023)**. *Mass-Editing Memory in a Transformer*. International Conference on Learning Representations (ICLR 2023). [arXiv:2210.07229](https://arxiv.org/abs/2210.07229).
3. **Chen, Z., Zhan, H., Huang, Y., Wu, X., Deng, D., Weng, D., & Wu, Y. (2026)**. *KEditVis: A Visual Analytics System for Knowledge Editing of Large Language Models*. IEEE Transactions on Visualization and Computer Graphics (TVCG), vol. 32, no. 6, pp. 4818–4828. [arXiv:2603.29689v1](2603.29689v1.pdf).
4. **Geva, M., et al. (2021)**. *Transformer Feed-Forward Layers Are Key-Value Memories*. Empirical Methods in Natural Language Processing (EMNLP 2021). [arXiv:2012.14913](https://arxiv.org/abs/2012.14913).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
