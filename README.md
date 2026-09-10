# KEditVis: Interactive Visual Analytics for Human-in-the-Loop Knowledge Editing in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![React: 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![Modal: A100](https://img.shields.io/badge/Cloud%20GPU-NVIDIA%20A100--40GB-76b900.svg)](https://modal.com/)
[![Status: Verified Pass](https://img.shields.io/badge/Evaluation-100%25%20Verified%20Pass-brightgreen.svg)](EVALUATION.md)

**Capstone Project | Batch A8-12 | Mohan Babu University, Tirupati**  
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
Production editing pipelines typically rely on rigid, model-wide layer presets (e.g., layers [13..17] for GPT-2-XL, layers [3..7] for GPT-J-6B). However, factual representations do not localize identically across different entities and semantic categories:
- **Mislocated Edits**: Editing outside the factual storage locus fails to generalize or destroys model coherence.
- **Parameter Explosion**: Unconstrained random layer selection induces up to **11.3x higher relative Frobenius parameter drift** (`||ΔW||_F / ||W_0||_F`), destabilizing nearby knowledge.
- **Paraphrase Fragility**: Standard single-context MEMIT often succeeds on the literal target prompt but fails under varied phrasing.

### The KEditVis Solution
**KEditVis** is an end-to-end interactive visual analytics system that integrates human domain judgment directly into the editing loop. By extracting layer-wise residual variance `Var_dim(h_l[t])`, directional cosine similarity dips, and logit-lens vocabulary projections in real time, KEditVis guides practitioners to stable layer bands, monitors parameter drift, and provides instantaneous, zero-drift transactional rollback.

---

## System Architecture

```
                      +------------------------------------------+
                      |         KEditVis React Dashboard         |
                      |   (Vite + TailwindCSS + D3.js Visuals)   |
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
1. **Layer Telemetry Strip**: Displays layer-by-layer residual variance and cosine similarity dips across subject tokens, pinpointing the critical information-processing layers.
2. **Multi-Context Vocabulary Projections**: Traces top-1 greedy token predictions across all layers via unembedding matrix projection (`h_l · W_U`).
3. **Multi-Metric Comparative Workspace**: Evaluates candidate layer selections and optimization profiles side-by-side across:
   - **Efficacy Score (ES)**: Preference probability `P(target) > P(original)`.
   - **Paraphrase Score (PS)**: Generalization across unseen rephrasings.
   - **Neighborhood Score (NS)**: Locality preservation on unedited sibling subjects.
4. **Frobenius Parameter Drift & KL Scatter**: Plots tensor weight deviation (`||ΔW||_F`) against hidden state representation divergence (`D_KL`), detecting localized over-fitting before committing changes.

---

## Key Empirical Findings

Extensive evaluations across the standardized **CounterFact** benchmark on NVIDIA A100-SXM4-40GB hardware yielded concrete insights:

| Configuration | Layer Range | Efficacy (ES) | Paraphrase (PS) | Locality (NS) | Mean Score (S) | Relative Drift (‖ΔW‖_F / ‖W_0‖_F) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Preset (MEMIT)** | [13..17] | **1.00** | 0.80 | **0.90** | **0.631** | 0.0095 |
| **Telemetry-Guided** | [14..18] | **1.00** | 0.80 | 0.85 | 0.621 | 0.0101 |
| **Arbitrary / Mislocated** | [40..44] | 0.80 | 0.40 | 0.80 | 0.435 | **0.1071** *(11.3x explosion)* |
| **Context-Robust MEMIT** | [13..17] | **1.00** | **0.90** | 0.85 | 0.612 | 0.0169 *(+77.9% budget)* |

### Insights
- **Telemetry Layer Selection Acts as a Critical Safety Filter**: While heuristic telemetry selection achieves parity with the static preset (S = 0.621 vs 0.631), its primary function is preventing catastrophic failure bands. Arbitrary layer selection causes an **11.3x explosion in parameter drift** (Rel-Frob = 0.1071) with severe degradation in paraphrase generalization (PS = 0.40).
- **Context-Robust MEMIT Rescues Fragile Edits**: Expanding the latent update budget (40 steps, clamp 1.5) and applying multi-context fitting raises paraphrase generalization from 0.80 to 0.90, successfully resolving previously impossible hard edits (e.g., *Wellington → Sheffield*).
- **Transactional Rollback Guarantees Safe Exploration**: In-memory transactional weight snapshots guarantee 0.000 residual weight drift upon rollback, verified across all tests and live runs.

---

## Core Differences: 2603.29689v1.pdf (Chen et al., TVCG 2026) vs. Our Implementation

The theoretical foundation of this project originates from the base paper:  
*Z. Chen et al., "KEditVis: A Visual Analytics System for Knowledge Editing of Large Language Models," IEEE Transactions on Visualization and Computer Graphics (TVCG), vol. 32, no. 6, pp. 4818–4828, June 2026 ([arXiv:2603.29689v1](2603.29689v1.pdf)).*

While Chen et al. introduced the visual analytics workflow, our capstone project extends their theoretical design into an operational, mathematically validated, and open-source full-stack platform. Below are the core technical, algorithmic, and empirical differences:

| Dimension | Base Paper (`2603.29689v1.pdf`) | Our Implemented System (`KEditVis`) |
| :--- | :--- | :--- |
| **Code Availability & Reproduction** | Theoretical academic publication. Full interactive dashboard and backend were **unreleased / proprietary**. | **Complete production open-source system**: React 19 + Vite frontend, FastAPI backend on NVIDIA A100 GPU, automated test suites, and standalone CLI probe. |
| **Telemetry Signals** | **Cosine similarity only** (`cos(x_in, x_out)`) and logit-lens token ranks. | **Cosine similarity + Layer-wise Residual Variance** (`Var_dim(h_l[t])` and delta variance `Var_dim(h_l - h_{l-1})`) with interactive signal switching. |
| **Editing Algorithms** | Standard ROME and standard MEMIT only. | Standard ROME, standard MEMIT, and **Context-Robust MEMIT** (multi-context fitting, consistency loss, expanded update budget). |
| **Paraphrase Generalization** | Fragile under standard MEMIT. If an edit fails generalization, user must hunt for different layers. | **Context MEMIT rescues fragile edits**: Paraphrase generalization jumps from 0.80 to 0.90 (4/5 phrasings on hard facts like *Wellington → Sheffield*). |
| **Drift & Safety Measurement** | Relied purely on **stochastic 2D t-SNE plots** for "global impact" (qualitative, visual only). | **Exact Frobenius norm parameter drift** (‖ΔW‖_F, relative drift) + hidden-state L2 distance and KL divergence on a quantitative scatter plot. |
| **Transactional Rollback** | Conceptual concept; no concrete state-management or memory guarantees specified. | **Bit-exact in-memory weight snapshots** guaranteeing verified **0.000 residual parameter drift** upon rollback. |
| **Empirical Discovery** | Implied that dynamic/human layer selection consistently beats fixed presets. | **Scientific Reality Reconciled**: Telemetry acts as a **safety filter** preventing catastrophic failure (11.3x parameter explosion on random layers), achieving parity with static presets (S = 0.621 vs 0.631). |
| **Diagnostic Diagnostics** | No root-cause analysis for facts that fail under every layer scheme. | Implemented [`prototype/error_analysis.py`](prototype/error_analysis.py) proving *Windows → Apple* fails due to flat subject representations (mean abs(cos) = 0.762). |

### Key Architectural Extensions

1. **Residual Variance Operationalization**: The base paper measured layer activity exclusively through cosine similarity between MLP inputs and outputs. Our implementation operationalized feature-wise hidden channel variance `Var_dim(h_l[t])` and delta variance `Var_dim(h_l - h_{l-1})`, providing mathematically sound, scale/shift-invariant signals with interactive toggle controls in the UI.
2. **Context-Robust Optimization**: To address single-context MEMIT brittleness, we engineered a local CORE-inspired multi-context optimization engine in `editing_optimizations.py` that fits across multiple diverse prefixes and applies consistency regularization.
3. **Rigorous Parameter Drift Quantification**: While the paper relied on stochastic t-SNE projections that mask weight matrix destruction, our system computes exact tensor Frobenius norms (`||ΔW||_F`), proving that unconstrained layer selection causes an 11.3x explosion in parameter corruption.
4. **Empirical Grounding**: Rather than claiming speculative superiority, our controlled 10-fact CounterFact benchmarks scientifically demonstrate that telemetry layer selection functions primarily as an essential guardrail against destructive out-of-band layers.

---

## Repository Structure

```
.
├── Knowledge_Editing_LLMs.docx         # Original submission abstract (baseline, strictly unmodified)
├── Knowledge_Editing_LLMs_Final.docx   # Reconciled final abstract with empirical benchmark findings
├── EVALUATION.md                       # Comprehensive empirical evaluation & claim reconciliation
├── README.md                           # Repository documentation (this file)
└── prototype/
    ├── modal_app.py                    # Production FastAPI backend deployed on Modal A100 GPU
    ├── editing_optimizations.py        # Context-robust MEMIT, loss functions & Frobenius drift
    ├── layer_selection.py              # Telemetry-based layer scoring and selection heuristics
    ├── local_probe.py                  # Standalone CLI probe for local GPU/CPU inspection
    ├── run_experiments.py              # Automated CounterFact benchmark execution pipeline
    ├── test_backend.py                 # Core backend unittests (FastAPI routes, rollback, invariance)
    ├── test_optimizations.py            # Optimization unittests (multi-context fitting, projections)
    ├── test_live.py                    # Live GPU integration verification suite
    ├── error_analysis.py               # Empirical diagnostic tool for hard/failing facts
    ├── export_doc.py                   # Automated docx generator matching university styling
    ├── verify_manifest.py              # Cryptographic SHA-256 verification manifest generator
    ├── data/
    │   └── benchmark_manifest.json     # Standardized 10-fact CounterFact evaluation dataset
    ├── audit/
    │   └── evaluation/
    │       ├── verification.json       # Cryptographic SHA-256 manifest of core artifacts
    │       ├── summary.json            # Aggregated benchmark metrics with bootstrap 95% CIs
    │       └── raw_results.json        # Raw per-fact execution logs
    └── frontend/                       # Interactive React 19 visual analytics dashboard
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

pip install -r requirements.txt
```

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

*Expected output: `Ran 22 tests in ~22s ... OK`*

### Frontend Browser Regressions (Puppeteer)

Run the full end-to-end browser regression suite:

```bash
cd prototype/frontend
node tests/audit.mjs
```

*Expected output: All 15 browser test groups passed with 0 errors.*

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
VITE_API_BASE_URL=https://<your-username>--keditvis-memit-web-app.modal.run
```

---

## Benchmarking & Reproduction

To reproduce the full 10-fact CounterFact benchmark matrix across static, telemetry, arbitrary, and context-robust conditions:

```bash
cd prototype
python run_experiments.py --live
```

To regenerate the document and update cryptographic verification manifests:

```bash
# Generate the updated submission document
python export_doc.py

# Cryptographically verify and hash all core artifacts
python verify_manifest.py
```

---

## Academic Reconciliation

The initial project proposal set ambitious benchmarks for automated layer selection. Through rigorous empirical testing on real A100 hardware, our findings provide a more nuanced, scientifically honest contribution:
- Rather than outperforming optimized static baselines across every single metric, **telemetry-guided selection operates as a robust guardrail**, preventing the catastrophic drift and failure modes caused by arbitrary layer choice.
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
