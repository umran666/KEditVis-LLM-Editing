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
4. [Repository Structure](#repository-structure)
5. [Quickstart & Installation](#quickstart--installation)
6. [Running the Test Suites](#running-the-test-suites)
7. [Cloud GPU Backend Deployment](#cloud-gpu-backend-deployment)
8. [Benchmarking & Reproduction](#benchmarking--reproduction)
9. [Academic Reconciliation](#academic-reconciliation)
10. [References](#references)

---

## Overview

Large Language Models (LLMs) store vast amounts of factual knowledge within their feedforward weight matrices ($W_{out}$). When facts become outdated or require correction, retraining the entire network is computationally prohibitive. Locate-then-edit techniques—principally **ROME** (Rank-One Model Editing) and **MEMIT** (Mass-Editing Memory in a Transformer)—treat feedforward layers as linear associative memories, modifying weights to store new associations:

$$W_{new} = W_0 + \Delta W$$

### The Problem with Static Presets
Production editing pipelines typically rely on rigid, model-wide layer presets (e.g., layers $[13..17]$ for GPT-2-XL, layers $[3..7]$ for GPT-J-6B). However, factual representations do not localize identically across different entities and semantic categories:
- **Mislocated Edits**: Editing outside the factual storage locus fails to generalize or destroys model coherence.
- **Parameter Explosion**: Unconstrained random layer selection induces up to **11.3x higher relative Frobenius parameter drift** ($\|\Delta W\|_F / \|W_0\|_F$), destabilizing nearby knowledge.
- **Paraphrase Fragility**: Standard single-context MEMIT often succeeds on the literal target prompt but fails under varied phrasing.

### The KEditVis Solution
**KEditVis** is an end-to-end interactive visual analytics system that integrates human domain judgment directly into the editing loop. By extracting layer-wise residual variance $\text{Var}_{\text{dim}}(h_l[t])$, directional cosine similarity dips, and logit-lens vocabulary projections in real time, KEditVis guides practitioners to stable layer bands, monitors parameter drift, and provides instantaneous, zero-drift transactional rollback.

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
2. **Multi-Context Vocabulary Projections**: Traces top-1 greedy token predictions across all layers via unembedding matrix projection ($h_l W_U$).
3. **Multi-Metric Comparative Workspace**: Evaluates candidate layer selections and optimization profiles side-by-side across:
   - **Efficacy Score (ES)**: Preference probability $P(\text{target}) > P(\text{original})$.
   - **Paraphrase Score (PS)**: Generalization across unseen rephrasings.
   - **Neighborhood Score (NS)**: Locality preservation on unedited sibling subjects.
4. **Frobenius Parameter Drift & KL Scatter**: Plots tensor weight deviation ($\|\Delta W\|_F$) against hidden state representation divergence ($D_{KL}$), detecting localized over-fitting before committing changes.

---

## Key Empirical Findings

Extensive evaluations across the standardized **CounterFact** benchmark on NVIDIA A100-SXM4-40GB hardware yielded concrete insights:

| Configuration | Layer Range | Efficacy (ES) | Paraphrase (PS) | Locality (NS) | Mean Score ($S$) | Relative Drift ($\|\Delta W\|_F / \|W\|_F$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Preset (MEMIT)** | $[13..17]$ | **1.00** | 0.80 | **0.90** | **0.631** | 0.0095 |
| **Telemetry-Guided** | $[14..18]$ | **1.00** | 0.80 | 0.85 | 0.621 | 0.0101 |
| **Arbitrary / Mislocated** | $[40..44]$ | 0.80 | 0.40 | 0.80 | 0.435 | **0.1071** *(11.3x explosion)* |
| **Context-Robust MEMIT** | $[13..17]$ | **1.00** | **0.90** | 0.85 | 0.612 | 0.0169 *(+77.9% budget)* |

### Insights
- **Telemetry Layer Selection Acts as a Critical Safety Filter**: While heuristic telemetry selection achieves parity with the static preset ($S = 0.621$ vs $0.631$), its primary function is preventing catastrophic failure bands. Arbitrary layer selection causes an **11.3x explosion in parameter drift** ($\text{Rel-Frob} = 0.1071$) with severe degradation in paraphrase generalization ($PS = 0.40$).
- **Context-Robust MEMIT Rescues Fragile Edits**: Expanding the latent update budget (40 steps, clamp 1.5) and applying multi-context fitting raises paraphrase generalization from 0.80 to 0.90, successfully resolving previously impossible hard edits (e.g., *Wellington $\to$ Sheffield*).
- **Transactional Rollback Guarantees Safe Exploration**: In-memory transactional weight snapshots guarantee $0.000$ residual weight drift upon rollback, verified across all tests and live runs.

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
3. **Wang, C., et al. (2024)**. *KEditVis: Interactive Visual Analytics for Knowledge Editing in Large Language Models*. IEEE Transactions on Visualization and Computer Graphics (TVCG).
4. **Geva, M., et al. (2021)**. *Transformer Feed-Forward Layers Are Key-Value Memories*. Empirical Methods in Natural Language Processing (EMNLP 2021). [arXiv:2012.14913](https://arxiv.org/abs/2012.14913).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
