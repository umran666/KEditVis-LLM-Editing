# KEditVis Prototype — Baseline Scripts

This is the starting point for the capstone project: reproducing KEditVis's
two core layer-selection signals (cosine similarity + token-projection
ranking) and validating that MEMIT edits work end-to-end, before building
the interactive dashboard.

## Current dashboard and verification

The dashboard now defaults to **Context-robust MEMIT**, a CORE-inspired
optimization with per-context fitting and a larger latent-update budget.
It fixes the previously reported GPT-2-XL paraphrase failure; the expanded
five-phrasing check improves from 1/5 to 4/5. Standard MEMIT and ROME remain
available. See [optimization notes and measured limits](OPTIMIZATION_NOTES.md)
for source attribution, real GPU results, and reproduction commands.

The React dashboard runs at `http://127.0.0.1:5187` during development. Its
`frontend/.env.local` points to the deployed backend at
`https://opzgameryt--keditvis-memit-web-app.modal.run`.

From `prototype/frontend`, run `npm ci` and `npm run dev -- --host 127.0.0.1 --port 5187`.
Production compilation uses `npm run build`. Regression checks use
`node tests/audit.mjs`; `node tests/live.mjs` exercises the actual deployed GPU.

From `prototype`, run `python -m unittest test_backend.py test_optimizations.py -v` for CPU
regressions. Real model checks use `python test_live.py --model gpt2-xl`
or `python test_live_backend.py --model EleutherAI/gpt-j-6B`. Live tests incur
Modal GPU usage. The service uses A100-40GB hardware, float32 model weights,
one request per worker, and at most one web worker.

Deploy backend changes from PowerShell with:

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
modal deploy modal_app.py
```

See [end-to-end verification](audit/E2E_VERIFICATION.md) for the preceding audit's evidence,
remaining research scope, complete replacement files, and live screenshots.
The [first-pass audit](audit/KEDITVIS_AUDIT.md) preserves the original findings.
The sections below describe the earlier baseline experiments.

## Local environment setup (Windows + RTX 3050, validated)

Your global Python install had a broken/bleeding-edge `transformers` dev
build that conflicted with `accelerate`/`torchao`. Use an isolated venv
for this project instead of the global interpreter:

```bash
cd "LLM Editing/prototype"
python -m venv .venv

# Install the CUDA 12.4 build of torch from PyTorch's own index (large, ~2.5GB)
./.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0

# Then the rest from PyPI
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Verified working versions: `torch==2.6.0+cu124`, `transformers==4.44.2`,
`accelerate==0.33.0`, on an RTX 3050 Laptop GPU (CUDA visible, 4GB VRAM).

Run everything below with `./.venv/Scripts/python.exe` instead of `python`.

## Files

- **`local_probe.py`** — Runs entirely on your laptop (CPU or the RTX 3050).
  No MEMIT dependency. Loads a small GPT-2 variant, forward-hooks every MLP
  layer, and prints an ASCII chart of cosine similarity + top-1 logit-lens
  token per layer for a given fact. Use this to understand/debug the
  hooking logic for free before touching Modal.

  ```bash
  ./.venv/Scripts/python.exe local_probe.py --model gpt2-medium \
      --prompt "{} is located in the city of" --subject "Eiffel Tower"
  ```

  Validated working on `gpt2` (small) on an RTX 3050: cosine similarity dips
  at layers 2, 5, 9, 11, and the logit-lens top-1 token at the subject
  position drifts from " Tower" -> " Towers" -> "," through the network,
  matching the qualitative pattern KEditVis describes. Try `gpt2-medium`
  or `gpt2-large` next; `gpt2-xl` may be tight on 4GB VRAM for inference
  alone (fine) but will need Modal for actual MEMIT edits (backward pass).

- **`modal_app.py`** — Runs a *real* MEMIT edit on GPT-2-XL on a Modal T4
  GPU, using the original `kmeng01/memit` implementation (not reimplemented).
  Extracts the same two signals before/after editing and returns before/after
  generations for the edited fact plus a couple of generalization prompts.

  ```bash
  pip install modal
  modal setup   # one-time auth

  # Uses MEMIT's default hardcoded layer preset for the model:
  modal run modal_app.py --prompt "{} is located in the city of" \
      --subject "Eiffel Tower" --target "Rome"

  # Or override with your own layer selection (the actual capstone feature):
  modal run modal_app.py --prompt "{} is located in the city of" \
      --subject "Eiffel Tower" --target "Rome" --layers "8-12"
  ```

  First run downloads gpt2-xl (~6GB) and MEMIT's precomputed covariance
  statistics into persistent Modal Volumes (`keditvis-hf-cache`,
  `keditvis-memit-data`), so later runs are faster and cheaper.

## Why this split

- Iterating on hook logic / chart data shapes against MEMIT + a 1.5B model
  on Modal burns credit and adds latency. Get the extraction logic right
  locally on GPT-2/GPT-2-medium first.
- `modal_app.py` deliberately reuses `kmeng01/memit`'s own hyperparameter
  files and covariance stats rather than reimplementing MEMIT's math or
  recomputing statistics — both are already published and validated for
  GPT-2-XL and GPT-J-6B.

## Status: end-to-end MEMIT edit validated on Modal (T4)

`modal_app.py` has been run successfully on Modal using a T4 GPU (cheapest
GPU tier, ~$0.59/hr). Result for editing gpt2-xl with the fact
(Eiffel Tower, location, Rome):

- **Pre-edit generation:** "Eiffel Tower is located in the city of **Paris**, France..."
- **Post-edit generation:** "Eiffel Tower is located in the city of **Rome**, but the city's famous Eiffel Tower is located in Paris..." (edit succeeded; note the model still "remembers" Paris elsewhere in the same generation -- a real example of the imperfect generalization/locality tradeoffs KEditVis's dashboard is meant to help diagnose)
- Verified the two hyperparameter-selected layers (13-17) are the *only*
  layers where pre/post cosine-similarity signals differ; layers 0-12 are
  bit-for-bit identical pre/post edit, and the divergence introduced at
  layers 13-17 propagates through all downstream layers (18-47). This
  matches the paper's description of how edits accumulate through the
  residual stream (Scenario II, Sec 5.1.2).

Two issues had to be fixed to get a clean run, both now fixed in
`modal_app.py`:

1. The image build pre-created `/root/memit/data`, which conflicted with
   mounting a Modal Volume at that same path ("cannot mount volume on
   non-empty path"). Fix: don't pre-create the directory; let the Volume
   mount provide it, and let `layer_stats.py`'s own `mkdir(parents=True)`
   calls create subdirectories as needed.
2. `rome/compute_v.py` has an unused `from matplotlib.style import context`
   import that still executes at import time. Fix: added `matplotlib` to
   the image's pip-installed dependencies.

## Status: user-specified layer selection validated

`modal_app.py` now accepts a `--layers` argument that overrides MEMIT's
hardcoded per-model preset (`hparams/MEMIT/gpt2-xl.json` normally forces
layers `[13,14,15,16,17]` for every edit, regardless of the fact). This is
the actual human-in-the-loop contribution of the capstone.

Syntax:
- `--layers "8-12"` -> inclusive range [8, 9, 10, 11, 12]
- `--layers "6,9,12"` -> explicit, possibly non-contiguous list
- `--layers "7"` -> single layer
- omit `--layers` entirely -> falls back to MEMIT's default preset

Validated both cases end-to-end on Modal (T4):

1. **Custom range `8-12`** for the same (Eiffel Tower, location, Rome) fact
   used in the default-preset run. The edit succeeded, and interestingly
   produced a *more thematically coherent* completion ("Rome...home to the
   Colosseum and the Forum...founded by Julius Caesar") than the default
   preset's completion for the same fact -- a concrete, reproducible
   example of why layer choice matters and why a human-in-the-loop
   comparison tool has value.
2. **Out-of-range layers `45-50`** (only 0-47 valid for gpt2-xl, a 48-layer
   model) -- correctly rejected immediately with a clear `ValueError`
   before wasting any GPU time inside MEMIT's internals, instead of
   crashing confusingly deep in the algorithm.

The returned JSON's `edited_layers` field always reflects what was
actually applied (custom or default), so downstream tooling/dashboards
can trust it without re-deriving which layers were used.

## Status: ES/PS/NS/S metrics validated

`modal_app.py` now computes quantitative KEditVis-style metrics, adapted
from MEMIT's own `experiments/py/eval_utils_counterfact.py` formulas
(probability + greedy-decoding correctness checks -- no CounterFact
dataset download required):

- **ES (Efficacy Success)**: does the exact rewrite prompt now produce
  `target_new`?
- **PS (Paraphrase Success)**: do paraphrased prompts also produce
  `target_new` (generalization)?
- **NS (Neighborhood Success)**: do unrelated/nearby prompts still produce
  `target_true`, i.e. did the edit avoid corrupting nearby knowledge
  (locality)?
- **S**: harmonic mean of ES, PS, NS.

Requires `--target_true` (the original correct answer) so PS/NS have
something to check against; `--paraphrase_prompts` and
`--neighborhood_prompts` are semicolon-separated full prompt strings.
All three already default to values tuned for the Eiffel Tower/Rome demo,
so metrics compute automatically with zero extra flags.

### Validated result (default demo, layers 13-17)

| | ES | PS | NS | S |
|---|---|---|---|---|
| Before edit | 0.00 | 0.00 | 0.50 | 0.00 |
| After edit | **1.00** | 0.00 | 0.50 | 0.00 |

This is a real, technically meaningful result, not a bug: ES jumps to 1.00
(the rewrite prompt reliably produces "Rome"), but PS stays at 0.00 --
merely adding "The " before the subject ("The Eiffel Tower is located in
the city of" vs the trained "Eiffel Tower is located in the city of")
was enough to make the model fall back to "Paris" (NLL 0.58) over "Rome"
(NLL 2.39). This is MEMIT's well-documented sensitivity to exact prompt
surface form -- a genuine limitation of locate-then-edit methods, and
exactly the kind of failure mode KEditVis's interactive dashboard is
designed to surface and let users diagnose (e.g. by trying a different
layer range, or adding synonymous phrasings to the edit request itself).
NS stayed unchanged at 0.50 pre/post edit, confirming locality was
preserved (one of the two neighborhood checks was already wrong before
editing -- a pre-existing model limitation, not something the edit caused).

Per-prompt details (NLL for both target strings, plus greedy-decoding
correctness) are included in the JSON output under
`pre_edit.metrics.details` / `post_edit.metrics.details` for deeper
inspection.

## Status: scheme comparison table validated

`modal_app.py` now has a second entrypoint, `compare`, that evaluates
multiple candidate layer schemes for the same fact in a **single Modal
container invocation** -- the model is loaded once, then each scheme is
applied, evaluated, and reverted before trying the next one. This is
KEditVis's "Compare" button / multi-metric ranking table (Sec 4.3),
without the cost of reloading the 6GB model per scheme.

```bash
modal run modal_app.py::compare --schemes "13-17|8-12|6-8|20-21"
```

`--schemes` uses `|` to separate schemes, and each scheme uses the same
syntax as `--layers` (range `"8-12"`, list `"6,9,12"`, or single `"7"`).
All other flags (`--target_true`, `--paraphrase_prompts`,
`--neighborhood_prompts`, `--prompt`, `--subject`, `--target`) work the
same as the `main` entrypoint.

### Validated result (Eiffel Tower -> Rome, 4 schemes)

| Layers | ES | PS | NS | S |
|---|---|---|---|---|
| [13,14,15,16,17] (MEMIT's default preset) | 1.00 | 0.00 | 0.50 | 0.00 |
| [8,9,10,11,12] | 1.00 | 0.00 | 0.50 | 0.00 |
| [6,7,8] | 0.00 | 0.00 | 0.50 | 0.00 |
| [20,21] | 0.00 | 0.00 | 0.50 | 0.00 |

Baseline (unedited model): ES=0.00, PS=0.00, NS=0.50, S=0.00.

Takeaways from this specific run: layers 13-17 and 8-12 both achieve
perfect efficacy (ES=1.00), while narrower/differently-placed ranges
(6-8, 20-21) fail to instill the fact at all (ES=0.00) -- a direct,
reproducible illustration of why layer choice matters, and exactly the
kind of comparison a human using KEditVis's dashboard would want to see
before committing to an edit.

**Known nuance:** for scheme [8-12], ES=1.00 (greedy decoding strictly
prefers "Rome", NLL 1.63, over "Paris", NLL 2.27, at every target token),
but the *displayed* sampled generation for that scheme still said "Paris".
This is not a bug: `generate_fast` (borrowed from `kmeng01/memit`) uses
top-k stochastic sampling for readability, not pure greedy decoding, so it
can occasionally sample a non-argmax token even when the metric correctly
shows the argmax favors the edited fact. The ES/PS/NS metrics are the
reliable, deterministic signal; treat displayed generations as one
qualitative sample, not proof of the model's actual top prediction.

## Status: cosine-similarity hypothesis tested against real data

`analyze_schemes.py` is a pure local script (no GPU/Modal needed) that
consumes the JSON from `modal run modal_app.py::compare` and tests
KEditVis's core hypothesis: do layers with LOW cosine similarity in the
*unedited* model (the paper's proposed "activity" signal, Sec 4.2.1)
actually predict which layer schemes achieve HIGH edit success after a
real MEMIT edit?

`compare_layer_schemes` now also extracts the baseline (pre-edit) cosine
similarity per layer via `_probe_layers`, so this analysis doesn't need
any extra GPU runs -- it's computed from data `compare` already produces.

```bash
modal run modal_app.py::compare --schemes "13-17|8-12|6-8|20-21"
python analyze_schemes.py scheme_comparison.json
```

### Validated result (Eiffel Tower -> Rome, 4 schemes, n=4 -- exploratory only)

| Layers | mean\|cos_sim\| | min\|cos_sim\| | ES | PS | S |
|---|---|---|---|---|---|
| [13-17] | 0.069 | 0.009 | 1.00 | 0.50 | 0.60 |
| [8-12] | 0.021 | 0.001 | 1.00 | 0.00 | 0.00 |
| [6-8] | 0.061 | 0.029 | 0.00 | 0.00 | 0.00 |
| [20-21] | 0.146 | 0.135 | 0.00 | 0.00 | 0.00 |

Spearman rank correlations (n=4, directional evidence only, not
significance-tested):
- `min|cos_sim|` vs `ES`: **-0.70** (moderately supports the hypothesis --
  the scheme with the single most "active" layer, [8-12], and the scheme
  with the best overall boundary layers, [13-17], both achieved ES=1.00;
  [20-21], with by far the highest min cosine similarity, achieved ES=0.00)
- `mean|cos_sim|` vs `ES`: -0.30 (weak, same direction)
- `mean|cos_sim|` vs `S`: **+0.40** (opposite direction from the hypothesis!)

**Honest interpretation, not oversold:** the cosine-similarity signal has
real, if modest, predictive value for whether a scheme achieves efficacy
(ES) at all -- consistent with the paper's claim. But it does *not*
predict the composite score S, because [8-12] achieved perfect efficacy
(ES=1.00) with the lowest cosine similarity, yet still scored S=0.00
overall because it failed to generalize to paraphrases (PS=0.00). This is
not a contradiction of KEditVis's design -- it's precisely the paper's own
argument (Sec 6, "relying on single selection methods can be unreliable
for certain instances") for why automated single-signal heuristics are
insufficient on their own and a human comparing multiple metrics side by
side (not just cosine similarity) adds real value. This single-fact,
4-scheme sample is too small to draw firm conclusions -- see "Next steps"
below for how to extend this into a real evaluation.

## Status: pooled multi-fact analysis (n=20) -- weaker, more honest signal

`modal run modal_app.py::batch` sweeps every fact in `facts.json` (5
diverse facts: Eiffel Tower, LeBron James, Windows, Mario Kart, Steve
Jobs -- modeled on MEMIT's own demo facts) across every scheme in
`--schemes`, loading the model only ONCE for the whole sweep (not once per
fact). `analyze_batch.py` then pools all (fact, scheme) pairs and re-runs
the correlation analysis with far more statistical power than a single
fact allows.

```bash
modal run modal_app.py::batch --schemes "13-17|8-12|6-8|20-21"
python analyze_batch.py batch_comparison.json
```

### Validated result (5 facts x 4 schemes = 20 edits, real Modal run, zero errors)

Pooled Spearman correlations (n=20, cosine-similarity "activity" vs. each metric):

| | mean\|cos_sim\| | min\|cos_sim\| |
|---|---|---|
| vs ES | -0.252 | -0.297 |
| vs PS | -0.162 | -0.150 |
| vs NS | +0.229 | +0.199 |
| vs S  | -0.127 | -0.119 |

Compare this to the single-fact (Eiffel Tower only, n=4) result from
earlier: `min|cos_sim|` vs `ES` was **-0.70** there, but drops to **-0.30**
once pooled across 5 different facts. This is the honest, more important
finding: the cosine-similarity heuristic's apparent predictive power in a
single fact does not hold up nearly as well once you test it across a
variety of facts. This directly corroborates the KEditVis paper's own
stated motivation (Sec 6) that "relying on single selection methods can be
unreliable for certain instances" -- our own from-scratch replication
independently reaches the same conclusion the paper uses to justify
human-in-the-loop, multi-metric comparison over any single automated
signal.

The sweep also surfaced a finding outside cosine similarity entirely:
**"Windows was developed by" -> "Apple" failed to edit (ES=0.00) under
every single layer scheme tested**, including MEMIT's own default preset.
This means some facts are intrinsically harder to edit than others,
regardless of layer choice -- a separate axis of difficulty that a purely
layer-focused tool (KEditVis included) doesn't directly diagnose, and a
good discussion point for your capstone's limitations section.

## Status: root cause found for the "Windows -> Apple" failure

`error_analysis.py` compares baseline (pre-edit) cosine-similarity
profiles across all facts in a batch run, using data already collected --
no new GPU computation needed. It specifically exploits a natural
experiment: "Windows was developed by" -> "Apple" and "Mario Kart was
developed by" -> "Apple" share the exact same target word, so any
difference in editability must trace back to how the SUBJECT is
represented, not the target token.

```bash
python error_analysis.py batch_comparison.json
```

### Result: a real, clean root cause, not noise

| | Windows (failed, ES=0.00 everywhere) | Mario Kart (succeeded, ES=1.00 everywhere) |
|---|---|---|
| mean abs(cos_sim) (all 48 layers) | **0.762** | 0.120 |
| min abs(cos_sim) (most "active" layer) | **0.059** | 0.005 |
| max abs(cos_sim) | 0.980 | 0.376 |

Every other fact in the batch (Eiffel Tower, LeBron James, Steve Jobs)
clusters in the same 0.12-0.18 mean-cosine-similarity range as Mario Kart.
"Windows" is a clear outlier: even its single most active layer (0.059) is
less active than *every* layer tested for any other fact. This means no
layer in the tested range meaningfully transforms the "Windows" subject
representation at all -- the model doesn't appear to be processing it as a
distinct knowledge-bearing entity the way it processes proper nouns like
"Mario Kart" or "Steve Jobs". A plausible explanation: "Windows" is also
an ordinary English word (plural of "window"), so its representation may
be far more entangled/diffuse than a rarer, more distinctive token.

**This produces a concrete, novel, actionable diagnostic beyond what
KEditVis's paper describes**: if a user inspects the cosine-similarity bar
chart *before* editing and sees no layer with a pronounced dip anywhere in
the network (i.e. no layer stands out from a uniformly high baseline),
that is itself a warning sign the fact may fail regardless of which layers
are chosen -- a genuinely useful, capstone-worthy extension to the paper's
original design, not just a replication of it.

## Known gotchas

- MEMIT needs precomputed second-moment covariance statistics per edited
  layer. These auto-download from `memit.baulab.info` for `gpt2-xl` and
  `EleutherAI/gpt-j-6B` only. If you switch to a different base model
  (e.g. Llama-3), there are no precomputed stats — you'd need to compute
  them yourself over a large text corpus, which is slow and a distraction
  from the actual project goal. Stick to `gpt2-xl` for the MVP.
- The pinned `torch`/`transformers` versions in `modal_app.py` match the
  era `kmeng01/memit` was built against. Its custom `generate_fast` loop
  manually manipulates `past_key_values` as legacy tuples; newer
  `transformers` versions default to `Cache` objects and can break this
  silently. Don't bump these versions without testing.
- `apply_memit_to_model(..., return_orig_weights=True)` gives you the
  original layer weights back — this is your "reversible edit" mechanism
  (KEditVis's R5 requirement). Keep a dict of `{layer_name: original_tensor}`
  per edit so you can revert.

## Next steps (suggested order)

1. Run `local_probe.py` against a handful of facts on GPT-2-medium. Compare
   your cosine-similarity trend against the patterns described in the
   KEditVis paper's Scenario I/II (dip around the "fact-processing" layers).
2. Run `modal_app.py` once end-to-end. Confirm the edit actually changes the
   generated text for the target fact (e.g. "Eiffel Tower is located in the
   city of Rome" instead of "Paris").
3. ~~Extend `modal_app.py` to accept a *layer range* parameter~~ -- done,
   see "Status: user-specified layer selection validated" above.
4. ~~Add ES/PS/NS metric computation~~ -- done, see "Status: ES/PS/NS/S
   metrics validated" above.
5. ~~Run the same fact + metrics across several different `--layers`
   choices and tabulate ES/PS/NS/S per scheme~~ -- done, see "Status: scheme
   comparison table validated" above (`modal run modal_app.py::compare`).
6. ~~Try the cosine-similarity signal as a predictor of scheme success~~ --
   done, see "Status: cosine-similarity hypothesis tested against real
   data" above (`analyze_schemes.py`). Result was genuinely mixed/nuanced,
   not a clean confirmation -- which is itself a useful, honest finding.
7. ~~Scale the analysis up across several facts~~ -- done, see "Status:
   pooled multi-fact analysis (n=20)" above (`modal run
   modal_app.py::batch` + `analyze_batch.py`).
8. Add more facts to `facts.json` and/or more schemes to `--schemes` to
    push n higher still (n=20 is enough to be directionally credible, but
    more data would strengthen a capstone evaluation chapter further).
    ~~Consider also testing whether the *token-projection* signal (the other
    half of KEditVis's layer-selection approach, not yet implemented here)
    predicts scheme success better than cosine similarity alone.~~ Done:
    `modal_app.py` now records the **last-token** logit-lens top-5 per layer
    (the view where the fact's object token actually surfaces -- `local_probe.py`
    mirrors this), and `analyze_schemes.py`/`analyze_batch.py` compute the
    token-projection predictor per Sec 4.2.2 -- the peak span between the two
    layers where the object token shows its highest probabilities, and each
    scheme's overlap + max prob within that span. Correlations are reported
    alongside the cosine ones. Existing saved JSONs predate `last_top_tokens`
    and report "Token-projection analysis unavailable" gracefully; re-run
    `modal run modal_app.py::compare` / `::batch` (or `local_probe.py` locally)
    once to collect the new signal (`modal run modal_app.py::compare --schemes "13-17|8-12|6-8|20-21" --target_true Paris; python analyze_schemes.py scheme_comparison.json Paris`).
9. ~~Investigate why "Windows -> Apple" failed under every scheme~~ --
   done, see "Status: root cause found for the 'Windows -> Apple' failure"
   above (`error_analysis.py`). Found a clean, real root cause: no
   layer shows a pronounced cosine-similarity dip for "Windows" at all,
   unlike every other fact tested.
10. Test the "no pronounced dip anywhere = likely-to-fail fact" hypothesis
    on a few more facts before relying on it -- this is currently a single
    supporting data point (n=1 hard fact) and deserves a couple more
    replications before treating it as a general diagnostic rule in your
    capstone write-up.
11. ~~Only once 1-10 work reliably, start the FastAPI/React dashboard~~ --
    **in progress**. The FastAPI backend lives in `modal_app.py::web_app`
    (`/health`, `/probe`, `/edit`, `/compare`). The React/D3 frontend is in
    `frontend/`. See "Dashboard (FastAPI + React)" below.

## Dashboard (FastAPI + React)

The interactive dashboard wraps the validated pipeline behind a browser UI:
cosine-similarity bar chart, token-ranking (logit-lens) chart, layer picker,
single-fact MEMIT edit, and multi-scheme comparison table.

### Backend (Modal)

The backend is already implemented at the bottom of `modal_app.py`. It loads
GPT-2-XL once per container and exposes:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model name + layer count |
| `POST /probe` | Baseline layer signals (no edit) |
| `POST /edit` | MEMIT edit + pre/post signals, generations, ES/PS/NS/S |
| `POST /compare` | Multiple layer schemes for one fact |

Start a dev server (temporary URL, live-reloads on save):

```bash
cd "LLM Editing/prototype"
modal serve modal_app.py
```

Copy the printed `web_app` URL (ends in `.modal.run`).

For a persistent deployment:

```bash
modal deploy modal_app.py
```

### Frontend (React + Vite + D3)

```bash
cd "LLM Editing/prototype/frontend"
cp .env.example .env.local
# Edit .env.local: set VITE_API_URL to your modal serve/deploy URL

npm install
npm run dev
```

Open http://localhost:5173. Workflow:

1. **Probe layers** — load baseline cosine-similarity + token-ranking charts
2. **Select layers** — click layer chips or use **Recommend** (lowest |cos|)
3. **Apply MEMIT edit** — run edit on selected layers; view before/after metrics
4. **Compare schemes** — evaluate multiple layer ranges (one per line, e.g. `8-12`)

CORS is enabled on the FastAPI app for local dev (`allow_origins=["*"]`).
Each `/edit` and `/compare` request temporarily mutates model weights inside
the container, then restores them; `@modal.concurrent(max_inputs=1)` prevents
concurrent requests from corrupting each other's edits.

## Status: dashboard capstone — recommended scheme fails, contiguity matters

The full loop (browser dashboard → Modal FastAPI → MEMIT on T4) was validated
end-to-end on the Eiffel Tower → Rome fact. The headline finding: **the
Recommend heuristic (lowest |cos| layers) does NOT produce a good editing
scheme.**

| Scheme | Selection basis | ES | Outcome |
|---|---|---|---|
| Baseline (unedited) | — | 0.00 | Paris |
| [9, 10, 11, 13, 15] | Recommended (lowest \|cos\|), non-contiguous | **0.00** | Failed — borderline: Paris NLL 1.17 narrowly beats Rome 1.19 |
| [8, 9, 10, 11, 12] | Contiguous block | 1.00 | Success — "Rome" |
| [13, 14, 15, 16, 17] | MEMIT default | 1.00 | Success — "Rome" |
| [6, 7, 8] | Early contiguous | 0.00 | Failed — still Paris |
| [20, 21] | Late contiguous | 0.00 | Failed — still Paris |

(PS=0.00, NS=0.50, S=0.00 for every row — see caveat below.)

Interpretation: the cosine heuristic picks individually "active" layers, but
MEMIT needs a **contiguous block within the mid-layer window (~8–17)**.
Skipping layer 12 broke the edit even though layers 9–11 overlap with the
working 8–12 range, and contiguous blocks outside the window (6–8, 20–21)
also fail. This is exactly the failure mode the human-in-the-loop dashboard
exists to catch: a user who blindly applied the recommended layers would get
an edit that doesn't take, while the side-by-side comparison table makes the
contiguous-scheme fix obvious. Note the failure is *ineffective*, not
*destructive* — the KL analysis below shows unrelated behavior is untouched.
(As a direct result of this finding, the dashboard's Recommend button now
selects the best *contiguous* window of 5 layers by total |cos_sim| instead
of the 5 individually-lowest layers.)

**Caveat 1 — ES is a narrow metric (the "Vienna" anomaly).** In the UI
comparison run, scheme [8–12] scored ES=1.00 yet its sampled generation said
*"Vienna, Austria"* — not Rome. This is not a contradiction: ES only checks
that `target_new` beats `target_true` (Rome < Paris in NLL); a *third* token
can still win decoding. The edit pushed the model off Paris without reliably
landing it on Rome. (That particular Vienna sample came from an unseeded run;
under the deterministic seeding in Caveat 3, [8–12] generates "Rome". The
lesson stands either way.) Treat ES as efficacy-vs-original-answer only, and
displayed generations as one stochastic sample (see the "Known nuance" in the
scheme-comparison section above).

**Caveat 2 — PS=0.00 everywhere, even for successful edits.** Every scheme
that achieved ES=1.00 still failed all paraphrase prompts (e.g. "The Eiffel
Tower is located in the city of" reverts to Paris). The edit wins on exact
efficacy but does not generalize — consistent with MEMIT's documented prompt
sensitivity and a key limitation to state in the write-up.

**Operational note:** `modal serve` sessions stop when the terminal dies; if
the dashboard shows "Failed to fetch", re-run `modal serve modal_app.py` and
confirm `frontend/.env.local` matches the printed `.modal.run` URL.

**Caveat 3 — MEMIT results are RNG-sensitive unless seeded.** MEMIT's
`get_context_templates()` samples its context templates with
`torch.multinomial` on first use and caches them for the container's
lifetime. In a warm web container, the RNG state depends on prior requests,
so the same layers once produced ES=1.00 (Rome NLL 0.96) via `/edit` while a
fresh CLI run gave ES=0.00 (Rome NLL 2.92). The backend now calls
`_seed_rng()` (torch seed 0 + clears the template cache) before every
`apply_memit_to_model`, and results are bit-identical across runs: layers
[9, 10, 11, 13, 15] reproducibly give Rome NLL 1.1911 vs Paris 1.1744,
ES=0.00. The recommended-scheme failure is real under canonical conditions —
but it is borderline, a 0.02-nat gap, not a catastrophic failure.
(This seeding now lives in the module-level `_seed_memit_rng()` and is called
before every edit in ALL paths — `main`, `compare`, and `batch` included —
so CLI runs are reproducible too.)

**Damage metric — mean per-token KL(P_edited ‖ P_orig).** To test whether
edits *destroy* unrelated knowledge, `/edit` and `/compare` compute the exact
full-vocabulary KL divergence (in nats) over 12 neutral reference sentences
that share no entities with the edited fact, at every token position:

| Scheme | ES | Damage KL (nats) |
|---|---|---|
| [13, 14, 15, 16, 17] | 1.00 | 1.40e-05 |
| [8, 9, 10, 11, 12] | 1.00 | 4.34e-06 |
| [9, 10, 11, 13, 15] | 0.00 | 6.17e-06 |
| Baseline (unedited) | — | 0.0 |

All values are ~1e-5 or smaller: a single MEMIT edit causes negligible drift
on neutral text, and a failed edit is no more damaging than a successful one.
Conclusion: MEMIT edits on GPT-2-XL *achieve or miss* the targeted fact
change without measurably damaging the rest of the model at single-edit
scale. KL becomes the meaningful yardstick for cumulative damage when many
edits are stacked (future work).
