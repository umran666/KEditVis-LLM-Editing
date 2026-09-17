# Context MEMIT optimization

Implementation date: 2026-09-10. This document supersedes the optimization status
in the historical audit reports; those reports and their raw results are retained.

## Outcome and scope

The reported GPT-2-XL failure was reproduced with standard MEMIT, layers 13-17,
editing Eiffel Tower from Paris to Rome. The frozen `context-v3` profile fixes
that specific tested paraphrase without feeding it to the optimizer. It is the
dashboard's default MEMIT profile. Standard MEMIT and ROME remain selectable.
The HTTP API defaults to `optimization: "standard"` for compatibility; request
`optimization: "context"` explicitly for the new behavior. ROME rejects the
context profile. Both `/edit` and `/compare` return the profile and its revision.
The standalone legacy Modal CLI functions still run standard MEMIT.

This is a CORE-inspired MEMIT adaptation, not an implementation or reproduction
of the full AlphaEdit, AnyEdit, or official CORE benchmark results.

## Sources and choice

All three requested repositories were inspected at pinned revisions:

| Repository | Revision | Relevant idea and decision |
| --- | --- | --- |
| [EasyEdit CORE](https://github.com/zjunlp/EasyEdit/blob/14cea8245f06715684592ab55184939b99d70784/easyeditor/models/core/compute_z.py) | `14cea8245f06715684592ab55184939b99d70784` | Adds consistency across prefix contexts to latent optimization. Adopted as an explicitly modified objective. |
| [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit/tree/b84624f44dfe8fc6cd9e41df916c44124a0c46dc) | `b84624f44dfe8fc6cd9e41df916c44124a0c46dc` | Projects edits into a preserved-knowledge null space. Deferred: this app restores each edit, while a sequential-edit evaluation and projection preparation are separate work. |
| [AnyEdit](https://github.com/jianghoucheng/AnyEdit/tree/057a77f185f7ffb55818f6bd9add37f43bb447e7) | `057a77f185f7ffb55818f6bd9add37f43bb447e7` | Iteratively handles extended targets through autoregressive decomposition. Deferred: the reported failure concerns a one-token target, Rome. |
| [MEMIT](https://github.com/kmeng01/memit/tree/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b) | `80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b` | Reuses its model-specific hyperparameters, context generator, covariance statistics, tracing helpers, and covariance-regularized weight solve. |

Source clones are under `audit/references/`. Production imports only the pinned
MEMIT package and the local `editing_optimizations.py`; no EasyEdit dependency
tree was added. Attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Implementation

`editing_optimizations.py` optimizes a shared latent displacement delta over the
upstream bare prompt and five model-generated prefix contexts. For the next
three available transformer blocks, it measures the subject-position hidden
states H and adds:

```
consistency = mean((H - mean(H, context_axis)) ** 2)
loss = target_NLL + upstream_KL_penalty + upstream_delta_penalty
       + 0.01 * consistency
```

This variance is normalized by layers, contexts, and hidden dimensions. Official
CORE uses a differently scaled pairwise expression, so its regularization values
cannot be transferred directly. The local profile sets a minimum clamp factor
of 1.5 and a minimum gradient-step budget of 40. Larger upstream settings are
preserved. The original learning rate, covariance penalty, and layer defaults
are retained. Early termination remains allowed.

> **Provenance note (2026-09-17).** The optimizer loop used to break on its final
> iteration *before* calling `backward()`/`step()`, so a configured budget of 40
> applied only **39** gradient updates. That off-by-one is fixed: 40 now means 40.
> The committed benchmark evidence in `audit/evaluation/` and the frozen results
> under `audit/optimization/` were produced under the old behaviour (39 effective
> updates) and have **not** been regenerated, because doing so requires a live
> GPU run. Treat those numbers as "40-step profile, pre-fix build".
>
> To reproduce them with the fixed code, set the budget to 39 — the exact
> equivalent of the old build's 40:
>
> ```python
> OPTIMIZATION_PROFILES["context_v3"]["minimum_gradient_steps"] = 39
> ```
>
> See [`audit/README.md`](audit/README.md) for the full evidence index and this
> caveat in context.

Instead of averaging all keys into a single key, the local adaptation retains
each context's key and its own original output plus delta. The bare context
has half the total solve weight and the generated-context group has the other
half, following the upstream grouping. Each column is scaled by the square root
of its weight before the covariance solve:

```
K = context_keys * sqrt_weights
R = (context_targets - current_context_outputs) * sqrt_weights
A = solve(lambda * C + K @ K.T, K)
delta_W = R @ A.T / remaining_layers
```

The total context weight per fact is one; adding contexts does not implicitly
multiply the strength relative to the covariance penalty. Weight orientation
uses the upstream GPT-2/GPT-J shape adapter. The vocabulary projection supports
both GPT-2's tied embedding and GPT-J's linear head.

Evaluation paraphrases, neighborhood prompts, and damage prompts are excluded
from optimizer requests. CPU API tests verify that separation. Evaluation still
uses actual pre/post model likelihoods; no metric thresholds or success labels
were weakened. Non-finite objectives and weight updates fail explicitly.
The existing CPU weight snapshots restore the model on success or exception;
temporary hooks and parameter gradient flags are also cleaned up.

## Real GPU evidence

Hardware: Modal A100-40GB, float32 model weights, serialized requests. ES and PS
compare the new target's mean token NLL against the old target's. NS compares
the old target against the new target on neighborhood prompts. These are
pairwise likelihood measures, not guaranteed greedy-generation accuracy.

| GPT-2-XL experiment | Standard MEMIT | Context v3 |
| --- | --- | --- |
| Reported paraphrase: new/old NLL | 3.0784 / 0.8819 (fails) | 0.9833 / 1.5319 (passes) |
| Five Eiffel Tower phrasings, PS | 1/5 | 4/5 |
| Eiffel Tower efficacy, ES | 1 | 1 |
| Eiffel Tower neighborhood, NS | 2/2 | 2/2 |
| Eiffel Tower neutral KL, nats | 4.3365e-6 | 7.3549e-6 |
| Three Big Ben phrasings, PS | 3/3 | 3/3 |
| Big Ben neighborhood, NS | 2/2 | 2/2 |
| Big Ben neutral KL, nats | 1.0779e-5 | 1.4183e-5 |

The known paraphrase was the development case used to choose the profile. Four
additional Eiffel Tower phrasings and the Big Ben case were evaluated only after
freezing the settings. This is a small regression set, not a representative
benchmark. One additional Eiffel Tower wording still fails:
`The city where the Eiffel Tower stands is` (new NLL 5.4763, old NLL 2.6360).
The neutral KL comparison uses only two reference prompts and slightly increases
with the stronger edit; it does not prove absence of wider side effects.

Raw requests, responses, timings, and exact baseline-probe restoration checks:

- [Frozen GPT-2 context, extra phrasings](audit/optimization/frozen-gpt2-context-extra/result.json)
- [Frozen GPT-2 standard, extra phrasings](audit/optimization/frozen-gpt2-standard-extra/result.json)
- [Frozen GPT-2 context, Big Ben](audit/optimization/frozen-gpt2-context-bigben/result.json)
- [Frozen GPT-2 standard, Big Ben](audit/optimization/frozen-gpt2-standard-bigben/result.json)

No speed improvement is claimed: timings include variable queuing, cold starts,
and cache state. The step budget is larger. The improvement is editing quality.
No budget-only ablation was run, so the improvement cannot be attributed to the
consistency term alone; the profile changes the objective, fit, and budget.

GPT-J-6B, using layers 3-8, also completed the frozen context profile on real
hardware: ES 1, PS 5/5, NS 2/2, and neutral KL 4.3793e-5 nats. Its original
baseline probe was restored exactly. See the
[raw GPT-J context result](audit/optimization/frozen-gptj-context-extra/result.json).
Standard GPT-J also passes 5/5 on the same phrasings, with KL 3.6482e-5;
the new profile does not improve its aggregate pass rate on this small set.
See the [paired standard result](audit/optimization/frozen-gptj-standard-extra/result.json).

## Development trials retained

- `context-initial`: vocabulary projection error; HTTP 500, original weights restored.
- `context-v1`: normalized consistency weight 1.0 with averaged keys; PS 0.
- `standard-earlier-window`: upstream MEMIT layers 8-12; PS 0.
- `context-multikey-v2`: individual context fitting, consistency weight 1.0; PS 0.
- `context-budget-v3`: consistency weight 0.01, minimum clamp 1.5 and 40 steps; PS 1 on the reported case.

All are retained under `audit/optimization/`. Earlier unsuccessful attempts
are not overwritten or reported as successful optimizations.

## Reproduction

From `prototype`:

```powershell
python -m unittest test_backend.py test_optimizations.py -v
python test_live.py --label rerun-context --profile context --extra-paraphrases
python test_live.py --label rerun-standard --profile standard --extra-paraphrases
python test_live.py --label rerun-bigben --profile context --case bigben
python test_live.py --label rerun-gptj --profile context --model EleutherAI/gpt-j-6B --layers 3,4,5,6,7,8 --extra-paraphrases
```

Use a new label to preserve previous evidence. Live commands incur GPU usage.
The script records local source hashes; responses also identify the server
profile revision. From `prototype/frontend`, with the dev server running:

```powershell
npm run build
node tests/audit.mjs
node tests/optimization-live.mjs
```

The live browser test exercises context edit and comparison, standard MEMIT's
reported failure, desktop/mobile layout, and baseline restoration.

## Final verification

- Production TypeScript/Vite build passed.
- 38 CPU regression tests passed; output is in `audit/optimization/cpu-tests.log`.
- 15 fixture browser groups passed, including profile payloads and stale-response invalidation.
- Five real browser groups passed, including two independent context comparison schemes.
- Both dashboard-default paraphrases pass with context v3; both fail with standard MEMIT on the checked GPT-2 layer window.
- The live dashboard's broader 12-prompt neutral corpus measured KL 2.2872e-5 nats for context v3.
- Desktop and mobile screenshots were inspected; the context selector fits both layouts.

The initial browser-test attempt incorrectly assumed that standard MEMIT would
fail exactly one of the two dashboard paraphrases. The assertion was corrected
to verify each UI result against its measured NLL comparison. The original
attempt is retained at `audit/optimization/browser-first-attempt/`.

Long greedy completions repeat the edited Eiffel Tower sentence under both
profiles. Neighborhood target preferences pass, but generated neighborhood text
can change. Neither free-form fluency nor universal locality is established by
these candidate-likelihood scores.

Run `node audit/verify-optimization.mjs` from `prototype` to independently check
the stored results and write the final source-hash manifest:
[verification.json](audit/optimization/verification.json).
Live browser evidence is in [browser/results.json](audit/optimization/browser/results.json),
with [desktop](audit/optimization/browser/desktop.png) and
[mobile](audit/optimization/browser/mobile.png) screenshots.
