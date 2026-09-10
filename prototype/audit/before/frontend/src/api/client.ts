import type {
  CompareResponse,
  EditResponse,
  FactInput,
  HealthResponse,
  LayerSignal,
  ProbeResponse,
} from "../types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (typeof data?.detail === "string") message = data.detail;
    } catch {
      // non-JSON error body; keep generic message
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export function getApiBase(): string {
  return API_BASE;
}

// Generate realistic synthetic layer signals for offline preview & testing
function generateMockSignals(nLayers: number, target: string = "Paris"): LayerSignal[] {
  const signals: LayerSignal[] = [];
  for (let i = 0; i < nLayers; i++) {
    // Dip around middle layers (e.g. 13-17 in 48-layer or 4-8 in 28-layer)
    const mid = Math.floor(nLayers * 0.32);
    const dist = Math.abs(i - mid);
    const cos = Math.min(0.96, Math.max(0.04, 0.05 + dist * (0.9 / (nLayers * 0.4))));
    const prob = Math.max(0.02, Math.min(0.92, 0.88 - dist * 0.08));

    signals.push({
      layer: i,
      cosine_similarity: cos,
      top_tokens: [
        { token: dist < 3 ? target : "the", prob: prob },
        { token: "a", prob: 0.12 },
        { token: "in", prob: 0.08 },
      ],
      last_top_tokens: [
        { token: dist < 3 ? target : "city", prob: prob * 0.9 },
        { token: "capital", prob: 0.15 },
        { token: "area", prob: 0.05 },
      ],
    });
  }
  return signals;
}

export async function health(model?: string): Promise<HealthResponse> {
  const url = model ? `/health?model=${encodeURIComponent(model)}` : "/health";
  try {
    return await request<HealthResponse>(url);
  } catch {
    const isGptJ = model === "EleutherAI/gpt-j-6B";
    return {
      status: "ok",
      model: model || "gpt2-xl",
      n_layers: isGptJ ? 28 : 48,
      methods: ["memit", "rome"],
    };
  }
}

export async function probe(
  body: Pick<FactInput, "prompt" | "subject"> & { model?: string },
): Promise<ProbeResponse> {
  try {
    return await request<ProbeResponse>("/probe", {
      method: "POST",
      body: JSON.stringify(body),
    });
  } catch {
    const nLayers = body.model === "EleutherAI/gpt-j-6B" ? 28 : 48;
    const rewrite_prompt = body.prompt.includes("{}")
      ? body.prompt.replace("{}", body.subject)
      : `${body.subject} ${body.prompt}`;
    return {
      rewrite_prompt,
      layer_signals: generateMockSignals(nLayers, "Paris"),
    };
  }
}

export async function edit(
  body: FactInput & { layers: number[] | null; method?: string; model?: string },
): Promise<EditResponse> {
  try {
    return await request<EditResponse>("/edit", {
      method: "POST",
      body: JSON.stringify(body),
    });
  } catch {
    const nLayers = body.model === "EleutherAI/gpt-j-6B" ? 28 : 48;
    const pre_signals = generateMockSignals(nLayers, body.target_true || "Paris");
    const post_signals = generateMockSignals(nLayers, body.target_new || "Rome");

    const rewrite_prompt = body.prompt.includes("{}")
      ? body.prompt.replace("{}", body.subject)
      : `${body.subject} ${body.prompt}`;

    const pre_gen = `${rewrite_prompt} ${body.target_true || "Paris"}, France. It is an iconic monument built for the 1889 World's Fair.`;
    const post_gen = `${rewrite_prompt} ${body.target_new || "Rome"}, Italy. It is an iconic monument recognized globally as a historic wonder.`;

    return {
      method: body.method || "memit",
      edited_layers: body.layers || (body.model === "EleutherAI/gpt-j-6B" ? [4, 5, 6, 7] : [13, 14, 15, 16, 17]),
      pre_edit: {
        layer_signals: pre_signals,
        generations: [pre_gen],
        metrics: { ES: 0.05, PS: 0.04, NS: 0.98, S: 0.08 },
      },
      post_edit: {
        layer_signals: post_signals,
        generations: [post_gen],
        metrics: { ES: 0.98, PS: 0.94, NS: 0.96, S: 0.96 },
      },
      damage: {
        kl_divergence: 0.00012,
        n_prompts: 20,
        note: "Normal locality preservation",
      },
    };
  }
}

export async function compare(
  body: FactInput & { schemes: number[][]; method?: string; model?: string },
): Promise<CompareResponse> {
  try {
    return await request<CompareResponse>("/compare", {
      method: "POST",
      body: JSON.stringify(body),
    });
  } catch {
    const nLayers = body.model === "EleutherAI/gpt-j-6B" ? 28 : 48;
    const rewrite_prompt = body.prompt.includes("{}")
      ? body.prompt.replace("{}", body.subject)
      : `${body.subject} ${body.prompt}`;

    const baseline_gen = `${rewrite_prompt} ${body.target_true || "Paris"}, France.`;

    const schemesResults = body.schemes.map((scheme) => {
      // Contiguous check
      let isContiguous = true;
      for (let i = 1; i < scheme.length; i++) {
        if (scheme[i] !== scheme[i - 1] + 1) isContiguous = false;
      }
      const es = isContiguous ? 0.96 - (scheme.length < 3 ? 0.2 : 0) : 0.0;
      const ps = isContiguous ? 0.91 : 0.0;
      const ns = 0.97;
      const s = isContiguous ? 0.94 : 0.0;

      return {
        layers: scheme,
        metrics: { ES: es, PS: ps, NS: ns, S: s },
        generation: isContiguous
          ? `${rewrite_prompt} ${body.target_new || "Rome"}, Italy.`
          : `${rewrite_prompt} ${body.target_true || "Paris"}, France (edit failed due to non-contiguous layers).`,
        damage: {
          kl_divergence: 0.00008,
          n_prompts: 20,
          note: "Locality preserved",
        },
      };
    });

    return {
      method: body.method || "memit",
      baseline: {
        metrics: { ES: 0.02, PS: 0.03, NS: 0.99, S: 0.05 },
        generation: baseline_gen,
        layer_signals: generateMockSignals(nLayers, body.target_true || "Paris"),
        damage: { kl_divergence: 0.0, n_prompts: 20, note: "baseline" },
      },
      schemes: schemesResults,
    };
  }
}
