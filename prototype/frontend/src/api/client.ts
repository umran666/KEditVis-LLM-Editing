import type { CompareResponse, EditResponse, FactInput, HealthResponse, OptimizationProfile, ProbeResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (typeof data?.detail === "string") message = data.detail;
      else if (Array.isArray(data?.detail)) {
        message = data.detail.map((item: { msg?: string }) => item.msg ?? "Invalid request").join("; ");
      }
    } catch {
      // Preserve the HTTP error when the response body is not JSON.
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export function getApiBase(): string {
  return API_BASE;
}

export function health(model?: string): Promise<HealthResponse> {
  return request(model ? `/health?model=${encodeURIComponent(model)}` : "/health");
}

export function probe(body: Pick<FactInput, "prompt" | "subject"> & { model?: string }): Promise<ProbeResponse> {
  return request("/probe", { method: "POST", body: JSON.stringify(body) });
}

export function generate(body: Pick<FactInput, "prompt" | "subject"> & { model?: string }): Promise<{ model: string; prompt: string; generation: string }> {
  return request("/generate", { method: "POST", body: JSON.stringify(body) });
}

function validateNeighborhoodTargets(body: FactInput): void {
  if (body.neighborhood_targets !== undefined && (
    body.neighborhood_targets.length !== body.neighborhood_prompts.length ||
    body.neighborhood_targets.some((target) => !target.trim())
  )) throw new Error("Provide one non-empty original answer for each neighborhood prompt, or leave neighborhood answers empty to use the fact's original answer.");
}

export function edit(body: FactInput & { layers: number[] | null; method?: string; model?: string; optimization?: OptimizationProfile }): Promise<EditResponse> {
  validateNeighborhoodTargets(body);
  return request("/edit", { method: "POST", body: JSON.stringify(body) });
}

export function compare(body: FactInput & { schemes: number[][]; method?: string; model?: string; optimization?: OptimizationProfile }): Promise<CompareResponse> {
  validateNeighborhoodTargets(body);
  return request("/compare", { method: "POST", body: JSON.stringify(body) });
}
