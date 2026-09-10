import type { SchemeResult } from "./types";

export function schemeKey(layers: number[]): string {
  return [...new Set(layers)].sort((a, b) => a - b).join("-");
}

export function comparisonSchemes(schemes: number[][], method: "memit" | "rome", nLayers: number): number[][] {
  const normalized = schemes.map((layers) => [...new Set(layers)].sort((a, b) => a - b));
  if (normalized.some((layers) => !layers.length || layers.some((l) => !Number.isInteger(l) || l < 0 || l >= nLayers))) {
    throw new Error(`Schemes must contain layers from 0 to ${nLayers - 1}.`);
  }
  const candidates = method === "rome" ? normalized.flatMap((layers) => layers.map((l) => [l])) : normalized;
  return [...new Map(candidates.map((layers) => [schemeKey(layers), layers])).values()];
}

export function sortSchemes(schemes: SchemeResult[]): SchemeResult[] {
  return [...schemes].sort((a, b) => (b.metrics?.S ?? -1) - (a.metrics?.S ?? -1));
}
