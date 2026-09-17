import type { SchemeResult } from "./types";

export function schemeKey(layers: number[]): string {
  return [...new Set(layers)].sort((a, b) => a - b).join("-");
}

/**
 * Compacts a layer list into contiguous ranges: [9,10,11,13,15] -> "9-11,13,15".
 *
 * A non-contiguous scheme must not be displayed as a single range: this project's
 * own finding is that skipping a layer inside an otherwise working window breaks
 * the edit, so rendering [9,10,11,13,15] as "9-15" would misrepresent a failing
 * scheme as a contiguous one.
 */
export function formatLayerSpec(layers: number[]): string {
  const sorted = [...new Set(layers)].sort((a, b) => a - b);
  if (!sorted.length) return "—";
  const parts: string[] = [];
  let start = sorted[0];
  let prev = sorted[0];
  for (const layer of sorted.slice(1)) {
    if (layer === prev + 1) {
      prev = layer;
      continue;
    }
    parts.push(start === prev ? `${start}` : `${start}-${prev}`);
    start = prev = layer;
  }
  parts.push(start === prev ? `${start}` : `${start}-${prev}`);
  return parts.join(",");
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
