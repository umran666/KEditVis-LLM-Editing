export interface TopToken {
  token: string;
  prob: number;
}

export interface LayerSignal {
  layer: number;
  cosine_similarity: number;
  /** Null when the quantity is undefined for that layer (layer 0 has no delta). */
  residual_variance?: number | null;
  residual_variance_last?: number | null;
  residual_delta_variance?: number | null;
  top_tokens: TopToken[];
  last_top_tokens?: TopToken[];
}

export interface LayerWeightDrift {
  param_name: string;
  absolute_frobenius: number;
  relative_frobenius: number;
  orig_frobenius: number;
}

export interface WeightDriftReport {
  total_absolute_frobenius: number;
  total_relative_frobenius: number;
  mean_layer_relative_frobenius: number;
  per_layer: Record<number | string, LayerWeightDrift>;
}

export interface Metrics {
  ES: number;
  PS: number | null;
  NS: number | null;
  S: number | null;
  ES_greedy?: number;
  PS_greedy?: number | null;
  NS_greedy?: number | null;
  S_greedy?: number | null;
  details?: {
    efficacy: PromptEvaluation[];
    paraphrase: PromptEvaluation[];
    neighborhood: PromptEvaluation[];
  };
}

export interface PromptEvaluation {
  prefix: string;
  target_new_nll: number;
  target_true_nll: number;
  target_new_correct: boolean;
  target_true_correct: boolean;
}

export interface NeighborhoodResult {
  prompt: string;
  pre_text: string;
  post_text: string;
  kl_divergence: number;
  projection?: { pre: [number, number]; post: [number, number] };
  hidden_state_drift?: number;
  drift_layer?: number;
  projection_method?: string;
}

export interface DamageReport {
  kl_divergence: number;
  n_prompts: number;
  note: string;
}

export interface FactInput {
  prompt: string;
  subject: string;
  target_new: string;
  target_true: string;
  paraphrase_prompts: string[];
  neighborhood_prompts: string[];
}


export interface DamageReport {
  kl_divergence: number;
  n_prompts: number;
  note: string;
}

export interface FactInput {
  prompt: string;
  subject: string;
  target_new: string;
  target_true: string;
  paraphrase_prompts: string[];
  neighborhood_prompts: string[];
}

/** MEMIT optimization profiles accepted by the backend. */
export type OptimizationProfile =
  | "standard"
  | "context"
  | "context_v3"
  | "standard_budget"
  | "context_no_consistency";

export interface ProbeResponse {
  rewrite_prompt: string;
  layer_signals: LayerSignal[];
}

export interface EditResponse {
  optimization?: OptimizationProfile;
  optimization_config?: { revision: string; [key: string]: string | number };
  neighborhood?: NeighborhoodResult[];
  method?: string;
  edited_layers: number[];
  weight_drift?: WeightDriftReport;
  pre_edit: {
    layer_signals: LayerSignal[];
    generations: string[];
    metrics: Metrics | null;
  };
  post_edit: {
    layer_signals: LayerSignal[];
    generations: string[];
    metrics: Metrics | null;
  };
  damage?: DamageReport;
}

export interface SchemeResult {
  layer_signals?: LayerSignal[];
  neighborhood?: NeighborhoodResult[];
  layers: number[];
  metrics: Metrics | null;
  weight_drift?: WeightDriftReport;
  generation: string;
  damage?: DamageReport;
}

export interface CompareResponse {
  optimization?: OptimizationProfile;
  optimization_config?: { revision: string; [key: string]: string | number };
  method?: string;
  baseline: {
    metrics: Metrics | null;
    generation: string;
    layer_signals: LayerSignal[];
    damage?: DamageReport;
  };
  schemes: SchemeResult[];
}

export interface HealthResponse {
  status: string;
  model: string;
  n_layers: number;
  methods: string[];
}
