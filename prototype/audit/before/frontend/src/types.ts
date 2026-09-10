export interface TopToken {
  token: string;
  prob: number;
}

export interface LayerSignal {
  layer: number;
  cosine_similarity: number;
  top_tokens: TopToken[];
  last_top_tokens?: TopToken[];
  fact_top_tokens?: TopToken[];
}

export interface Metrics {
  ES: number;
  PS: number | null;
  NS: number | null;
  S: number;
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

export interface ProbeResponse {
  rewrite_prompt: string;
  layer_signals: LayerSignal[];
}

export interface EditResponse {
  method?: string;
  edited_layers: number[];
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
  layers: number[];
  metrics: Metrics | null;
  generation: string;
  damage?: DamageReport;
}

export interface CompareResponse {
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
