/**
 * frontend/src/lib/api.ts
 * Extended API client with typed error handling + CSV export helper.
 */

import axios, { AxiosError } from 'axios';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const api = axios.create({ baseURL: API_BASE_URL, timeout: 15000 });

// ── Types ────────────────────────────────────────────────────────────────────

export interface ClusterNode {
  nodes:  Record<string, string>;
  status: Record<string, boolean>;
  self:   string;
}

export interface JobSettings {
  categories:    string;
  provider:      string;
  model_name:    string;
  category_embedding_model: string;
  ocr_engine:    string;
  ocr_mode:      string;
  scan_mode:     string;
  express_mode?: boolean;
  override:      boolean;
  enable_search: boolean;
  enable_web_search?: boolean;
  enable_agentic_search?: boolean;
  enable_vision_board: boolean;
  enable_llm_frame: boolean;
  product_focus_guidance_enabled?: boolean;
  enable_vision?: boolean;
  context_size:  number;
}

export interface JobStatus {
  job_id:     string;
  status:     string;
  stage?:     string;
  stage_detail?: string;
  duration_seconds?: number | null;
  created_at: string;
  updated_at: string;
  progress:   number;
  error?:     string;
  settings?:  JobSettings;
  mode:       string;
  url:        string;
  brand?:     string;
  category?:  string;
  category_id?: string;
}

export interface ArtifactFrame {
  timestamp?: number | null;
  label?: string;
  url: string;
}

export interface ArtifactOCR {
  text?: string;
  lines?: string[];
  url?: string | null;
}

export interface ArtifactVisionMatch {
  label: string;
  score: number;
  category_id?: number | null;
}

export interface SignalVectorPlotPoint {
  x: number;
  y: number;
  label: string;
  category_id?: string | number | null;
  score?: number | null;
  kind?: 'query' | 'selected' | 'neighbor' | 'leader' | 'background';
}

export interface SignalVectorBounds {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
}

export interface SignalVectorPlot {
  space: 'mapper' | 'visual';
  title?: string;
  subtitle?: string;
  backend?: string;
  query_label?: string;
  query_fragments?: string[];
  selected_label?: string;
  selected_category_id?: string | null;
  points: SignalVectorPlotPoint[];
  full_bounds?: SignalVectorBounds;
  focus_bounds?: SignalVectorBounds;
}

export interface ArtifactVisionBoard {
  image_url?: string | null;
  plot_url?: string | null;
  top_matches?: ArtifactVisionMatch[];
  metadata?: Record<string, unknown>;
  vector_plot?: SignalVectorPlot | null;
}

export interface ArtifactCategoryMapper {
  category?: string;
  category_id?: string | null;
  method?: string;
  score?: number | null;
  confidence?: number | null;
  query_fragments?: string[];
  top_matches?: ArtifactVisionMatch[];
  vector_plot?: SignalVectorPlot | null;
}

export interface ProcessingTraceResult {
  brand?: string;
  category?: string;
  confidence?: number | null;
  brand_ambiguity_flag?: boolean;
  brand_ambiguity_reason?: string;
  brand_ambiguity_resolved?: boolean;
  brand_disambiguation_reason?: string;
  brand_evidence_strength?: string;
}

export interface ProcessingTraceAttempt {
  attempt_type: string;
  title: string;
  status: 'accepted' | 'rejected' | 'skipped';
  detail?: string;
  trigger_reason?: string;
  elapsed_ms?: number | null;
  frame_count?: number;
  frame_times?: number[];
  ocr_excerpt?: string;
  ocr_signal?: boolean;
  ocr_mode?: string;
  llm_mode?: string;
  evidence_note?: string;
  result?: ProcessingTraceResult;
}

export interface ProcessingTraceSummary {
  headline?: string;
  attempt_count?: number;
  retry_count?: number;
  accepted_attempt_type?: string;
  trigger_reason?: string;
}

export interface ArtifactProcessingTrace {
  mode?: string;
  provider?: string;
  model?: string;
  ocr_engine?: string;
  ocr_mode?: string;
  scan_mode?: string;
  attempts?: ProcessingTraceAttempt[];
  summary?: ProcessingTraceSummary;
}

export interface PerFrameVision {
  frame_index: number;
  top_category: string;
  top_score: number;
}

export interface JobArtifacts {
  latest_frames: ArtifactFrame[];
  llm_frames?: ArtifactFrame[];
  per_frame_vision: PerFrameVision[];
  ocr_text: ArtifactOCR;
  vision_board: ArtifactVisionBoard;
  category_mapper: ArtifactCategoryMapper;
  processing_trace?: ArtifactProcessingTrace | null;
  extras?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface JobExplanationFinal {
  brand?: string;
  category?: string;
  category_id?: string | null;
  confidence?: number | null;
  mapper_method?: string;
  mapper_score?: number | null;
  brand_ambiguity_flag?: boolean;
  brand_ambiguity_reason?: string;
  brand_ambiguity_resolved?: boolean;
  brand_disambiguation_reason?: string;
  brand_evidence_strength?: string;
}

export interface JobExplanationEvidence {
  ocr_excerpt?: string;
  latest_frames?: ArtifactFrame[];
  event_count?: number;
  recent_events?: string[];
}

export interface JobExplanation {
  job_id: string;
  mode?: string | null;
  status: string;
  stage?: string | null;
  stage_detail?: string | null;
  summary: ProcessingTraceSummary;
  attempts: ProcessingTraceAttempt[];
  final: JobExplanationFinal;
  evidence: JobExplanationEvidence;
}

export interface OllamaModel {
  name: string;
  size?: number;
  modified_at?: string;
}

export interface ResultRow {
  Brand:       string;
  Category:    string;
  'Category ID': string;
  Confidence:  number;
  Reasoning:   string;
  [key: string]: unknown;
}

export interface Metrics {
  jobs_queued:               number;
  jobs_processing:           number;
  jobs_completed:            number;
  jobs_failed:               number;
  jobs_submitted_this_process: number;
  uptime_seconds:            number;
  node:                      string;
}

export interface DurationPercentiles {
  count: number;
  p50: number | null;
  p90: number | null;
  p95: number | null;
  p99: number | null;
}

export interface DurationSeriesPoint {
  bucket: string;
  count: number;
  p50: number | null;
  p90: number | null;
  p95: number | null;
  p99: number | null;
}

export interface DurationSamplePoint {
  completed_at: string;
  duration_seconds: number;
}

export interface AnalyticsPathCount {
  attempt_type: string;
  title: string;
  count: number;
}

export interface AnalyticsPathMetrics {
  jobs_with_trace: number;
  accepted_paths: AnalyticsPathCount[];
  transit_paths: AnalyticsPathCount[];
}

export interface AnalyticsData {
  top_brands: { brand: string; count: number }[];
  categories: { category: string; count: number }[];
  avg_duration_by_mode: { mode: string; avg_duration: number | null; count: number }[];
  avg_duration_by_scan: { scan_mode: string; avg_duration: number | null; count: number }[];
  daily_outcomes: { day: string; status: string; count: number }[];
  providers: { provider: string; count: number }[];
  totals: {
    total: number;
    completed: number;
    failed: number;
    avg_duration: number | null;
  };
  duration_percentiles: DurationPercentiles;
  duration_series: DurationSeriesPoint[];
  recent_duration_points: DurationSamplePoint[];
  path_metrics: AnalyticsPathMetrics;
}

export interface SystemProfileWarning {
  model: string;
  severity: string;
  message: string;
  requirements: {
    min_ram_mb: number;
    min_vram_mb: number;
    accelerator: string;
  };
}

export interface SystemProfile {
  timestamp: string;
  hardware: {
    cpu_count_logical: number;
    cpu_count_physical: number;
    total_ram_mb: number;
    used_ram_mb: number;
    free_ram_mb: number;
    memory_percent: number;
    accelerator: string;
    device_name: string;
    cuda_available: boolean;
    mps_available: boolean;
    total_vram_mb: number | null;
    free_vram_mb: number | null;
  };
  capability_matrix: Array<Record<string, unknown>>;
  warnings: SystemProfileWarning[];
}

export interface BenchmarkTruth {
  test_id: string;
  truth_id: string;
  suite_id?: string;
  name: string;
  source_url?: string;
  video_url: string;
  expected_ocr_text: string;
  expected_categories: string[];
  expected_category?: string;
  expected_brand?: string;
  expected_confidence?: number | null;
  expected_reasoning?: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface BenchmarkSuiteSummary {
  suite_id: string;
  truth_id: string;
  truth_name?: string;
  name?: string;
  description?: string;
  status: string;
  total_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  test_count?: number;
  evaluated_at?: string | null;
  created_at: string;
  updated_at: string;
}

export interface BenchmarkSuiteDetail extends BenchmarkSuiteSummary {
  video_url?: string;
  matrix: Record<string, unknown>;
  tests: BenchmarkTruth[];
}

export interface BenchmarkPoint {
  job_id: string;
  x_duration_seconds: number;
  y_composite_accuracy_pct: number;
  classification_accuracy: number;
  ocr_accuracy: number;
  params: Record<string, unknown>;
  label: string;
}

export interface BenchmarkPathCount {
  attempt_type: string;
  title: string;
  count: number;
}

export interface BenchmarkPathMetrics {
  jobs_with_trace: number;
  accepted_paths: BenchmarkPathCount[];
  transit_paths: BenchmarkPathCount[];
}

export interface BenchmarkSuiteResults {
  suite_id: string;
  status: string;
  total_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  points: BenchmarkPoint[];
  path_metrics?: BenchmarkPathMetrics;
}

export interface ModelCombo {
  provider: string;
  model: string;
  key: string; // `${provider}::${model}`
}

export function emptyAnalytics(): AnalyticsData {
  return {
    top_brands: [],
    categories: [],
    avg_duration_by_mode: [],
    avg_duration_by_scan: [],
    daily_outcomes: [],
    providers: [],
    totals: {
      total: 0,
      completed: 0,
      failed: 0,
      avg_duration: null,
    },
    duration_percentiles: {
      count: 0,
      p50: null,
      p90: null,
      p95: null,
      p99: null,
    },
    duration_series: [],
    recent_duration_points: [],
    path_metrics: {
      jobs_with_trace: 0,
      accepted_paths: [],
      transit_paths: [],
    },
  };
}

function percentile(values: number[], q: number): number | null {
  if (!values.length) return null;
  const ordered = [...values].sort((a, b) => a - b);
  if (ordered.length === 1) return ordered[0];
  const clamped = Math.max(0, Math.min(1, q));
  const pos = (ordered.length - 1) * clamped;
  const lower = Math.floor(pos);
  const upper = Math.ceil(pos);
  if (lower === upper) return ordered[lower];
  const weight = pos - lower;
  return ordered[lower] * (1 - weight) + ordered[upper] * weight;
}

function roundOrNull(value: number | null): number | null {
  return value == null || !Number.isFinite(value) ? null : Math.round(value * 10) / 10;
}

function computeDurationAnalyticsFromPoints(points: DurationSamplePoint[]): {
  durationPercentiles: DurationPercentiles;
  durationSeries: DurationSeriesPoint[];
} {
  const durations = points
    .map((point) => Number(point.duration_seconds))
    .filter((value) => Number.isFinite(value));

  const durationPercentiles: DurationPercentiles = {
    count: durations.length,
    p50: roundOrNull(percentile(durations, 0.5)),
    p90: roundOrNull(percentile(durations, 0.9)),
    p95: roundOrNull(percentile(durations, 0.95)),
    p99: roundOrNull(percentile(durations, 0.99)),
  };

  const bucketMap = new Map<string, number[]>();
  for (const point of points) {
    const completedAt = String(point.completed_at || '').trim();
    const duration = Number(point.duration_seconds);
    if (!completedAt || !Number.isFinite(duration)) continue;
    const bucket = completedAt.length >= 13 ? `${completedAt.slice(0, 13)}:00` : completedAt;
    const current = bucketMap.get(bucket) || [];
    current.push(duration);
    bucketMap.set(bucket, current);
  }

  const durationSeries = Array.from(bucketMap.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .slice(-48)
    .map(([bucket, values]) => ({
      bucket,
      count: values.length,
      p50: roundOrNull(percentile(values, 0.5)),
      p90: roundOrNull(percentile(values, 0.9)),
      p95: roundOrNull(percentile(values, 0.95)),
      p99: roundOrNull(percentile(values, 0.99)),
    }));

  return { durationPercentiles, durationSeries };
}

function mergeAnalytics(responses: AnalyticsData[]): AnalyticsData {
  if (responses.length === 0) return emptyAnalytics();
  if (responses.length === 1) return responses[0];

  const mergeCounts = (
    key: 'brand' | 'category' | 'provider',
    arrays: Array<Array<Record<string, string | number | null>>>,
  ): Array<{ key: string; count: number }> => {
    const map = new Map<string, number>();
    for (const arr of arrays) {
      for (const item of arr) {
        const label = String(item[key] ?? '').trim();
        if (!label) continue;
        const count = Number(item.count ?? 0);
        map.set(label, (map.get(label) || 0) + count);
      }
    }
    return Array.from(map.entries())
      .map(([label, count]) => ({ key: label, count }))
      .sort((a, b) => b.count - a.count);
  };

  const mergeWeightedAvg = (
    key: 'mode' | 'scan_mode',
    arrays: Array<Array<Record<string, string | number | null>>>,
  ): Array<{ key: string; avg_duration: number | null; count: number }> => {
    const map = new Map<string, { totalDuration: number; totalCount: number }>();
    for (const arr of arrays) {
      for (const item of arr) {
        const label = String(item[key] ?? '').trim();
        if (!label) continue;
        const count = Number(item.count ?? 0);
        const avg = Number(item.avg_duration ?? 0);
        if (!Number.isFinite(count) || count <= 0) continue;
        if (!Number.isFinite(avg)) continue;
        const existing = map.get(label) || { totalDuration: 0, totalCount: 0 };
        existing.totalDuration += avg * count;
        existing.totalCount += count;
        map.set(label, existing);
      }
    }

    return Array.from(map.entries())
      .map(([label, value]) => ({
        key: label,
        avg_duration:
          value.totalCount > 0
            ? Math.round((value.totalDuration / value.totalCount) * 10) / 10
            : null,
        count: value.totalCount,
      }))
      .sort((a, b) => b.count - a.count);
  };

  const dailyMap = new Map<string, number>();
  for (const response of responses) {
    for (const row of response.daily_outcomes) {
      const key = `${row.day}|${row.status}`;
      dailyMap.set(key, (dailyMap.get(key) || 0) + row.count);
    }
  }
  const daily_outcomes = Array.from(dailyMap.entries())
    .map(([key, count]) => {
      const [day, status] = key.split('|');
      return { day, status, count };
    })
    .sort((a, b) => a.day.localeCompare(b.day));

  let total = 0;
  let completed = 0;
  let failed = 0;
  let weightedDurationSum = 0;
  let weightedDurationCount = 0;

  for (const response of responses) {
    total += response.totals.total || 0;
    completed += response.totals.completed || 0;
    failed += response.totals.failed || 0;
    if (response.totals.avg_duration != null && response.totals.completed > 0) {
      weightedDurationSum += response.totals.avg_duration * response.totals.completed;
      weightedDurationCount += response.totals.completed;
    }
  }

  const mergedDurationPoints = responses
    .flatMap((response) => response.recent_duration_points || [])
    .filter((point) => point?.completed_at && Number.isFinite(Number(point.duration_seconds)))
    .sort((a, b) => String(a.completed_at).localeCompare(String(b.completed_at)))
    .slice(-1200);

  const { durationPercentiles, durationSeries } = computeDurationAnalyticsFromPoints(mergedDurationPoints);

  const mergePathCounts = (key: 'accepted_paths' | 'transit_paths'): AnalyticsPathCount[] => {
    const counts = new Map<string, { count: number; title: string }>();
    for (const response of responses) {
      for (const row of response.path_metrics?.[key] || []) {
        const attemptType = String(row.attempt_type || '').trim();
        if (!attemptType) continue;
        const current = counts.get(attemptType) || {
          count: 0,
          title: String(row.title || '').trim() || attemptType,
        };
        current.count += Number(row.count || 0);
        if (!current.title) current.title = String(row.title || '').trim() || attemptType;
        counts.set(attemptType, current);
      }
    }
    return Array.from(counts.entries())
      .map(([attempt_type, row]) => ({ attempt_type, title: row.title, count: row.count }))
      .sort((a, b) => b.count - a.count || a.title.localeCompare(b.title));
  };

  const jobsWithTrace = responses.reduce(
    (sum, response) => sum + Number(response.path_metrics?.jobs_with_trace || 0),
    0,
  );

  return {
    top_brands: mergeCounts(
      'brand',
      responses.map((r) => r.top_brands),
    )
      .map((row) => ({ brand: row.key, count: row.count }))
      .slice(0, 20),
    categories: mergeCounts(
      'category',
      responses.map((r) => r.categories),
    )
      .map((row) => ({ category: row.key, count: row.count }))
      .slice(0, 25),
    avg_duration_by_mode: mergeWeightedAvg(
      'mode',
      responses.map((r) => r.avg_duration_by_mode),
    ).map((row) => ({ mode: row.key, avg_duration: row.avg_duration, count: row.count })),
    avg_duration_by_scan: mergeWeightedAvg(
      'scan_mode',
      responses.map((r) => r.avg_duration_by_scan),
    ).map((row) => ({ scan_mode: row.key, avg_duration: row.avg_duration, count: row.count })),
    daily_outcomes,
    providers: mergeCounts(
      'provider',
      responses.map((r) => r.providers),
    ).map((row) => ({ provider: row.key, count: row.count })),
    totals: {
      total,
      completed,
      failed,
      avg_duration:
        weightedDurationCount > 0
          ? Math.round((weightedDurationSum / weightedDurationCount) * 10) / 10
          : null,
    },
    duration_percentiles: durationPercentiles,
    duration_series: durationSeries,
    recent_duration_points: mergedDurationPoints,
    path_metrics: {
      jobs_with_trace: jobsWithTrace,
      accepted_paths: mergePathCounts('accepted_paths'),
      transit_paths: mergePathCounts('transit_paths'),
    },
  };
}

// ── API helpers with typed errors ─────────────────────────────────────────────

function parseError(err: unknown): string {
  if (err instanceof AxiosError) {
    if (!err.response) return `Network error — cannot reach ${API_BASE_URL}`;
    const detail = err.response.data?.detail;
    return detail ? String(detail) : `HTTP ${err.response.status}: ${err.response.statusText}`;
  }
  return String(err);
}

async function safe<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (err) {
    throw new Error(parseError(err));
  }
}

// ── Exported API functions ───────────────────────────────────────────────────

export const getClusterNodes  = () => safe(() => api.get<ClusterNode>('/cluster/nodes').then(r => r.data));
export const getClusterJobs   = () => safe(() => api.get<JobStatus[]>('/cluster/jobs').then(r => r.data));
export const getMetrics       = () => safe(() => api.get<Metrics>('/metrics').then(r => r.data));
export const getAnalytics     = () => safe(() => api.get<AnalyticsData>('/analytics').then(r => r.data));
export const getSystemProfile = () => safe(() => api.get<SystemProfile>('/api/system/profile').then(r => r.data));
export const getJob           = (id: string) => safe(() => api.get<JobStatus>(`/jobs/${id}`).then(r => r.data));
export const getJobResult     = (id: string) => safe(() => api.get<{ result: ResultRow[] | null }>(`/jobs/${id}/result`).then(r => r.data));
export const getJobArtifacts  = (id: string) => safe(() => api.get<{ artifacts: JobArtifacts }>(`/jobs/${id}/artifacts`).then(r => r.data));
export const getJobEvents     = (id: string) => safe(() => api.get<{ events: string[] }>(`/jobs/${id}/events`).then(r => r.data));
export const getJobExplanation = (id: string) => safe(() => api.get<{ explanation: JobExplanation }>(`/jobs/${id}/explanation`).then(r => r.data));
export const getProviderModels = (provider: string) =>
  safe(() => api.get<OllamaModel[]>('/api/v1/models', { params: { provider } }).then((r) => r.data));
export const getOllamaModels  = () => getProviderModels('ollama');
export const getLlamaServerModels = () => getProviderModels('llama-server');
export const deleteJob        = (id: string) => safe(() => api.delete(`/jobs/${id}`).then(r => r.data));
export const deleteJobsBulk   = async (jobIds: string[]) => {
  const results = await Promise.allSettled(jobIds.map((id) => deleteJob(id)));
  const deleted = results.filter((result) => result.status === 'fulfilled').length;
  const failed = results.length - deleted;
  return { status: 'deleted', requested: jobIds.length, deleted, failed };
};
export const submitUrls       = (data: unknown) => safe(() => api.post('/jobs/by-urls', data).then(r => r.data));
export const submitFilePath   = (data: unknown) => safe(() => api.post('/jobs/by-filepath', data).then(r => r.data));
export const submitFolderPath = (data: unknown) => safe(() => api.post('/jobs/by-folder', data).then(r => r.data));
export const createBenchmarkTruth = (data: unknown) =>
  safe(() => api.post<{ truth_id: string }>('/api/benchmark/truths', data).then((r) => r.data));
export const getBenchmarkTruths = () =>
  safe(() => api.get<{ truths: BenchmarkTruth[] }>('/api/benchmark/truths').then((r) => r.data));
export const runBenchmarkSuite = (data: unknown) =>
  safe(() => api.post('/api/benchmark/run', data).then((r) => r.data));
export const getBenchmarkSuites = () =>
  safe(() => api.get<{ suites: BenchmarkSuiteSummary[] }>('/api/benchmark/suites').then((r) => r.data));
export const getBenchmarkSuite = (suiteId: string) =>
  safe(() => api.get<BenchmarkSuiteDetail>(`/api/benchmark/suites/${suiteId}`).then((r) => r.data));
export const getBenchmarkSuiteResults = (suiteId: string) =>
  safe(() => api.get<BenchmarkSuiteResults>(`/api/benchmark/suites/${suiteId}/results`).then((r) => r.data));
export const updateBenchmarkSuite = (suiteId: string, data: { name: string; description: string }) =>
  safe(() => api.put<BenchmarkSuiteSummary>(`/benchmarks/suites/${suiteId}`, data).then((r) => r.data));
export const deleteBenchmarkSuite = (suiteId: string) =>
  safe(() => api.delete(`/benchmarks/suites/${suiteId}`).then((r) => r.data));
export const deleteBenchmarkTruth = (truthId: string) =>
  safe(() => api.delete(`/api/benchmark/truths/${truthId}`).then((r) => r.data));
export const getBenchmarkModels = async (): Promise<ModelCombo[]> => {
  const combos: ModelCombo[] = [];
  try {
    const res = await api.get<{ name: string }[]>('/ollama/models');
    for (const m of res.data || []) {
      if (m.name) combos.push({ provider: 'Ollama', model: m.name, key: `Ollama::${m.name}` });
    }
  } catch { /* no Ollama */ }
  try {
    const res = await api.get<{ name?: string; id?: string }[]>('/llama-server/models');
    for (const m of res.data || []) {
      const name = m.name || m.id || '';
      if (name) combos.push({ provider: 'Llama Server', model: name, key: `Llama Server::${name}` });
    }
  } catch { /* no Llama Server */ }
  return combos;
};
export const updateBenchmarkTest = (
  testId: string,
  data: {
    source_url?: string;
    expected_category?: string;
    expected_brand?: string;
    expected_confidence?: number | null;
    expected_reasoning?: string;
    expected_ocr_text?: string;
    expected_categories?: string[];
  },
) => safe(() => api.put<BenchmarkTruth>(`/benchmarks/tests/${testId}`, data).then((r) => r.data));
export const deleteBenchmarkTest = (testId: string) =>
  safe(() => api.delete(`/benchmarks/tests/${testId}`).then((r) => r.data));
export const getJobVideoUrl   = (jobId: string): string => `${API_BASE_URL}/jobs/${jobId}/video`;
export const getJobVideoPosterUrl = (jobId: string): string =>
  `${API_BASE_URL}/jobs/${jobId}/video-poster`;

export async function getClusterAnalytics(): Promise<AnalyticsData> {
  try {
    return await safe(() => api.get<AnalyticsData>('/cluster/analytics').then((r) => r.data));
  } catch {
    // Backward-compatible fallback for nodes without /cluster/analytics.
  }

  const cluster = await getClusterNodes();
  const nodeUrls = Object.values(cluster.nodes || {});
  if (nodeUrls.length === 0) return emptyAnalytics();

  const responses = await Promise.allSettled(
    nodeUrls.map(async (url) => {
      const target = `${String(url).replace(/\/$/, '')}/analytics`;
      const res = await fetch(target);
      if (!res.ok) throw new Error(`analytics fetch failed: ${res.status}`);
      return (await res.json()) as AnalyticsData;
    }),
  );

  const successful = responses
    .filter((row): row is PromiseFulfilledResult<AnalyticsData> => row.status === 'fulfilled')
    .map((row) => row.value);

  return mergeAnalytics(successful);
}

// ── CSV export ───────────────────────────────────────────────────────────────

/**
 * Convert an array of result rows to a CSV string and trigger a browser download.
 */
export function exportResultsCSV(rows: ResultRow[], filename = 'results.csv'): void {
  if (!rows.length) return;

  const cols: (keyof ResultRow)[] = ['Brand', 'Category', 'Category ID', 'Confidence', 'Reasoning'];
  const header = cols.map(c => `"${String(c)}"`).join(',');
  const body   = rows.map(row =>
    cols.map(c => `"${String(row[c] ?? '').replace(/"/g, '""')}"`).join(',')
  ).join('\n');

  const blob = new Blob([header + '\n' + body], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement('a'), { href: url, download: filename });
  a.click();
  URL.revokeObjectURL(url);
}

// ── Copy helpers ─────────────────────────────────────────────────────────────

export async function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard) {
    await navigator.clipboard.writeText(text);
  } else {
    // Fallback for http contexts
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  }
}
