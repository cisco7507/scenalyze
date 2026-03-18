import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { Link } from 'react-router-dom';
import {
  CheckCircledIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ClockIcon,
  Cross2Icon,
  InfoCircledIcon,
  MagnifyingGlassIcon,
  PlayIcon,
  TrashIcon,
  UpdateIcon,
} from '@radix-ui/react-icons';
import { formatDistanceToNow } from 'date-fns';
import { HelpTooltip } from '../components/HelpTooltip';
import { deleteJobsBulk, getClusterJobs, getProviderModels, submitFilePath, submitFolderPath, submitUrls } from '../lib/api';
import type { JobSettings, JobStatus } from '../lib/api';

type InputMode = 'urls' | 'filepath' | 'dirpath';
type QueueFilter = 'all' | 'processing' | 'queued' | 'completed' | 'failed';

const PROVIDER_OPTIONS = ['Ollama', 'LM Studio', 'Llama Server', 'Gemini CLI'] as const;
const CATEGORY_EMBEDDING_MODEL_OPTIONS = [
  'BAAI/bge-large-en-v1.5',
  'jinaai/jina-embeddings-v3',
  'Alibaba-NLP/gte-large-en-v1.5',
  'sentence-transformers/all-mpnet-base-v2',
  'sentence-transformers/all-MiniLM-L6-v2',
  'google/embeddinggemma-300m',
] as const;

const controlClass =
  'h-10 w-full rounded-[1.15rem] border border-slate-300 bg-white px-3 text-sm text-slate-700 shadow-sm transition-colors focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15';
const monoControlClass = `${controlClass} font-mono`;
const statusTabs: Array<{ value: QueueFilter; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'processing', label: 'Running' },
  { value: 'queued', label: 'Pending' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
];

function formatStageLabel(stage?: string): string {
  const raw = (stage || '').trim();
  if (!raw) return 'unknown';
  return raw.replace(/_/g, ' ');
}

function formatDurationLabel(durationSeconds?: number | null): string {
  if (durationSeconds == null || !Number.isFinite(durationSeconds)) return '—';
  if (durationSeconds < 60) return `${durationSeconds.toFixed(1)}s`;
  const minutes = Math.floor(durationSeconds / 60);
  const seconds = Math.round(durationSeconds % 60);
  return `${minutes}m ${seconds}s`;
}

function formatRelativeTimestamp(value?: string): string {
  if (!value) return '—';
  const normalized = value.includes('T') ? value : value.replace(' ', 'T');
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) return value;
  return formatDistanceToNow(parsed, { addSuffix: true });
}

function sanitizeDecorativeLabel(value?: string | null): string {
  const text = (value || '').trim();
  if (!text) return '—';
  const cleaned = text.replace(/^[^\p{L}\p{N}]+/u, '').trim();
  return cleaned || text;
}

function formatConfidenceLabel(confidence?: number | string | null): string {
  const numeric = Number(confidence);
  if (!Number.isFinite(numeric)) return '—';
  const normalized = numeric > 1 ? numeric : numeric * 100;
  return `${normalized.toFixed(normalized >= 99.5 ? 0 : 1)}%`;
}

function normalizeConfidenceValue(confidence?: number | string | null): number | null {
  const numeric = Number(confidence);
  if (!Number.isFinite(numeric)) return null;
  const normalized = numeric > 1 ? numeric : numeric * 100;
  return Math.max(0, Math.min(100, normalized));
}

function getStatusBadgeClass(status: string): string {
  if (status === 'completed') return 'border-emerald-200 bg-emerald-50 text-emerald-700';
  if (status === 'failed') return 'border-rose-200 bg-rose-50 text-rose-700';
  if (status === 'processing') return 'border-primary-200 bg-primary-50 text-primary-700';
  return 'border-slate-200 bg-slate-50 text-slate-600';
}

function getStatusText(status: string): string {
  if (status === 'processing') return 'Running';
  if (status === 'completed') return 'Completed';
  if (status === 'failed') return 'Failed';
  if (status === 're-queued') return 'Pending';
  return 'Pending';
}

function shortenMiddle(value: string, edge = 16): string {
  if (value.length <= edge * 2 + 3) return value;
  return `${value.slice(0, edge)}...${value.slice(-edge)}`;
}

function getSourceMeta(rawSource?: string | null): { primary: string; secondary: string; tertiary: string; title: string } {
  const raw = (rawSource || '').trim();
  if (!raw) {
    return {
      primary: 'No source provided',
      secondary: 'Unknown input',
      tertiary: '—',
      title: 'No source provided',
    };
  }

  if (/^https?:\/\//i.test(raw)) {
    try {
      const parsed = new URL(raw);
      const hostname = parsed.hostname.replace(/^www\./i, '') || raw;
      const pathBits = parsed.pathname.split('/').filter(Boolean);
      const pathLabel = pathBits.length > 0 ? pathBits[pathBits.length - 1] : 'Remote URL';
      return {
        primary: hostname,
        secondary: pathLabel,
        tertiary: shortenMiddle(raw, 28),
        title: raw,
      };
    } catch {
      return {
        primary: raw,
        secondary: 'Remote URL',
        tertiary: shortenMiddle(raw, 28),
        title: raw,
      };
    }
  }

  const normalized = raw.replace(/\\/g, '/');
  const parts = normalized.split('/').filter(Boolean);
  const lastPart = parts[parts.length - 1] || raw;
  const looksLikeFile = /\.[A-Za-z0-9]{2,6}$/.test(lastPart);
  return {
    primary: lastPart,
    secondary: looksLikeFile ? 'Local file' : 'Server path',
    tertiary: shortenMiddle(raw, 28),
    title: raw,
  };
}

function QueueSummaryCard({
  label,
  value,
  detail,
  accent,
}: {
  label: string;
  value: string | number;
  detail: string;
  accent: 'slate' | 'blue' | 'green' | 'rose';
}) {
  const accentClass = {
    slate: 'text-slate-950',
    blue: 'text-primary-700',
    green: 'text-emerald-600',
    rose: 'text-rose-600',
  } as const;

  return (
    <div className="rounded-[1.7rem] border border-slate-200/90 bg-white px-5 py-4 shadow-[0_16px_30px_rgba(0,55,120,0.06)]">
      <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">{label}</div>
      <div className={`mt-3 text-[2.1rem] font-bold ${accentClass[accent]}`}>{value}</div>
      <div className="mt-2 text-xs text-slate-500">{detail}</div>
    </div>
  );
}

function FieldLabel({
  label,
  help,
}: {
  label: string;
  help?: string;
}) {
  return (
    <label className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-[0.2em] text-primary-700/78">
      <span>{label}</span>
      {help ? <HelpTooltip content={help} /> : null}
    </label>
  );
}

export function Jobs() {
  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [showLaunchPanel, setShowLaunchPanel] = useState(false);
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null);

  const [submitLoading, setSubmitLoading] = useState(false);
  const [inputMode, setInputMode] = useState<InputMode>('urls');
  const [urls, setUrls] = useState('https://www.youtube.com/watch?v=M7FIvfx5J10');
  const [filePath, setFilePath] = useState('');
  const [folderPath, setFolderPath] = useState('');
  const [mode, setMode] = useState('pipeline');
  const [categories, setCategories] = useState('');
  const [provider, setProvider] = useState('Llama Server');
  const [modelName, setModelName] = useState('qwen3-vl:8b-instruct');
  const [categoryEmbeddingModel, setCategoryEmbeddingModel] = useState('BAAI/bge-large-en-v1.5');
  const [providerModels, setProviderModels] = useState<string[]>([]);
  const [providerModelsLoading, setProviderModelsLoading] = useState(false);
  const [ocrEngine, setOcrEngine] = useState('EasyOCR');
  const [ocrMode, setOcrMode] = useState('Fast');
  const [scanMode, setScanMode] = useState('Tail Only');
  const [expressMode, setExpressMode] = useState(false);
  const [enableVisionBoard, setEnableVisionBoard] = useState(true);
  const [enableLlmFrame, setEnableLlmFrame] = useState(true);
  const [enableWebSearch, setEnableWebSearch] = useState(true);
  const [productFocusGuidanceEnabled, setProductFocusGuidanceEnabled] = useState(true);
  const [contextSize, setContextSize] = useState(8192);

  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<QueueFilter>('all');
  const [selectedJobs, setSelectedJobs] = useState<Set<string>>(new Set());
  const [deleteLoading, setDeleteLoading] = useState(false);

  const fetchJobs = async () => {
    try {
      const data = await getClusterJobs();
      setJobs(data);
      setLastUpdated(new Date());
    } catch {
      // no-op
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 4000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const providerName = provider.trim().toLowerCase();
    if (providerName !== 'ollama' && providerName !== 'llama-server' && providerName !== 'llama server') {
      setProviderModels([]);
      setProviderModelsLoading(false);
      return;
    }

    const modelProvider = providerName === 'ollama' ? 'ollama' : 'llama-server';
    let active = true;
    setProviderModelsLoading(true);
    getProviderModels(modelProvider)
      .then((models) => {
        if (!active) return;
        const names = models.map((m) => m.name).filter(Boolean);
        setProviderModels(names);
        if (names.length > 0) {
          setModelName((current) => (names.includes(current) ? current : names[0]));
        }
      })
      .catch(() => {
        if (!active) return;
        setProviderModels([]);
      })
      .finally(() => {
        if (!active) return;
        setProviderModelsLoading(false);
      });
    return () => {
      active = false;
    };
  }, [provider]);

  const providerName = provider.trim().toLowerCase();
  const showProviderModelPicker =
    (providerName === 'ollama' || providerName === 'llama-server' || providerName === 'llama server') &&
    providerModels.length > 0;
  const modelInProviderList = showProviderModelPicker && providerModels.includes(modelName);

  const settingsPayload: JobSettings = useMemo(
    () => ({
      categories,
      provider,
      model_name: modelName,
      category_embedding_model: categoryEmbeddingModel,
      ocr_engine: ocrEngine,
      ocr_mode: ocrMode,
      scan_mode: scanMode,
      express_mode: expressMode,
      override: false,
      enable_search: enableWebSearch,
      enable_web_search: enableWebSearch,
      enable_vision_board: enableVisionBoard,
      enable_llm_frame: enableLlmFrame,
      product_focus_guidance_enabled: productFocusGuidanceEnabled,
      context_size: contextSize,
    }),
    [
      categories,
      provider,
      modelName,
      categoryEmbeddingModel,
      ocrEngine,
      ocrMode,
      scanMode,
      expressMode,
      enableWebSearch,
      enableVisionBoard,
      enableLlmFrame,
      productFocusGuidanceEnabled,
      contextSize,
    ],
  );

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setSubmitLoading(true);

    try {
      if (inputMode === 'urls') {
        const urlList = urls
          .split('\n')
          .map((u) => u.trim())
          .filter(Boolean);
        if (!urlList.length) return;
        await submitUrls({ urls: urlList, mode, settings: settingsPayload });
      } else if (inputMode === 'filepath') {
        if (!filePath.trim()) return;
        await submitFilePath({ file_path: filePath, mode, settings: settingsPayload });
      } else {
        if (!folderPath.trim()) return;
        await submitFolderPath({ folder_path: folderPath, mode, settings: settingsPayload });
      }
      await fetchJobs();
      setShowLaunchPanel(false);
    } catch (err) {
      console.error(err);
    } finally {
      setSubmitLoading(false);
    }
  };

  const filteredJobs = jobs.filter((job) => {
    if (statusFilter !== 'all') {
      if (statusFilter === 'queued') {
        if (job.status !== 'queued' && job.status !== 're-queued') return false;
      } else if (job.status !== statusFilter) {
        return false;
      }
    }

    if (!search.trim()) return true;
    const query = search.toLowerCase();
    return (
      job.job_id.toLowerCase().includes(query) ||
      (job.brand || '').toLowerCase().includes(query) ||
      (job.category_name || job.category || '').toLowerCase().includes(query) ||
      (job.parent_category || '').toLowerCase().includes(query) ||
      (job.url || '').toLowerCase().includes(query)
    );
  });

  useEffect(() => {
    if (filteredJobs.length === 0) {
      setExpandedJobId(null);
      return;
    }
    setExpandedJobId((current) => {
      if (current && filteredJobs.some((job) => job.job_id === current)) return current;
      return (
        filteredJobs.find((job) => job.status === 'processing')?.job_id ||
        filteredJobs.find((job) => job.status === 'failed')?.job_id ||
        filteredJobs[0].job_id
      );
    });
  }, [filteredJobs]);

  const summary = useMemo(() => {
    const counts = {
      total: jobs.length,
      visible: filteredJobs.length,
      queued: 0,
      processing: 0,
      completed: 0,
      failed: 0,
    };

    for (const job of jobs) {
      if (job.status === 'processing') counts.processing += 1;
      else if (job.status === 'completed') counts.completed += 1;
      else if (job.status === 'failed') counts.failed += 1;
      else counts.queued += 1;
    }

    return counts;
  }, [jobs, filteredJobs.length]);

  const finishedCount = summary.completed + summary.failed;
  const completionRate = finishedCount > 0 ? Math.round((summary.completed / finishedCount) * 100) : 0;

  const disableSubmit =
    submitLoading ||
    (inputMode === 'urls' && !urls.trim()) ||
    (inputMode === 'filepath' && !filePath.trim()) ||
    (inputMode === 'dirpath' && !folderPath.trim());

  const toggleSelectJob = (jobId: string) => {
    setSelectedJobs((prev) => {
      const next = new Set(prev);
      if (next.has(jobId)) next.delete(jobId);
      else next.add(jobId);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (filteredJobs.length === 0) return;
    if (selectedJobs.size === filteredJobs.length) {
      setSelectedJobs(new Set());
      return;
    }
    setSelectedJobs(new Set(filteredJobs.map((job) => job.job_id)));
  };

  const isAllSelected = filteredJobs.length > 0 && selectedJobs.size === filteredJobs.length;
  const hasSelection = selectedJobs.size > 0;

  const handleBulkDelete = async () => {
    if (!hasSelection) return;

    const count = selectedJobs.size;
    const confirmed = window.confirm(`Delete ${count} job${count > 1 ? 's' : ''}? This cannot be undone.`);
    if (!confirmed) return;

    setDeleteLoading(true);
    try {
      const result = await deleteJobsBulk([...selectedJobs]);
      if (result.failed > 0) {
        console.error(`Bulk delete completed with failures. deleted=${result.deleted} failed=${result.failed}`);
      }
      setSelectedJobs(new Set());
      await fetchJobs();
    } catch (err) {
      console.error('Bulk delete failed:', err);
    } finally {
      setDeleteLoading(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <section className="bell-hero">
        <div className="relative z-10 flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
          <div className="max-w-3xl">
            <div className="bell-badge">
              <CheckCircledIcon className="h-3.5 w-3.5" />
              Queue control surface
            </div>
            <h2 className="mt-4 max-w-3xl text-[3rem] font-bold text-primary-700">Review the queue first, then launch the next run.</h2>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
              This page is tuned for fast queue scanning. Brand, category, status, and stage lead the row. Technical run metadata stays available, but it now lives behind a cleaner run-profile disclosure.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => setShowLaunchPanel((current) => !current)}
                className="bell-button-primary h-12 gap-2 px-5 text-sm uppercase tracking-[0.2em]"
              >
                <PlayIcon className="h-4 w-4" />
                {showLaunchPanel ? 'Hide analysis form' : 'New analysis run'}
              </button>
              <div className="bell-data-pill">
                <UpdateIcon className="h-3.5 w-3.5 text-primary-500" />
                Auto-syncing every 4s
              </div>
              <div className="bell-data-pill">
                <ClockIcon className="h-3.5 w-3.5 text-primary-500" />
                Refreshed {formatDistanceToNow(lastUpdated, { addSuffix: true })}
              </div>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 xl:w-[24rem]">
            <div className="rounded-[1.7rem] border border-white/90 bg-white/88 px-4 py-4 shadow-[0_14px_28px_rgba(0,55,120,0.08)]">
              <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Running now</div>
              <div className="mt-2 text-4xl font-bold text-slate-950">{summary.processing}</div>
              <div className="mt-2 text-sm text-slate-500">Jobs currently moving through OCR, vision, or LLM stages.</div>
            </div>
            <div className="rounded-[1.7rem] border border-primary-700/30 bg-primary-700 px-4 py-4 text-white shadow-[0_18px_36px_rgba(0,55,120,0.22)]">
              <div className="text-[11px] uppercase tracking-[0.22em] text-white/65">Needs review</div>
              <div className="mt-2 text-4xl font-bold text-white">{summary.failed}</div>
              <div className="mt-2 text-sm text-white/72">
                {summary.failed > 0 ? 'Completed runs with failed status need operator attention.' : 'No failed runs at the moment.'}
              </div>
            </div>
          </div>
        </div>
      </section>

      {showLaunchPanel ? (
        <section className="bell-panel overflow-hidden">
          <div className="border-b border-slate-200/80 bg-primary-50/75 px-6 py-4">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <div className="bell-badge">
                  <PlayIcon className="h-3.5 w-3.5" />
                  Launch pipeline
                </div>
                <h3 className="mt-3 text-xl font-bold text-slate-950">Start a new analysis run.</h3>
                <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-500">
                  Pick the evidence path, runtime, and OCR profile. The defaults stay fast for normal work; the advanced controls stay here when you need them.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowLaunchPanel(false)}
                className="bell-button-secondary h-11 gap-2 px-4 text-xs uppercase tracking-[0.2em]"
              >
                <Cross2Icon className="h-4 w-4" />
                Close panel
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5 px-6 py-6">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div className="inline-flex w-fit rounded-full border border-primary-100 bg-white p-1.5 shadow-[0_12px_24px_rgba(0,55,120,0.08)]">
                <button type="button" onClick={() => setInputMode('urls')} className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] transition-colors ${inputMode === 'urls' ? 'bg-primary-500 text-white shadow-[0_10px_20px_rgba(0,112,206,0.20)]' : 'text-primary-700 hover:bg-primary-50'}`}>URLs</button>
                <button type="button" onClick={() => setInputMode('filepath')} className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] transition-colors ${inputMode === 'filepath' ? 'bg-primary-500 text-white shadow-[0_10px_20px_rgba(0,112,206,0.20)]' : 'text-primary-700 hover:bg-primary-50'}`}>File Path</button>
                <button type="button" onClick={() => setInputMode('dirpath')} className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] transition-colors ${inputMode === 'dirpath' ? 'bg-primary-500 text-white shadow-[0_10px_20px_rgba(0,112,206,0.20)]' : 'text-primary-700 hover:bg-primary-50'}`}>Directory Path</button>
              </div>

              <div className="flex items-center justify-between gap-3 lg:min-w-[18rem] lg:justify-end">
                <div className="hidden lg:block text-right">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">Ready to run</div>
                  <div className="mt-1 text-sm text-slate-500">
                    {inputMode === 'urls'
                      ? 'Submit the current URL batch.'
                      : inputMode === 'filepath'
                        ? 'Analyze one server-side file path.'
                        : 'Scan one server-side directory.'}
                  </div>
                </div>
                <button
                  type="submit"
                  disabled={disableSubmit}
                  className="bell-button-primary h-12 min-w-[13rem] gap-2 px-5 text-sm uppercase tracking-[0.22em] disabled:opacity-50"
                >
                  {submitLoading ? <UpdateIcon className="h-4 w-4 animate-spin" /> : <PlayIcon className="h-4 w-4" />}
                  {submitLoading ? 'Submitting…' : 'Execute'}
                </button>
              </div>
            </div>

            <div className="rounded-[1.9rem] border border-slate-200/90 bg-white/74 p-3 shadow-[inset_0_0_0_1px_rgba(212,221,230,0.55)]">
              {inputMode === 'urls' && (
                <textarea
                  value={urls}
                  onChange={(e) => setUrls(e.target.value)}
                  placeholder="Enter URLs (one per line)..."
                  className="h-36 w-full resize-none rounded-[1.5rem] border border-slate-200 bg-white/96 p-4 font-mono text-sm text-slate-700 shadow-sm transition-colors focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15"
                />
              )}
              {inputMode === 'filepath' && (
                <input
                  value={filePath}
                  onChange={(e) => setFilePath(e.target.value)}
                  placeholder={'C:\\videos\\ad.mp4 or \\\\server\\share\\ads\\spot.mp4'}
                  className={`${monoControlClass} h-12 rounded-[20px]`}
                />
              )}
              {inputMode === 'dirpath' && (
                <input
                  value={folderPath}
                  onChange={(e) => setFolderPath(e.target.value)}
                  placeholder={'C:\\videos\\ads or \\\\server\\share\\ads or /mnt/media/ads'}
                  className={`${monoControlClass} h-12 rounded-[20px]`}
                />
              )}
            </div>

            {inputMode !== 'urls' && (
              <div className="flex items-start gap-3 rounded-[1.45rem] border border-primary-100 bg-primary-50/80 px-4 py-3 text-sm text-slate-600">
                <InfoCircledIcon className="mt-0.5 h-4 w-4 shrink-0 text-primary-500" />
                <div className="space-y-1">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-primary-700">Server path note</div>
                  <div>
                    File and directory paths are resolved on the backend server, not in your browser. UNC paths only work if the server can reach that share and has permission to read it.
                  </div>
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3 rounded-[2rem] border border-slate-200/90 bg-[linear-gradient(180deg,rgba(255,255,255,0.88)_0%,rgba(244,247,250,0.96)_100%)] p-5 md:grid-cols-4">
              <div className="space-y-1.5">
                <FieldLabel
                  label="Mode"
                  help="Standard Pipeline uses the deterministic OCR + vision + LLM flow. ReACT Agent lets the model reason in steps and call tools during analysis."
                />
                <select value={mode} onChange={(e) => setMode(e.target.value)} className={controlClass}>
                  <option value="pipeline">Standard Pipeline</option>
                  <option value="agent">ReACT Agent</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <FieldLabel
                  label="Web Search"
                  help="Lets the classifier query the web when local evidence is weak. Improves recovery on ambiguous brands but adds latency and more external dependency."
                />
                <select value={enableWebSearch ? 'true' : 'false'} onChange={(e) => setEnableWebSearch(e.target.value === 'true')} className={controlClass}>
                  <option value="true">Enabled</option>
                  <option value="false">Disabled</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <FieldLabel
                  label="Scan Strategy"
                  help="Tail Only samples the end of the ad where brand cards usually appear. Full Video scans across the whole video for more coverage at higher cost."
                />
                <select value={scanMode} onChange={(e) => setScanMode(e.target.value)} className={controlClass}>
                  <option value="Tail Only">Tail Only</option>
                  <option value="Full Video">Full Video</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <FieldLabel
                  label="Express Mode"
                  help="Skips the normal OCR-heavy path and classifies from a key visual frame. Fastest option, but it trades away some text evidence."
                />
                <select
                  value={expressMode ? 'true' : 'false'}
                  onChange={(e) => setExpressMode(e.target.value === 'true')}
                  className={controlClass}
                >
                  <option value="false">Disabled</option>
                  <option value="true">Enabled (Vision only)</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <FieldLabel
                  label="Vision Board"
                  help="Computes category similarity scores from the selected frames. Useful for debugging why the visual encoder leaned toward certain categories."
                />
                <select value={enableVisionBoard ? 'true' : 'false'} onChange={(e) => setEnableVisionBoard(e.target.value === 'true')} className={controlClass}>
                  <option value="true">Generate Vision Board (SigLIP/OpenCLIP)</option>
                  <option value="false">Disabled</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <FieldLabel
                  label="LLM Keyframe"
                  help="Sends a representative video frame to the multimodal model. Disable this only if you want the LLM to rely on OCR text alone."
                />
                <select value={enableLlmFrame ? 'true' : 'false'} onChange={(e) => setEnableLlmFrame(e.target.value === 'true')} className={controlClass}>
                  <option value="true">Send Keyframe to LLM</option>
                  <option value="false">Disabled</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <FieldLabel
                  label="Product Focus Guidance"
                  help="Adds prompt guidance that prefers the promoted product family over the advertiser's broader industry when both appear. Disable this to evaluate the model without that bias."
                />
                <select
                  value={productFocusGuidanceEnabled ? 'true' : 'false'}
                  onChange={(e) => setProductFocusGuidanceEnabled(e.target.value === 'true')}
                  className={controlClass}
                >
                  <option value="true">Enabled</option>
                  <option value="false">Disabled</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <FieldLabel
                  label="Category Embedding Model"
                  help="Allowlisted sentence-transformer model used by the taxonomy mapper. This affects semantic category matching, neighbor lookup, and mapper-space debug plots."
                />
                <select
                  value={categoryEmbeddingModel}
                  onChange={(e) => setCategoryEmbeddingModel(e.target.value)}
                  className={controlClass}
                >
                  {CATEGORY_EMBEDDING_MODEL_OPTIONS.map((option) => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>
              <div className="space-y-1.5">
                <FieldLabel
                  label="OCR Engine"
                  help="EasyOCR is lighter and faster for most runs. Florence-2 is heavier but can recover harder text cases when speed matters less than recall."
                />
                <select value={ocrEngine} onChange={(e) => setOcrEngine(e.target.value)} className={controlClass}>
                  <option value="EasyOCR">EasyOCR</option>
                  <option value="Florence-2 (Microsoft)">Florence-2</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <FieldLabel
                  label="OCR Mode"
                  help="Fast favors latency and aggressive shortcuts. Detailed keeps more OCR work enabled for harder frames and better text recovery."
                />
                <select value={ocrMode} onChange={(e) => setOcrMode(e.target.value)} className={controlClass}>
                  <option value="Fast">Fast</option>
                  <option value="Detailed">Detailed</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <FieldLabel
                  label="Context Limit"
                  help="Maximum prompt context passed to the selected model. Higher values can preserve more evidence but use more memory and may slow inference."
                />
                <input type="number" min={512} step={512} value={contextSize} onChange={(e) => setContextSize(Number(e.target.value || 8192))} className={monoControlClass} />
              </div>
              <div className="space-y-1.5">
                <FieldLabel
                  label="Provider"
                  help="Runtime that serves the LLM. Different providers support different models, multimodal behavior, JSON modes, and latency profiles."
                />
                <select value={provider} onChange={(e) => setProvider(e.target.value)} className={controlClass}>
                  {PROVIDER_OPTIONS.map((providerOption) => (
                    <option key={providerOption} value={providerOption}>{providerOption}</option>
                  ))}
                </select>
              </div>
              <div className="space-y-1.5 md:col-span-2">
                <FieldLabel
                  label="Model"
                  help="Specific model loaded under the chosen provider. This controls reasoning style, multimodal support, speed, and memory usage."
                />
                {showProviderModelPicker ? (
                  <div className="space-y-2">
                    <select
                      value={modelInProviderList ? modelName : '__custom__'}
                      onChange={(e) => {
                        if (e.target.value === '__custom__') {
                          if (modelInProviderList) setModelName('');
                          return;
                        }
                        setModelName(e.target.value);
                      }}
                      className={controlClass}
                    >
                      {providerModels.map((name) => (
                        <option key={name} value={name}>{name}</option>
                      ))}
                      <option value="__custom__">Custom model...</option>
                    </select>
                    {!modelInProviderList && (
                      <input
                        value={modelName}
                        onChange={(e) => setModelName(e.target.value)}
                        placeholder="Type custom model name..."
                        className={controlClass}
                      />
                    )}
                  </div>
                ) : (
                  <input value={modelName} onChange={(e) => setModelName(e.target.value)} className={controlClass} />
                )}
                {(providerName === 'ollama' || providerName === 'llama-server' || providerName === 'llama server') && providerModelsLoading && (
                  <div className="text-[10px] text-gray-400">
                    Loading available {providerName === 'ollama' ? 'Ollama' : 'Llama Server'} models...
                  </div>
                )}
              </div>
              <div className="space-y-1.5 md:col-span-4">
                <FieldLabel
                  label="Target Categories (Comma Separated)"
                  help="Optional hint list passed into the pipeline. Leave blank to let the system classify against the full taxonomy."
                />
                <input value={categories} onChange={(e) => setCategories(e.target.value)} className={monoControlClass} />
              </div>
            </div>
          </form>
        </section>
      ) : null}

      <section className="bell-panel overflow-hidden">
        <div className="border-b border-slate-200/80 bg-primary-50/75 px-6 py-5">
          <div className="flex flex-col gap-5">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-3">
                  <h3 className="text-xl font-bold text-slate-950">Job queue</h3>
                  <div className="bell-data-pill">
                    <ClockIcon className="h-3.5 w-3.5 text-primary-500" />
                    Auto-syncing
                  </div>
                </div>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
                  Scan outcomes first. Open technical run details only when you need to investigate why a job landed where it did.
                </p>
              </div>

              <div className="flex w-full flex-col gap-3 lg:flex-row xl:w-auto">
                <div className="relative min-w-0 lg:w-[24rem] xl:w-[28rem]">
                  <MagnifyingGlassIcon className="absolute left-3 top-3.5 h-4 w-4 text-slate-400" />
                  <input
                    value={search}
                    onChange={(e) => {
                      setSearch(e.target.value);
                      setSelectedJobs(new Set());
                    }}
                    placeholder="Search jobs, brands, or categories..."
                    className="h-11 w-full rounded-full border border-slate-300 bg-white pl-10 pr-3 text-sm text-slate-700 shadow-sm focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15"
                  />
                </div>
                <button
                  type="button"
                  onClick={() => setShowLaunchPanel((current) => !current)}
                  className="bell-button-primary h-11 shrink-0 gap-2 px-4 text-xs uppercase tracking-[0.18em]"
                >
                  <PlayIcon className="h-3.5 w-3.5" />
                  New analysis run
                </button>
                <button
                  type="button"
                  onClick={fetchJobs}
                  className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-slate-300 bg-white text-primary-700 shadow-sm transition-colors hover:border-primary-300 hover:bg-primary-50"
                  title="Refresh queue"
                >
                  <UpdateIcon className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <QueueSummaryCard label="Total jobs" value={summary.total} detail={`${summary.visible} in current view`} accent="slate" />
              <QueueSummaryCard label="Running" value={summary.processing} detail={summary.processing > 0 ? 'Active now' : 'No live runs'} accent="blue" />
              <QueueSummaryCard label="Completed" value={summary.completed} detail={finishedCount > 0 ? `${completionRate}% success rate` : 'No finished jobs yet'} accent="green" />
              <QueueSummaryCard label="Failed" value={summary.failed} detail={summary.failed > 0 ? 'Action required' : 'No action required'} accent="rose" />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="inline-flex flex-wrap items-center rounded-full border border-slate-200 bg-white p-1.5 shadow-[0_10px_24px_rgba(0,55,120,0.06)]">
                {statusTabs.map((tab) => {
                  const active = statusFilter === tab.value;
                  return (
                    <button
                      key={tab.value}
                      type="button"
                      onClick={() => {
                        setStatusFilter(tab.value);
                        setSelectedJobs(new Set());
                      }}
                      className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] transition-colors ${active ? 'bg-primary-500 text-white shadow-[0_10px_20px_rgba(0,112,206,0.18)]' : 'text-slate-600 hover:bg-primary-50 hover:text-primary-700'}`}
                    >
                      {tab.label}
                    </button>
                  );
                })}
              </div>
              <div className="text-[11px] font-medium uppercase tracking-[0.18em] text-slate-400">
                Last refreshed {formatDistanceToNow(lastUpdated, { addSuffix: true })}
              </div>
            </div>
          </div>
        </div>

        <div className="px-6 py-4">
          <div className="hidden items-center gap-4 border-b border-slate-200/80 px-4 py-3 lg:grid lg:grid-cols-[36px_minmax(0,1.45fr)_minmax(0,1.05fr)_minmax(0,1.05fr)_100px_110px_110px_40px]">
            <div className="flex items-center justify-center">
              <input
                type="checkbox"
                checked={isAllSelected}
                onChange={toggleSelectAll}
                className="h-4 w-4 rounded border-slate-300 bg-white text-primary-500 focus:ring-primary-500/20"
                title={isAllSelected ? 'Deselect all visible jobs' : 'Select all visible jobs'}
              />
            </div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">Brand</div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">Final category</div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">Status & stage</div>
            <div className="text-right text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">Confidence</div>
            <div className="text-right text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">Runtime</div>
            <div className="text-right text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">Updated</div>
            <div />
          </div>

          <div className="divide-y divide-slate-100">
            {loading && jobs.length === 0 ? (
              <div className="px-4 py-12 text-center text-slate-400">Syncing node cluster state...</div>
            ) : filteredJobs.length === 0 ? (
              <div className="px-4 py-12 text-center text-slate-400">No jobs found.</div>
            ) : (
              filteredJobs.map((job) => {
                const categoryLabel = job.category_name || job.category || '—';
                const categoryId = (job.category_id || '').trim();
                const parentCategory = (job.parent_category || '').trim();
                const parentCategoryId = (job.parent_category_id || '').trim();
                const sourceMeta = getSourceMeta(job.url);
                const selected = selectedJobs.has(job.job_id);
                const expanded = expandedJobId === job.job_id;
                const confidenceValue = normalizeConfidenceValue(job.confidence);
                const statusText = getStatusText(job.status);
                const providerLabel = job.settings?.provider || '—';
                const modelLabel = job.settings?.model_name || '—';
                const ocrLabel = `${job.settings?.ocr_engine || '—'} · ${sanitizeDecorativeLabel(job.settings?.ocr_mode)}`;
                const scanLabel = sanitizeDecorativeLabel(job.settings?.scan_mode);
                const stageLabel = formatStageLabel(job.stage);
                const stageDetail = job.error || job.stage_detail || 'No additional stage detail recorded.';
                const showDistinctStage =
                  stageLabel &&
                  stageLabel.toLowerCase() !== 'unknown' &&
                  stageLabel.toLowerCase() !== statusText.toLowerCase();

                return (
                  <div key={job.job_id} className={`py-2 transition-colors ${selected ? 'bg-primary-50/55' : ''}`}>
                    <div className="rounded-[1.7rem] px-4 py-4 transition-colors hover:bg-slate-50/80">
                      <div className="grid gap-4 lg:grid-cols-[36px_minmax(0,1.45fr)_minmax(0,1.05fr)_minmax(0,1.05fr)_100px_110px_110px_40px] lg:items-center">
                        <div className="flex items-start justify-center pt-1 lg:pt-0">
                          <input
                            type="checkbox"
                            checked={selected}
                            onChange={() => toggleSelectJob(job.job_id)}
                            className="h-4 w-4 rounded border-slate-300 bg-white text-primary-500 focus:ring-primary-500/20"
                          />
                        </div>

                        <div className="min-w-0">
                          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400 lg:hidden">Brand</div>
                          <Link to={`/jobs/${job.job_id}`} className="block text-lg font-bold text-slate-950 transition-colors hover:text-primary-700">
                            {job.brand || 'Unknown brand'}
                          </Link>
                          <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-slate-500" title={sourceMeta.title}>
                            <span className="font-medium text-slate-700">{sourceMeta.primary}</span>
                            <span className="text-slate-300">·</span>
                            <span>{sourceMeta.secondary}</span>
                          </div>
                          <div className="mt-2 font-mono text-[11px] text-slate-400" title={job.job_id}>
                            {shortenMiddle(job.job_id, 16)}
                          </div>
                        </div>

                        <div className="min-w-0">
                          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400 lg:hidden">Final category</div>
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="text-sm font-semibold text-slate-900">{categoryLabel}</span>
                            {categoryId ? (
                              <span className="inline-flex items-center rounded-full border border-primary-200 bg-primary-50 px-2 py-0.5 text-[10px] font-mono font-semibold text-primary-700">
                                ID {categoryId}
                              </span>
                            ) : null}
                          </div>
                          <div className="mt-2 text-xs text-slate-500">
                            {parentCategory && parentCategory.toLowerCase() !== categoryLabel.toLowerCase()
                              ? `Parent: ${parentCategory}${parentCategoryId ? ` · ID ${parentCategoryId}` : ''}`
                              : 'Top-level taxonomy result'}
                          </div>
                        </div>

                        <div className="min-w-0">
                          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400 lg:hidden">Status & stage</div>
                          <div className="flex flex-wrap items-center gap-2">
                            <span className={`inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-bold uppercase tracking-[0.18em] ${getStatusBadgeClass(job.status)}`}>
                              {statusText}
                            </span>
                            {showDistinctStage ? <span className="text-sm font-medium text-slate-800">{stageLabel}</span> : null}
                          </div>
                          <div className="mt-2 truncate text-xs text-slate-500" title={stageDetail}>
                            {stageDetail}
                          </div>
                        </div>

                        <div className="text-left lg:text-right">
                          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400 lg:hidden">Confidence</div>
                          <div className="text-sm font-semibold text-slate-900">{formatConfidenceLabel(job.confidence)}</div>
                          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100 lg:ml-auto lg:w-14">
                            <div
                              className="h-full rounded-full bg-primary-500 transition-all"
                              style={{ width: `${confidenceValue ?? 0}%` }}
                            />
                          </div>
                        </div>

                        <div className="text-left lg:text-right">
                          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400 lg:hidden">Runtime</div>
                          <div className="text-sm font-semibold text-slate-900">{formatDurationLabel(job.duration_seconds)}</div>
                          <div className="mt-1 text-xs text-slate-500">
                            {job.status === 'processing' ? 'in progress' : 'wall time'}
                          </div>
                        </div>

                        <div className="text-left lg:text-right">
                          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400 lg:hidden">Updated</div>
                          <div className="text-sm font-semibold text-slate-900">{formatRelativeTimestamp(job.updated_at)}</div>
                          <div className="mt-1 text-xs text-slate-500">
                            {job.status === 'processing' ? 'live status' : 'last persisted'}
                          </div>
                        </div>

                        <div className="flex items-start justify-end lg:justify-center">
                          <button
                            type="button"
                            onClick={() => setExpandedJobId((current) => (current === job.job_id ? null : job.job_id))}
                            className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 transition-colors hover:border-primary-200 hover:bg-primary-50 hover:text-primary-700"
                            aria-label={expanded ? 'Collapse run profile' : 'Expand run profile'}
                          >
                            {expanded ? <ChevronUpIcon className="h-4 w-4" /> : <ChevronDownIcon className="h-4 w-4" />}
                          </button>
                        </div>
                      </div>

                      {expanded ? (
                        <div className="mt-4 rounded-[1.5rem] border border-slate-200/90 bg-slate-50/70 p-4">
                          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
                            {[
                              { label: 'Provider', value: providerLabel },
                              { label: 'Model', value: modelLabel },
                              { label: 'OCR', value: ocrLabel },
                              { label: 'Scan', value: scanLabel },
                              {
                                label: 'Signals',
                                value: [
                                  job.settings?.enable_search ? 'Search on' : 'Search off',
                                  job.settings?.enable_vision_board ? 'Vision board on' : 'Vision board off',
                                  job.settings?.enable_llm_frame ? 'LLM frame on' : 'LLM frame off',
                                ].join(' · '),
                              },
                            ].map((item) => (
                              <div key={`${job.job_id}-${item.label}`} className="rounded-[1.25rem] border border-slate-200 bg-white px-4 py-3 shadow-sm">
                                <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">{item.label}</div>
                                <div className="mt-2 text-sm font-semibold text-slate-900 break-words">{item.value}</div>
                              </div>
                            ))}
                          </div>

                          <div className="mt-4 flex flex-col gap-3 border-t border-slate-200 pt-4 lg:flex-row lg:items-center lg:justify-between">
                            <div className="min-w-0">
                              <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">Source</div>
                              <div className="mt-1 truncate text-sm text-slate-600" title={sourceMeta.title}>
                                {sourceMeta.title}
                              </div>
                              <div className="mt-2 font-mono text-[11px] text-slate-400" title={job.job_id}>
                                {job.job_id}
                              </div>
                            </div>
                            <div className="flex flex-wrap items-center gap-3">
                              <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                                {job.mode || '—'}
                              </span>
                              <Link
                                to={`/jobs/${job.job_id}`}
                                className="bell-button-secondary h-10 gap-2 px-4 text-xs uppercase tracking-[0.18em]"
                              >
                                Open details
                              </Link>
                            </div>
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </section>

      {hasSelection ? (
        <div className="fixed bottom-6 left-1/2 z-30 flex -translate-x-1/2 items-center gap-4 rounded-full border border-primary-900/80 bg-primary-950 px-5 py-3 text-white shadow-[0_22px_44px_rgba(0,18,43,0.32)]">
          <div className="text-sm font-semibold">
            {selectedJobs.size} job{selectedJobs.size > 1 ? 's' : ''} selected
          </div>
          <div className="h-5 w-px bg-white/16" />
          <button
            type="button"
            onClick={handleBulkDelete}
            disabled={deleteLoading}
            className="inline-flex items-center gap-2 text-sm font-semibold text-rose-200 transition-colors hover:text-white disabled:opacity-50"
          >
            {deleteLoading ? <UpdateIcon className="h-4 w-4 animate-spin" /> : <TrashIcon className="h-4 w-4" />}
            {deleteLoading ? 'Deleting…' : 'Delete'}
          </button>
          <button
            type="button"
            onClick={() => setSelectedJobs(new Set())}
            className="inline-flex items-center gap-2 text-sm font-semibold text-white/78 transition-colors hover:text-white"
          >
            <Cross2Icon className="h-4 w-4" />
            Clear
          </button>
        </div>
      ) : null}
    </div>
  );
}
