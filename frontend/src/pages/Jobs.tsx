import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { deleteJobsBulk, getClusterJobs, getProviderModels, submitFilePath, submitFolderPath, submitUrls } from '../lib/api';
import type { JobStatus, JobSettings } from '../lib/api';
import { PlayIcon, UpdateIcon, MagnifyingGlassIcon, ClockIcon, TrashIcon, InfoCircledIcon } from '@radix-ui/react-icons';
import { formatDistanceToNow } from 'date-fns';
import { HelpTooltip } from '../components/HelpTooltip';

type InputMode = 'urls' | 'filepath' | 'dirpath';
const PROVIDER_OPTIONS = ['Ollama', 'LM Studio', 'Llama Server', 'Gemini CLI'] as const;
const CATEGORY_EMBEDDING_MODEL_OPTIONS = [
  'BAAI/bge-large-en-v1.5',
  'jinaai/jina-embeddings-v3',
  'Alibaba-NLP/gte-large-en-v1.5',
  'sentence-transformers/all-mpnet-base-v2',
  'sentence-transformers/all-MiniLM-L6-v2',
  'google/embeddinggemma-300m',
] as const;
const panelClass =
  'bell-panel';
const controlClass =
  'h-10 w-full rounded-[1.15rem] border border-slate-300 bg-white px-3 text-sm text-slate-700 shadow-sm transition-colors focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15';
const monoControlClass = `${controlClass} font-mono`;
const queueColumns = 'xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_minmax(0,1fr)_220px]';
const queueFrameColumns = 'grid-cols-[36px_minmax(0,1fr)]';

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

function getStatusBadgeClass(status: string): string {
  if (status === 'completed') return 'bg-primary-50 text-primary-700 border-primary-200';
  if (status === 'failed') return 'bg-red-50 text-red-700 border-red-200';
  if (status === 'processing') return 'bg-primary-100 text-primary-800 border-primary-200';
  if (status === 're-queued') return 'bg-[rgba(247,244,237,0.95)] text-slate-700 border-[#e5d9c7]';
  return 'bg-[rgba(247,244,237,0.95)] text-slate-700 border-[#e5d9c7]';
}

function getStatusText(status: string): string {
  return status === 're-queued' ? 'waiting (recovered)' : status;
}

function isTerminalStatus(status: string): boolean {
  return status === 'completed' || status === 'failed';
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

  // Submit form state
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

  // Filtering
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
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
    ]
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
    } catch (err) {
      console.error(err);
    } finally {
      setSubmitLoading(false);
    }
  };

  const filteredJobs = jobs.filter((j) => {
    if (statusFilter !== 'all') {
      if (statusFilter === 'queued') {
        if (j.status !== 'queued' && j.status !== 're-queued') return false;
      } else if (j.status !== statusFilter) {
        return false;
      }
    }
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      j.job_id.toLowerCase().includes(q) ||
      (j.brand || '').toLowerCase().includes(q) ||
      (j.category || '').toLowerCase().includes(q) ||
      (j.url || '').toLowerCase().includes(q)
    );
  });

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

  const disableSubmit =
    submitLoading ||
    (inputMode === 'urls' && !urls.trim()) ||
    (inputMode === 'filepath' && !filePath.trim()) ||
    (inputMode === 'dirpath' && !folderPath.trim());

  const toggleSelectJob = (jobId: string) => {
    setSelectedJobs((prev) => {
      const next = new Set(prev);
      if (next.has(jobId)) {
        next.delete(jobId);
      } else {
        next.add(jobId);
      }
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selectedJobs.size === filteredJobs.length) {
      setSelectedJobs(new Set());
      return;
    }
    setSelectedJobs(new Set(filteredJobs.map((job) => job.job_id)));
  };

  const handleSearchChange = (value: string) => {
    setSearch(value);
    setSelectedJobs(new Set());
  };

  const handleStatusFilterChange = (value: string) => {
    setStatusFilter(value);
    setSelectedJobs(new Set());
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
      <div className="bell-hero">
        <div className="relative z-10 mb-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="bell-badge">
              <PlayIcon className="h-3.5 w-3.5" />
              Launch pipeline
            </div>
            <h2 className="mt-4 max-w-2xl text-[2.9rem] font-bold text-primary-700">Start a new analysis run.</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
              Pick the evidence path, model runtime, and OCR profile. The defaults stay fast for normal work; the advanced controls help when you need to probe harder edge cases.
            </p>
          </div>
          <div className="grid min-w-[17rem] grid-cols-2 gap-3">
            <div className="rounded-[1.7rem] border border-white/90 bg-white/88 px-4 py-4 shadow-[0_14px_28px_rgba(0,55,120,0.08)]">
              <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">Primary mode</div>
              <div className="mt-2 text-lg font-bold text-slate-950">{mode === 'pipeline' ? 'Pipeline' : 'Agent'}</div>
            </div>
            <div className="rounded-[1.7rem] border border-primary-700/30 bg-primary-700 px-4 py-4 text-white shadow-[0_18px_36px_rgba(0,55,120,0.22)]">
              <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-white/65">Model host</div>
              <div className="mt-2 text-lg font-bold text-white">{provider}</div>
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="relative z-10 flex flex-col gap-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="inline-flex w-fit rounded-full border border-primary-100 bg-white/88 p-1.5 shadow-[0_12px_24px_rgba(0,55,120,0.08)]">
              <button type="button" onClick={() => setInputMode('urls')} className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] transition-colors ${inputMode === 'urls' ? 'bg-primary-500 text-white shadow-[0_10px_20px_rgba(0,112,206,0.20)]' : 'text-primary-700 hover:bg-primary-50'}`}>URLs</button>
              <button type="button" onClick={() => setInputMode('filepath')} className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] transition-colors ${inputMode === 'filepath' ? 'bg-primary-500 text-white shadow-[0_10px_20px_rgba(0,112,206,0.20)]' : 'text-primary-700 hover:bg-primary-50'}`}>File Path</button>
              <button type="button" onClick={() => setInputMode('dirpath')} className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] transition-colors ${inputMode === 'dirpath' ? 'bg-primary-500 text-white shadow-[0_10px_20px_rgba(0,112,206,0.20)]' : 'text-primary-700 hover:bg-primary-50'}`}>Directory Path</button>
            </div>

            <div className="flex items-center justify-between gap-3 lg:min-w-[18rem] lg:justify-end">
              <div className="hidden lg:block text-right">
                <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">Ready to run</div>
                <div className="mt-1 text-sm text-slate-500">
                  {inputMode === 'urls' ? 'Submit the current URL batch.' : inputMode === 'filepath' ? 'Analyze one server-side file path.' : 'Scan one server-side directory.'}
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
      </div>

      <div className={`${panelClass} flex flex-col overflow-hidden`}>
        <div className="border-b border-slate-200/80 bg-primary-50/70 px-6 py-5">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-3">
                <h3 className="text-xl font-bold text-slate-950">Job queue</h3>
                <div className="bell-data-pill">
                  <ClockIcon className="h-3.5 w-3.5 text-primary-500" />
                  Auto-syncing
                </div>
              </div>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
                Track active runs, spot failures, and jump straight into the evidence trail.
              </p>
              <div className="mt-2 text-[11px] font-medium uppercase tracking-[0.18em] text-slate-400">
                Last refreshed {formatDistanceToNow(lastUpdated, { addSuffix: true })}
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_220px_48px] xl:min-w-[42rem]">
              <div className="relative">
                <MagnifyingGlassIcon className="absolute left-3 top-3.5 h-4 w-4 text-slate-400" />
                <input
                  value={search}
                  onChange={(e) => handleSearchChange(e.target.value)}
                  placeholder="Search job, brand, category..."
                  className="h-11 w-full rounded-full border border-slate-300 bg-white pl-10 pr-3 font-mono text-sm text-slate-700 shadow-sm focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15"
                />
              </div>
              <select
                value={statusFilter}
                onChange={(e) => handleStatusFilterChange(e.target.value)}
                className="h-11 rounded-full border border-slate-300 bg-white px-4 text-sm font-semibold tracking-wide text-slate-700 shadow-sm focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15"
              >
                <option value="all">ALL STATUSES</option>
                <option value="queued">QUEUED</option>
                <option value="re-queued">RE-QUEUED</option>
                <option value="processing">PROCESSING</option>
                <option value="completed">COMPLETED</option>
                <option value="failed">FAILED</option>
              </select>
              <button
                onClick={fetchJobs}
                className="flex h-11 w-11 items-center justify-center rounded-full border border-slate-300 bg-white text-primary-700 shadow-sm transition-colors hover:border-primary-300 hover:bg-primary-50"
                title="Refresh queue"
              >
                <UpdateIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {hasSelection && (
          <div className="flex items-center justify-between border-b border-rose-200 bg-rose-50/80 px-6 py-3 animate-in fade-in slide-in-from-top-2 duration-200">
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-700">
                <span className="font-bold text-gray-900">{selectedJobs.size}</span> job{selectedJobs.size > 1 ? 's' : ''} selected
              </span>
              <button
                onClick={() => setSelectedJobs(new Set())}
                className="text-xs text-gray-400 hover:text-gray-700 transition-colors underline"
              >
                Clear selection
              </button>
            </div>
            <button
              onClick={handleBulkDelete}
              disabled={deleteLoading}
              className="flex items-center gap-2 px-4 py-2 text-xs font-bold uppercase tracking-wider rounded-lg bg-red-600 hover:bg-red-500 active:bg-red-700 text-white transition-colors disabled:opacity-50 shadow-sm"
            >
              {deleteLoading ? (
                <UpdateIcon className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <TrashIcon className="w-3.5 h-3.5" />
              )}
              {deleteLoading ? 'Deleting...' : `Delete ${selectedJobs.size} Job${selectedJobs.size > 1 ? 's' : ''}`}
            </button>
          </div>
        )}

        <div className="border-b border-slate-200/80 bg-white/85 px-6 py-4">
          <div className="overflow-hidden rounded-[1.9rem] border border-slate-200 bg-white shadow-[0_16px_32px_rgba(0,55,120,0.06)]">
            <div className="grid grid-cols-2 divide-x divide-y divide-slate-200 xl:grid-cols-5 xl:divide-y-0">
              <div className="flex min-h-[110px] flex-col gap-3 px-5 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">Jobs</div>
                <div className="text-2xl font-bold text-slate-950">{summary.total}</div>
                <div className="mt-auto text-xs text-slate-500">{summary.visible} visible</div>
              </div>
              <div className="flex min-h-[110px] flex-col gap-3 bg-primary-50/65 px-5 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-primary-600">Processing</div>
                <div className="text-2xl font-bold text-primary-800">{summary.processing}</div>
                <div className="mt-auto text-xs text-primary-700">active now</div>
              </div>
              <div className="flex min-h-[110px] flex-col gap-3 bg-slate-50 px-5 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">Queued</div>
                <div className="text-2xl font-bold text-slate-900">{summary.queued}</div>
                <div className="mt-auto text-xs text-slate-600">waiting or re-queued</div>
              </div>
              <div className="flex min-h-[110px] flex-col gap-3 bg-[rgba(221,238,250,0.65)] px-5 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-primary-600">Completed</div>
                <div className="text-2xl font-bold text-primary-800">{summary.completed}</div>
                <div className="mt-auto text-xs text-primary-700">finished successfully</div>
              </div>
              <div className="flex min-h-[110px] flex-col gap-3 bg-red-50/55 px-5 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-red-500">Failed</div>
                <div className="text-2xl font-bold text-red-800">{summary.failed}</div>
                <div className="mt-auto text-xs text-red-700">needs attention</div>
              </div>
            </div>
          </div>
        </div>

        <div className="border-b border-slate-200/80 bg-white/75 px-6 py-4">
          <div className={`grid items-center gap-4 ${queueFrameColumns}`}>
            <div className="flex items-center justify-center pt-0.5">
              <input
                type="checkbox"
                checked={isAllSelected}
                onChange={toggleSelectAll}
                className="w-3.5 h-3.5 rounded border-gray-300 bg-gray-100 text-primary-500 focus:ring-primary-500/30 cursor-pointer"
                title={isAllSelected ? 'Deselect all visible jobs' : 'Select all visible jobs'}
              />
            </div>
            <div className={`grid items-center gap-4 ${queueColumns}`}>
              <div>
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">Identity & Source</div>
                <div className="mt-1 text-sm text-slate-600">Select visible jobs for bulk actions.</div>
              </div>
              <div className="hidden xl:block text-[10px] uppercase tracking-[0.22em] text-slate-500 font-semibold">Classification</div>
              <div className="hidden xl:block text-[10px] uppercase tracking-[0.22em] text-slate-500 font-semibold">Execution State</div>
              <div className="hidden xl:block text-right text-[10px] uppercase tracking-[0.22em] text-slate-500 font-semibold">Runtime</div>
            </div>
          </div>
        </div>

        <div className="min-h-[400px] divide-y divide-gray-100">
          {loading && jobs.length === 0 ? (
            <div className="px-6 py-12 text-center text-gray-400">Syncing node cluster state...</div>
          ) : filteredJobs.length === 0 ? (
            <div className="px-6 py-12 text-center text-gray-400">No jobs found.</div>
          ) : filteredJobs.map((job) => {
            const selected = selectedJobs.has(job.job_id);
            const progressValue =
              job.status === 'completed'
                ? 100
                : job.status === 'processing'
                  ? Math.max(0, Math.min(100, Number(job.progress || 0)))
                  : 0;
            const providerLabel = job.settings?.provider || '—';
            const modelLabel = job.settings?.model_name || '—';
            const ocrEngineLabel = job.settings?.ocr_engine || '—';
            const ocrModeLabel = sanitizeDecorativeLabel(job.settings?.ocr_mode);
            const scanModeLabel = sanitizeDecorativeLabel(job.settings?.scan_mode);
            const categoryLabel = job.category || '—';
            const brandLabel = job.brand || '—';
            const categoryId = (job.category_id || '').trim();
            const terminal = isTerminalStatus(job.status);
            const ageLabel =
              job.status === 'processing'
                ? 'Running Since'
                : job.status === 'queued' || job.status === 're-queued'
                  ? 'Queued Since'
                  : 'Age';

            return (
              <div
                key={job.job_id}
                className={`px-6 py-5 transition-colors ${selected ? 'bg-primary-50/70' : 'bg-white hover:bg-gray-50/80'}`}
              >
                <div className={`grid items-start gap-4 ${queueFrameColumns}`}>
                  <div className="pt-1" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() => toggleSelectJob(job.job_id)}
                      className="w-3.5 h-3.5 rounded border-gray-300 bg-gray-100 text-primary-500 focus:ring-primary-500/30 cursor-pointer"
                    />
                  </div>

                  <div className={`min-w-0 grid grid-cols-1 gap-4 ${queueColumns}`}>
                    <div className="min-w-0 space-y-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <Link
                          to={`/jobs/${job.job_id}`}
                          className="font-mono text-xs text-primary-600 hover:text-primary-700 transition-colors break-all"
                        >
                          {job.job_id}
                        </Link>
                        <span className={`px-2 py-1 rounded inline-flex text-[10px] font-bold tracking-wider uppercase border ${getStatusBadgeClass(job.status)} ${job.status === 'processing' ? 'animate-pulse' : ''}`}>
                          {/** Native title tooltip for compressed row status */}
                          <span title={`Status: ${getStatusText(job.status)}`}>
                            {getStatusText(job.status)}
                          </span>
                        </span>
                        <span
                          title={`Execution mode: ${job.mode || '—'}`}
                          className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-slate-600"
                        >
                          {job.mode || '—'}
                        </span>
                      </div>
                      <div
                        className="break-all font-mono text-[11px] text-slate-600"
                        title={job.url}
                      >
                        {job.url}
                      </div>
                      <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
                        <span className="inline-flex items-center rounded border border-slate-200 bg-slate-50 px-2 py-1">
                          <span title={`Provider: ${providerLabel}`}>{providerLabel}</span>
                        </span>
                        <span
                          className="inline-flex max-w-full items-center truncate rounded border border-slate-200 bg-slate-50 px-2 py-1 font-mono"
                          title={`Model: ${modelLabel}`}
                        >
                          {modelLabel}
                        </span>
                        <span
                          className="inline-flex items-center rounded border border-slate-200 bg-slate-50 px-2 py-1"
                          title={`OCR engine: ${ocrEngineLabel}`}
                        >
                          {ocrEngineLabel}
                        </span>
                        <span
                          className="inline-flex items-center rounded border border-slate-200 bg-slate-50 px-2 py-1"
                          title={`OCR mode: ${ocrModeLabel}`}
                        >
                          {ocrModeLabel}
                        </span>
                        <span
                          className="inline-flex items-center rounded border border-slate-200 bg-slate-50 px-2 py-1"
                          title={`Scan mode: ${scanModeLabel}`}
                        >
                          {scanModeLabel}
                        </span>
                      </div>
                    </div>

                    <div className="min-w-0 space-y-2">
                      <div>
                        <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Brand</div>
                        <div className="mt-1 break-words text-sm font-semibold text-slate-900">{brandLabel}</div>
                      </div>
                      <div>
                        <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Category</div>
                        <div className="mt-1 flex flex-wrap items-center gap-2">
                            <span
                              className="break-words text-sm text-slate-800"
                              title={`Category: ${categoryLabel}`}
                            >
                              {categoryLabel}
                            </span>
                          {categoryId && (
                            <span
                              title={`Category ID: ${categoryId}`}
                              className="inline-flex items-center rounded-full border border-primary-200 bg-primary-50 px-2 py-0.5 text-[10px] font-mono font-semibold text-primary-700"
                            >
                              ID {categoryId}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="min-w-0 space-y-2">
                      <div>
                        <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Stage</div>
                        <div className="mt-1 text-sm font-semibold capitalize text-slate-900">
                          <span title={`Stage: ${formatStageLabel(job.stage)}`}>
                            {formatStageLabel(job.stage)}
                          </span>
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Detail</div>
                        <div
                          className="mt-1 break-words text-xs text-slate-700"
                          title={job.stage_detail || '—'}
                        >
                          {job.stage_detail || '—'}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-500">
                          <span>Progress</span>
                          <span className="font-mono text-slate-700">
                            {job.status === 'completed'
                              ? '100%'
                              : job.status === 'processing'
                                ? `${progressValue.toFixed(1)}%`
                                : '—'}
                          </span>
                        </div>
                        <div className="h-1.5 rounded-full bg-gray-100 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              job.status === 'failed'
                                ? 'bg-red-400'
                                : job.status === 'completed'
                                  ? 'bg-primary-500'
                                  : 'bg-primary-500'
                            }`}
                            style={{ width: `${job.status === 'failed' ? 100 : progressValue}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="min-w-0 flex flex-col xl:items-end gap-2">
                      <div className={`grid gap-2 w-full xl:w-auto ${terminal ? 'grid-cols-1' : 'grid-cols-2 xl:grid-cols-1'}`}>
                        <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                          <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Duration</div>
                          <div
                            className="mt-1 text-sm font-mono text-slate-900"
                            title={job.duration_seconds != null ? `${job.duration_seconds.toFixed(3)} seconds` : 'No duration recorded'}
                          >
                            {formatDurationLabel(job.duration_seconds)}
                          </div>
                        </div>
                        {!terminal && (
                          <div className="rounded-lg border border-primary-100 bg-primary-50/70 px-3 py-2">
                            <div className="text-[10px] font-semibold uppercase tracking-wider text-primary-600">{ageLabel}</div>
                            <div className="mt-1 text-sm text-primary-900" title={job.created_at}>
                              {formatRelativeTimestamp(job.created_at)}
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="text-right text-xs text-slate-600" title={job.updated_at}>
                        Updated {formatRelativeTimestamp(job.updated_at)}
                      </div>
                      <Link
                        to={`/jobs/${job.job_id}`}
                        className="inline-flex items-center justify-center rounded-lg border border-primary-200 bg-primary-50 px-3 py-2 text-xs font-semibold text-primary-700 hover:bg-primary-100 transition-colors"
                      >
                        Open details
                      </Link>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
