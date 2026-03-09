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
const panelClass =
  'rounded-[30px] border border-slate-200/80 bg-white/82 shadow-[0_18px_45px_rgba(15,23,42,0.06)] backdrop-blur';
const controlClass =
  'h-10 w-full rounded-2xl border border-slate-200 bg-white/90 px-3 text-sm text-slate-700 shadow-sm transition-colors focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15';
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
  if (status === 'completed') return 'bg-emerald-50 text-emerald-700 border-emerald-200';
  if (status === 'failed') return 'bg-red-50 text-red-700 border-red-200';
  if (status === 'processing') return 'bg-blue-50 text-blue-700 border-blue-200';
  if (status === 're-queued') return 'bg-orange-50 text-orange-700 border-orange-200';
  return 'bg-amber-50 text-amber-700 border-amber-200';
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

function FieldLabel({
  label,
  help,
}: {
  label: string;
  help?: string;
}) {
  return (
    <label className="flex items-center gap-1.5 text-xs uppercase tracking-wider font-semibold text-gray-400">
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
  const [provider, setProvider] = useState('Ollama');
  const [modelName, setModelName] = useState('qwen3-vl:8b-instruct');
  const [providerModels, setProviderModels] = useState<string[]>([]);
  const [providerModelsLoading, setProviderModelsLoading] = useState(false);
  const [ocrEngine, setOcrEngine] = useState('EasyOCR');
  const [ocrMode, setOcrMode] = useState('🚀 Fast');
  const [scanMode, setScanMode] = useState('Tail Only');
  const [expressMode, setExpressMode] = useState(false);
  const [enableVisionBoard, setEnableVisionBoard] = useState(true);
  const [enableLlmFrame, setEnableLlmFrame] = useState(true);
  const [enableWebSearch, setEnableWebSearch] = useState(true);
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
      ocr_engine: ocrEngine,
      ocr_mode: ocrMode,
      scan_mode: scanMode,
      express_mode: expressMode,
      override: false,
      enable_search: enableWebSearch,
      enable_web_search: enableWebSearch,
      enable_vision_board: enableVisionBoard,
      enable_llm_frame: enableLlmFrame,
      context_size: contextSize,
    }),
    [
      categories,
      provider,
      modelName,
      ocrEngine,
      ocrMode,
      scanMode,
      expressMode,
      enableWebSearch,
      enableVisionBoard,
      enableLlmFrame,
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
      <div className={`${panelClass} p-6`}>
        <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-primary-200 bg-primary-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.24em] text-primary-700">
              <PlayIcon className="h-3.5 w-3.5" />
              Launch pipeline
            </div>
            <h2 className="mt-4 text-3xl font-black tracking-[-0.05em] text-slate-950">Start a new analysis run.</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
              Pick the evidence path, model runtime, and OCR profile. The defaults stay fast for normal work; the advanced controls help when you need to probe harder edge cases.
            </p>
          </div>
          <div className="grid min-w-[17rem] grid-cols-2 gap-3">
            <div className="rounded-[22px] border border-slate-200/80 bg-slate-50/80 px-4 py-4 shadow-sm">
              <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">Primary mode</div>
              <div className="mt-2 text-lg font-black tracking-[-0.04em] text-slate-950">{mode === 'pipeline' ? 'Pipeline' : 'Agent'}</div>
            </div>
            <div className="rounded-[22px] border border-slate-200/80 bg-slate-950 px-4 py-4 text-slate-50 shadow-[0_18px_36px_rgba(15,23,42,0.18)]">
              <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">Model host</div>
              <div className="mt-2 text-lg font-black tracking-[-0.04em] text-white">{provider}</div>
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="inline-flex w-fit rounded-full border border-slate-200 bg-slate-50/85 p-1 shadow-inner">
              <button type="button" onClick={() => setInputMode('urls')} className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] transition-colors ${inputMode === 'urls' ? 'bg-primary-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-900'}`}>URLs</button>
              <button type="button" onClick={() => setInputMode('filepath')} className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] transition-colors ${inputMode === 'filepath' ? 'bg-primary-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-900'}`}>File Path</button>
              <button type="button" onClick={() => setInputMode('dirpath')} className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] transition-colors ${inputMode === 'dirpath' ? 'bg-primary-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-900'}`}>Directory Path</button>
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
                className="flex h-12 min-w-[13rem] items-center justify-center gap-2 rounded-[20px] bg-[linear-gradient(135deg,#4f46e5,#2563eb)] px-5 text-sm font-bold uppercase tracking-[0.24em] text-white shadow-[0_18px_36px_rgba(79,70,229,0.22)] transition-transform duration-200 hover:-translate-y-0.5 hover:shadow-[0_24px_44px_rgba(79,70,229,0.26)] disabled:opacity-50"
              >
                {submitLoading ? <UpdateIcon className="h-4 w-4 animate-spin" /> : <PlayIcon className="h-4 w-4" />}
                {submitLoading ? 'Submitting…' : 'Execute'}
              </button>
            </div>
          </div>

          <div className="rounded-[26px] border border-slate-200/80 bg-slate-50/70 p-3 shadow-inner">
            {inputMode === 'urls' && (
              <textarea
                value={urls}
                onChange={(e) => setUrls(e.target.value)}
                placeholder="Enter URLs (one per line)..."
                className="h-36 w-full resize-none rounded-[20px] border border-slate-200 bg-white/90 p-4 font-mono text-sm text-slate-700 shadow-sm transition-colors focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15"
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
            <div className="flex items-start gap-3 rounded-[22px] border border-sky-100 bg-sky-50/75 px-4 py-3 text-sm text-slate-600">
              <InfoCircledIcon className="mt-0.5 h-4 w-4 shrink-0 text-sky-500" />
              <div className="space-y-1">
                <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-sky-700">Server path note</div>
                <div>
                  File and directory paths are resolved on the backend server, not in your browser. UNC paths only work if the server can reach that share and has permission to read it.
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3 rounded-[28px] border border-slate-200/80 bg-[linear-gradient(180deg,rgba(248,250,252,0.85)_0%,rgba(241,245,249,0.92)_100%)] p-5 md:grid-cols-4">
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
                <option value="true">📸 Generate Vision Board (SigLIP/OpenCLIP)</option>
                <option value="false">Disabled</option>
              </select>
            </div>
            <div className="space-y-1.5">
              <FieldLabel
                label="LLM Keyframe"
                help="Sends a representative video frame to the multimodal model. Disable this only if you want the LLM to rely on OCR text alone."
              />
              <select value={enableLlmFrame ? 'true' : 'false'} onChange={(e) => setEnableLlmFrame(e.target.value === 'true')} className={controlClass}>
                <option value="true">🧠 Send Keyframe to LLM</option>
                <option value="false">Disabled</option>
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
                <option value="🚀 Fast">Fast</option>
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

      <div className={`${panelClass} overflow-hidden flex flex-col`}>
        <div className="border-b border-slate-200/80 bg-white/75 px-6 py-5">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-3">
                <h3 className="text-xl font-black tracking-[-0.04em] text-slate-950">Job queue</h3>
                <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500 shadow-inner">
                  <ClockIcon className="h-3.5 w-3.5 text-emerald-500" />
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
                  className="h-11 w-full rounded-2xl border border-slate-200 bg-white pl-10 pr-3 text-sm text-slate-700 shadow-sm focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15 font-mono"
                />
              </div>
              <select
                value={statusFilter}
                onChange={(e) => handleStatusFilterChange(e.target.value)}
                className="h-11 rounded-2xl border border-slate-200 bg-white px-4 text-sm font-semibold tracking-wide text-slate-700 shadow-sm focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15"
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
                className="flex h-11 w-11 items-center justify-center rounded-2xl border border-slate-200 bg-white text-slate-600 shadow-sm transition-colors hover:border-slate-300 hover:bg-slate-50"
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

        <div className="border-b border-slate-200/80 bg-slate-50/70 px-6 py-4">
          <div className="overflow-hidden rounded-[26px] border border-slate-200 bg-white/80 shadow-sm">
            <div className="grid grid-cols-2 divide-x divide-y divide-slate-200 xl:grid-cols-5 xl:divide-y-0">
              <div className="flex min-h-[102px] flex-col justify-between px-5 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">Jobs</div>
                <div className="flex items-end justify-between gap-3">
                  <div className="text-2xl font-black tracking-[-0.04em] text-slate-950">{summary.total}</div>
                  <div className="text-xs text-slate-500">{summary.visible} visible</div>
                </div>
              </div>
              <div className="flex min-h-[102px] flex-col justify-between bg-blue-50/55 px-5 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-blue-500">Processing</div>
                <div className="flex items-end justify-between gap-3">
                  <div className="text-2xl font-black tracking-[-0.04em] text-blue-800">{summary.processing}</div>
                  <div className="text-xs text-blue-700">active</div>
                </div>
              </div>
              <div className="flex min-h-[102px] flex-col justify-between bg-amber-50/55 px-5 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-amber-500">Queued</div>
                <div className="flex items-end justify-between gap-3">
                  <div className="text-2xl font-black tracking-[-0.04em] text-amber-800">{summary.queued}</div>
                  <div className="text-xs text-amber-700">waiting</div>
                </div>
              </div>
              <div className="flex min-h-[102px] flex-col justify-between bg-emerald-50/55 px-5 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-emerald-500">Completed</div>
                <div className="flex items-end justify-between gap-3">
                  <div className="text-2xl font-black tracking-[-0.04em] text-emerald-800">{summary.completed}</div>
                  <div className="text-xs text-emerald-700">done</div>
                </div>
              </div>
              <div className="flex min-h-[102px] flex-col justify-between bg-red-50/55 px-5 py-4">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-red-500">Failed</div>
                <div className="flex items-end justify-between gap-3">
                  <div className="text-2xl font-black tracking-[-0.04em] text-red-800">{summary.failed}</div>
                  <div className="text-xs text-red-700">review</div>
                </div>
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
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">Identity & Source</div>
                <div className="mt-1 text-sm text-slate-500">Select visible jobs for bulk actions.</div>
              </div>
              <div className="hidden xl:block text-[10px] uppercase tracking-[0.22em] text-slate-400 font-semibold">Classification</div>
              <div className="hidden xl:block text-[10px] uppercase tracking-[0.22em] text-slate-400 font-semibold">Execution State</div>
              <div className="hidden xl:block text-right text-[10px] uppercase tracking-[0.22em] text-slate-400 font-semibold">Runtime</div>
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
            const ocrModeLabel = job.settings?.ocr_mode || '—';
            const scanModeLabel = job.settings?.scan_mode || '—';
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
                          className="px-2 py-1 rounded border border-gray-200 bg-gray-50 text-[10px] uppercase tracking-wider font-semibold text-gray-500"
                        >
                          {job.mode || '—'}
                        </span>
                      </div>
                      <div
                        className="text-[11px] text-gray-500 font-mono break-all"
                        title={job.url}
                      >
                        {job.url}
                      </div>
                      <div className="flex flex-wrap items-center gap-2 text-[11px] text-gray-500">
                        <span className="inline-flex items-center rounded border border-gray-200 bg-gray-50 px-2 py-1">
                          <span title={`Provider: ${providerLabel}`}>{providerLabel}</span>
                        </span>
                        <span
                          className="inline-flex items-center rounded border border-gray-200 bg-gray-50 px-2 py-1 font-mono max-w-full truncate"
                          title={`Model: ${modelLabel}`}
                        >
                          {modelLabel}
                        </span>
                        <span
                          className="inline-flex items-center rounded border border-gray-200 bg-gray-50 px-2 py-1"
                          title={`OCR engine: ${ocrEngineLabel}`}
                        >
                          {ocrEngineLabel}
                        </span>
                        <span
                          className="inline-flex items-center rounded border border-gray-200 bg-gray-50 px-2 py-1"
                          title={`OCR mode: ${ocrModeLabel}`}
                        >
                          {ocrModeLabel}
                        </span>
                        <span
                          className="inline-flex items-center rounded border border-gray-200 bg-gray-50 px-2 py-1"
                          title={`Scan mode: ${scanModeLabel}`}
                        >
                          {scanModeLabel}
                        </span>
                      </div>
                    </div>

                    <div className="min-w-0 space-y-2">
                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Brand</div>
                        <div className="mt-1 text-sm font-semibold text-gray-900 break-words">{brandLabel}</div>
                      </div>
                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Category</div>
                        <div className="mt-1 flex flex-wrap items-center gap-2">
                            <span
                              className="text-sm text-gray-700 break-words"
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
                        <div className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Stage</div>
                        <div className="mt-1 text-sm font-semibold text-gray-900 capitalize">
                          <span title={`Stage: ${formatStageLabel(job.stage)}`}>
                            {formatStageLabel(job.stage)}
                          </span>
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Detail</div>
                        <div
                          className="mt-1 text-xs text-gray-600 break-words"
                          title={job.stage_detail || '—'}
                        >
                          {job.stage_detail || '—'}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-gray-400">
                          <span>Progress</span>
                          <span className="font-mono text-gray-600">
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
                                  ? 'bg-emerald-500'
                                  : 'bg-primary-500'
                            }`}
                            style={{ width: `${job.status === 'failed' ? 100 : progressValue}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="min-w-0 flex flex-col xl:items-end gap-2">
                      <div className={`grid gap-2 w-full xl:w-auto ${terminal ? 'grid-cols-1' : 'grid-cols-2 xl:grid-cols-1'}`}>
                        <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2">
                          <div className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Duration</div>
                          <div
                            className="mt-1 text-sm font-mono text-gray-800"
                            title={job.duration_seconds != null ? `${job.duration_seconds.toFixed(3)} seconds` : 'No duration recorded'}
                          >
                            {formatDurationLabel(job.duration_seconds)}
                          </div>
                        </div>
                        {!terminal && (
                          <div className="rounded-lg border border-blue-100 bg-blue-50/70 px-3 py-2">
                            <div className="text-[10px] uppercase tracking-wider text-blue-500 font-semibold">{ageLabel}</div>
                            <div className="mt-1 text-sm text-blue-900" title={job.created_at}>
                              {formatRelativeTimestamp(job.created_at)}
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 text-right" title={job.updated_at}>
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
