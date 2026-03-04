import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { deleteJobsBulk, getClusterJobs, getProviderModels, submitFilePath, submitFolderPath, submitUrls } from '../lib/api';
import type { JobStatus, JobSettings } from '../lib/api';
import { PlayIcon, UpdateIcon, MagnifyingGlassIcon, ClockIcon, TrashIcon } from '@radix-ui/react-icons';
import { formatDistanceToNow } from 'date-fns';

type InputMode = 'urls' | 'filepath' | 'dirpath';
const PROVIDER_OPTIONS = ['Ollama', 'LM Studio', 'Llama Server', 'Gemini CLI'] as const;

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
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
          <PlayIcon className="w-5 h-5 text-primary-400" /> Start Analysis Job
        </h2>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => setInputMode('urls')} className={`px-3 py-1.5 text-xs rounded border ${inputMode === 'urls' ? 'bg-primary-600 border-primary-500 text-white' : 'bg-gray-50 border-gray-200 text-gray-700'}`}>URLs</button>
            <button type="button" onClick={() => setInputMode('filepath')} className={`px-3 py-1.5 text-xs rounded border ${inputMode === 'filepath' ? 'bg-primary-600 border-primary-500 text-white' : 'bg-gray-50 border-gray-200 text-gray-700'}`}>File Path</button>
            <button type="button" onClick={() => setInputMode('dirpath')} className={`px-3 py-1.5 text-xs rounded border ${inputMode === 'dirpath' ? 'bg-primary-600 border-primary-500 text-white' : 'bg-gray-50 border-gray-200 text-gray-700'}`}>Directory Path</button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
            <div className="lg:col-span-3">
              {inputMode === 'urls' && (
                <textarea
                  value={urls}
                  onChange={(e) => setUrls(e.target.value)}
                  placeholder="Enter URLs (one per line)..."
                  className="w-full h-32 p-3 text-sm bg-gray-50 border border-gray-200 rounded-lg text-gray-700 focus:ring-1 focus:ring-primary-500 font-mono shadow-inner resize-none"
                />
              )}
              {inputMode === 'filepath' && (
                <input
                  value={filePath}
                  onChange={(e) => setFilePath(e.target.value)}
                  placeholder={'C:\\videos\\ad.mp4 or \\\\server\\share\\ads\\spot.mp4'}
                  className="w-full h-12 px-3 text-sm bg-gray-50 border border-gray-200 rounded-lg text-gray-700 focus:ring-1 focus:ring-primary-500 font-mono shadow-inner"
                />
              )}
              {inputMode === 'dirpath' && (
                <input
                  value={folderPath}
                  onChange={(e) => setFolderPath(e.target.value)}
                  placeholder={'C:\\videos\\ads or \\\\server\\share\\ads or /mnt/media/ads'}
                  className="w-full h-12 px-3 text-sm bg-gray-50 border border-gray-200 rounded-lg text-gray-700 focus:ring-1 focus:ring-primary-500 font-mono shadow-inner"
                />
              )}
            </div>
            <div className="flex flex-col justify-end">
              <button
                type="submit"
                disabled={disableSubmit}
                className="w-full h-12 bg-primary-600 hover:bg-primary-500 active:bg-primary-700 text-white font-bold rounded-lg shadow disabled:opacity-50 transition-colors uppercase tracking-wider text-sm flex items-center justify-center gap-2"
              >
                {submitLoading ? <UpdateIcon className="animate-spin w-4 h-4" /> : <PlayIcon className="w-4 h-4" />}
                {submitLoading ? 'Submitting...' : 'Execute'}
              </button>
            </div>
          </div>

          <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
            File/Directory paths must be accessible to the backend server (not your browser). UNC paths require server access/permissions.
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 bg-gray-50/80 p-4 rounded-lg border border-gray-200">
            <div className="space-y-1">
              <label className="text-xs uppercase tracking-wider font-semibold text-gray-400">Mode</label>
              <select value={mode} onChange={(e) => setMode(e.target.value)} className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700">
                <option value="pipeline">Standard Pipeline</option>
                <option value="agent">ReACT Agent</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs uppercase tracking-wider font-semibold text-gray-400">Web Search</label>
              <select value={enableWebSearch ? 'true' : 'false'} onChange={(e) => setEnableWebSearch(e.target.value === 'true')} className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700">
                <option value="true">Enabled</option>
                <option value="false">Disabled</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs uppercase tracking-wider font-semibold text-gray-400">Scan Strategy</label>
              <select value={scanMode} onChange={(e) => setScanMode(e.target.value)} className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700">
                <option value="Tail Only">Tail Only</option>
                <option value="Full Video">Full Video</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs uppercase tracking-wider font-semibold text-gray-400">Vision Board</label>
              <select value={enableVisionBoard ? 'true' : 'false'} onChange={(e) => setEnableVisionBoard(e.target.value === 'true')} className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700">
                <option value="true">📸 Generate Vision Board (SigLIP/OpenCLIP)</option>
                <option value="false">Disabled</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs uppercase tracking-wider font-semibold text-gray-400">LLM Keyframe</label>
              <select value={enableLlmFrame ? 'true' : 'false'} onChange={(e) => setEnableLlmFrame(e.target.value === 'true')} className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700">
                <option value="true">🧠 Send Keyframe to LLM</option>
                <option value="false">Disabled</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs uppercase tracking-wider font-semibold text-gray-400">OCR Engine</label>
              <select value={ocrEngine} onChange={(e) => setOcrEngine(e.target.value)} className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700">
                <option value="EasyOCR">EasyOCR</option>
                <option value="Florence-2 (Microsoft)">Florence-2</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs uppercase tracking-wider font-semibold text-gray-400">OCR Mode</label>
              <select value={ocrMode} onChange={(e) => setOcrMode(e.target.value)} className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700">
                <option value="🚀 Fast">Fast</option>
                <option value="Detailed">Detailed</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs uppercase tracking-wider font-semibold text-gray-400">Context Limit</label>
              <input type="number" min={512} step={512} value={contextSize} onChange={(e) => setContextSize(Number(e.target.value || 8192))} className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700 font-mono" />
            </div>
            <div className="space-y-1">
              <label className="text-xs uppercase tracking-wider font-semibold text-gray-400">Provider</label>
              <select value={provider} onChange={(e) => setProvider(e.target.value)} className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700">
                {PROVIDER_OPTIONS.map((providerOption) => (
                  <option key={providerOption} value={providerOption}>{providerOption}</option>
                ))}
              </select>
            </div>
            <div className="space-y-1 md:col-span-2">
              <label className="text-xs uppercase tracking-wider font-semibold text-gray-400">Model</label>
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
                    className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700"
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
                      className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700"
                    />
                  )}
                </div>
              ) : (
                <input value={modelName} onChange={(e) => setModelName(e.target.value)} className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700" />
              )}
              {(providerName === 'ollama' || providerName === 'llama-server' || providerName === 'llama server') && providerModelsLoading && (
                <div className="text-[10px] text-gray-400">
                  Loading available {providerName === 'ollama' ? 'Ollama' : 'Llama Server'} models...
                </div>
              )}
            </div>
            <div className="space-y-1 md:col-span-4">
              <label className="text-xs uppercase tracking-wider font-semibold text-gray-400">Target Categories (Comma Separated)</label>
              <input value={categories} onChange={(e) => setCategories(e.target.value)} className="w-full h-8 text-xs bg-white border border-gray-200 rounded px-2 text-gray-700 font-mono" />
            </div>
          </div>
        </form>
      </div>

      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm flex flex-col">
        <div className="px-6 py-4 border-b border-gray-200 bg-white flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <h3 className="font-bold text-gray-900 tracking-wide">Job Queue</h3>
            <div className="flex items-center gap-1.5 text-xs text-gray-400 bg-gray-50 px-2 py-1 rounded shadow-inner border border-gray-200">
              <ClockIcon className="w-3 h-3 text-emerald-500" /> Auto-syncing ({formatDistanceToNow(lastUpdated, { addSuffix: true })})
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-2.5 top-2.5 w-4 h-4 text-gray-400" />
              <input value={search} onChange={(e) => handleSearchChange(e.target.value)} placeholder="Search job, brand, category..." className="pl-8 pr-3 py-1.5 text-xs bg-gray-50 border border-gray-200 rounded text-gray-700 w-56 focus:ring-1 focus:ring-primary-500 font-mono" />
            </div>
            <select value={statusFilter} onChange={(e) => handleStatusFilterChange(e.target.value)} className="py-1.5 px-3 text-xs bg-gray-50 border border-gray-200 rounded text-gray-700 font-medium tracking-wide">
              <option value="all">ALL STATUSES</option>
              <option value="queued">QUEUED</option>
              <option value="re-queued">RE-QUEUED</option>
              <option value="processing">PROCESSING</option>
              <option value="completed">COMPLETED</option>
              <option value="failed">FAILED</option>
            </select>
            <button onClick={fetchJobs} className="p-1.5 bg-gray-100 hover:bg-gray-200 rounded text-gray-700 transition-colors border border-gray-300">
              <UpdateIcon className="w-4 h-4" />
            </button>
          </div>
        </div>

        {hasSelection && (
          <div className="px-6 py-3 bg-red-50 border-b border-red-200 flex items-center justify-between animate-in fade-in slide-in-from-top-2 duration-200">
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

        <div className="overflow-x-auto min-h-[400px]">
          <table className="w-full text-sm text-left whitespace-nowrap">
            <thead className="text-[10px] uppercase font-bold tracking-wider text-gray-400 bg-gray-50/80">
              <tr>
                <th className="px-4 py-4 w-10">
                  <input
                    type="checkbox"
                    checked={isAllSelected}
                    onChange={toggleSelectAll}
                    className="w-3.5 h-3.5 rounded border-gray-300 bg-gray-100 text-primary-500 focus:ring-primary-500/30 cursor-pointer"
                    title={isAllSelected ? 'Deselect all' : 'Select all'}
                  />
                </th>
                <th className="px-6 py-4">Job</th>
                <th className="px-6 py-4">Brand</th>
                <th className="px-6 py-4">Category</th>
                <th className="px-6 py-4">Status</th>
                <th className="px-6 py-4">Mode</th>
                <th className="px-6 py-4">Stage</th>
                <th className="px-6 py-4 text-right">Progress</th>
                <th className="px-6 py-4">Duration</th>
                <th className="px-6 py-4">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {loading && jobs.length === 0 ? (
                <tr><td colSpan={10} className="px-6 py-12 text-center text-gray-400">Syncing node cluster state...</td></tr>
              ) : filteredJobs.length === 0 ? (
                <tr><td colSpan={10} className="px-6 py-12 text-center text-gray-400">No jobs found.</td></tr>
              ) : filteredJobs.map((job) => (
                <tr key={job.job_id} className={`hover:bg-gray-50 transition-colors group ${selectedJobs.has(job.job_id) ? 'bg-primary-50' : ''}`}>
                  <td className="px-4 py-4" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={selectedJobs.has(job.job_id)}
                      onChange={() => toggleSelectJob(job.job_id)}
                      className="w-3.5 h-3.5 rounded border-gray-300 bg-gray-100 text-primary-500 focus:ring-primary-500/30 cursor-pointer"
                    />
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex flex-col gap-1">
                      <Link to={`/jobs/${job.job_id}`} className="font-mono text-xs text-primary-600 group-hover:text-primary-700 transition-colors">{job.job_id}</Link>
                      <span className="text-[10px] text-gray-400 font-mono max-w-xs truncate">{job.url}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-xs text-gray-700">{job.brand || '—'}</td>
                  <td className="px-6 py-4 text-xs text-gray-700">{job.category || '—'}</td>
                  <td className="px-6 py-4">
                    {(() => {
                      const statusClass = job.status === 'completed'
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : job.status === 'failed'
                          ? 'bg-red-50 text-red-700 border-red-200'
                          : job.status === 'processing'
                            ? 'bg-blue-50 text-blue-700 border-blue-200 animate-pulse'
                            : job.status === 're-queued'
                              ? 'bg-orange-50 text-orange-700 border-orange-200'
                              : 'bg-amber-50 text-amber-700 border-amber-200';
                      const statusText = job.status === 're-queued' ? 'waiting (recovered)' : job.status;
                      return (
                    <span className={`px-2 py-1 rounded inline-flex text-[10px] font-bold tracking-wider uppercase border ${
                      statusClass
                    }`}>
                      {statusText}
                    </span>
                      );
                    })()}
                  </td>
                  <td className="px-6 py-4 text-[10px] uppercase font-bold tracking-widest text-gray-500">{job.mode || '—'}</td>
                  <td className="px-6 py-4 text-[10px] uppercase font-bold tracking-widest text-gray-500">{job.stage || '—'}</td>
                  <td className="px-6 py-4 font-mono text-xs text-right text-gray-700">
                    {job.status === 'processing' ? `${job.progress.toFixed(1)}%` : job.status === 'completed' ? '100%' : '—'}
                  </td>
                  <td className="px-6 py-4 font-mono text-xs text-gray-500">
                    {job.duration_seconds != null
                      ? job.duration_seconds < 60
                        ? `${job.duration_seconds.toFixed(1)}s`
                        : `${Math.floor(job.duration_seconds / 60)}m ${Math.round(job.duration_seconds % 60)}s`
                      : '—'}
                  </td>
                  <td className="px-6 py-4 font-mono text-[10px] text-gray-400">{job.created_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
