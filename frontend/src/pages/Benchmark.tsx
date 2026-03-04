import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import {
  ExclamationTriangleIcon,
  Pencil2Icon,
  RocketIcon,
  TrashIcon,
  UpdateIcon,
} from '@radix-ui/react-icons';
import {
  createBenchmarkTruth,
  deleteBenchmarkSuite,
  deleteBenchmarkTest,
  getBenchmarkSuite,
  getBenchmarkSuiteResults,
  getBenchmarkSuites,
  getBenchmarkTruths,
  getSystemProfile,
  runBenchmarkSuite,
  updateBenchmarkSuite,
  updateBenchmarkTest,
} from '../lib/api';
import type {
  BenchmarkPoint,
  BenchmarkSuiteDetail,
  BenchmarkSuiteResults,
  BenchmarkSuiteSummary,
  BenchmarkTruth,
  SystemProfile,
} from '../lib/api';

function formatSeconds(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—';
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}m ${seconds}s`;
}

function formatGbFromMb(valueMb: number | null | undefined): string {
  if (valueMb == null || !Number.isFinite(valueMb)) return '—';
  return `${(valueMb / 1024).toFixed(1)} GB`;
}

function safePercent(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—';
  return `${value.toFixed(1)}%`;
}

function truncate(text: string | null | undefined, max = 80): string {
  const value = String(text || '');
  if (!value) return '—';
  if (value.length <= max) return value;
  return `${value.slice(0, Math.max(0, max - 1))}…`;
}

interface ModalShellProps {
  title: string;
  open: boolean;
  onClose: () => void;
  children: ReactNode;
}

function ModalShell({ title, open, onClose, children }: ModalShellProps) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-2xl rounded-xl border border-gray-200 bg-white p-4 text-gray-900 shadow-2xl">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-base font-semibold text-gray-900">{title}</h3>
          <button
            type="button"
            onClick={onClose}
            className="rounded border border-gray-300 px-3 py-1 text-xs text-gray-600 hover:bg-gray-50"
          >
            Close
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

export function Benchmark() {
  const [profile, setProfile] = useState<SystemProfile | null>(null);
  const [truths, setTruths] = useState<BenchmarkTruth[]>([]);
  const [suites, setSuites] = useState<BenchmarkSuiteSummary[]>([]);
  const [selectedSuiteId, setSelectedSuiteId] = useState('');
  const [suiteDetail, setSuiteDetail] = useState<BenchmarkSuiteDetail | null>(null);
  const [suiteResults, setSuiteResults] = useState<BenchmarkSuiteResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');

  const [truthName, setTruthName] = useState('Golden Video');
  const [truthVideoUrl, setTruthVideoUrl] = useState('');
  const [truthExpectedOcr, setTruthExpectedOcr] = useState('');
  const [truthCategories, setTruthCategories] = useState('');
  const [truthExpectedBrand, setTruthExpectedBrand] = useState('');
  const [truthExpectedCategory, setTruthExpectedCategory] = useState('');
  const [truthExpectedConfidence, setTruthExpectedConfidence] = useState('');
  const [truthExpectedReasoning, setTruthExpectedReasoning] = useState('');
  const [runTruthId, setRunTruthId] = useState('');
  const [runCategories, setRunCategories] = useState('');

  const [suiteModalOpen, setSuiteModalOpen] = useState(false);
  const [suiteModalId, setSuiteModalId] = useState('');
  const [suiteModalName, setSuiteModalName] = useState('');
  const [suiteModalDescription, setSuiteModalDescription] = useState('');

  const [testModalOpen, setTestModalOpen] = useState(false);
  const [testModalId, setTestModalId] = useState('');
  const [testModalSourceUrl, setTestModalSourceUrl] = useState('');
  const [testModalExpectedCategory, setTestModalExpectedCategory] = useState('');
  const [testModalExpectedBrand, setTestModalExpectedBrand] = useState('');
  const [testModalExpectedConfidence, setTestModalExpectedConfidence] = useState('');
  const [testModalExpectedReasoning, setTestModalExpectedReasoning] = useState('');
  const [testModalExpectedOcr, setTestModalExpectedOcr] = useState('');

  const fetchBaseData = async () => {
    const [profilePayload, truthPayload, suitePayload] = await Promise.all([
      getSystemProfile(),
      getBenchmarkTruths(),
      getBenchmarkSuites(),
    ]);
    setProfile(profilePayload);
    setTruths(truthPayload.truths || []);
    setSuites(suitePayload.suites || []);

    if (!selectedSuiteId && suitePayload.suites?.length) {
      setSelectedSuiteId(suitePayload.suites[0].suite_id);
    }
    if (!runTruthId && truthPayload.truths?.length) {
      setRunTruthId(truthPayload.truths[0].truth_id);
    }
  };

  const refreshSelectedSuite = async (suiteId: string) => {
    if (!suiteId) {
      setSuiteDetail(null);
      setSuiteResults(null);
      return;
    }
    const [detail, results] = await Promise.all([
      getBenchmarkSuite(suiteId),
      getBenchmarkSuiteResults(suiteId),
    ]);
    setSuiteDetail(detail);
    setSuiteResults(results);
  };

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        await fetchBaseData();
        if (!cancelled) setError('');
      } catch (err: any) {
        if (!cancelled) setError(err?.message || 'Failed to load benchmark data');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void tick();
    const interval = setInterval(() => {
      void tick();
    }, 15000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    const refresh = async () => {
      if (!selectedSuiteId) {
        setSuiteDetail(null);
        setSuiteResults(null);
        return;
      }
      try {
        await refreshSelectedSuite(selectedSuiteId);
      } catch (err: any) {
        if (!cancelled) setError(err?.message || 'Failed to load benchmark suite');
      }
    };

    void refresh();
    const interval = setInterval(() => {
      if (!selectedSuiteId) return;
      void refresh();
    }, 4000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [selectedSuiteId]);

  const scatterOption: EChartsOption = useMemo(() => {
    const points: BenchmarkPoint[] = suiteResults?.points || [];
    return {
      backgroundColor: '#f8fafc',
      animationDuration: 600,
      grid: { left: 48, right: 24, top: 30, bottom: 42, containLabel: true },
      tooltip: {
        trigger: 'item',
        backgroundColor: 'rgba(255,255,255,0.97)',
        borderColor: 'rgba(99,102,241,0.4)',
        textStyle: { color: '#1e293b' },
        formatter: (params: any) => {
          const point = params?.data?.meta as BenchmarkPoint | undefined;
          if (!point) return 'No data';
          return [
            `<strong>${point.label}</strong>`,
            `Duration: ${formatSeconds(point.x_duration_seconds)}`,
            `Composite Accuracy: ${safePercent(point.y_composite_accuracy_pct)}`,
            `Classification: ${safePercent((point.classification_accuracy || 0) * 100)}`,
            `OCR: ${safePercent((point.ocr_accuracy || 0) * 100)}`,
          ].join('<br/>');
        },
      },
      xAxis: {
        name: 'Duration (seconds)',
        nameLocation: 'middle',
        nameGap: 28,
        type: 'value',
        axisLabel: { color: '#6b7280' },
        axisLine: { lineStyle: { color: '#e5e7eb' } },
        splitLine: { lineStyle: { color: 'rgba(107,114,128,0.15)' } },
      },
      yAxis: {
        name: 'Composite Accuracy %',
        nameLocation: 'middle',
        nameGap: 42,
        type: 'value',
        min: 0,
        max: 100,
        axisLabel: { color: '#6b7280' },
        axisLine: { lineStyle: { color: '#e5e7eb' } },
        splitLine: { lineStyle: { color: 'rgba(107,114,128,0.15)' } },
      },
      series: [
        {
          type: 'scatter',
          symbolSize: 10,
          itemStyle: {
            color: '#6366f1',
            borderColor: '#a5b4fc',
            borderWidth: 1,
            shadowBlur: 8,
            shadowColor: 'rgba(99, 102, 241, 0.35)',
          },
          data: points.map((point) => ({
            value: [point.x_duration_seconds, point.y_composite_accuracy_pct],
            meta: point,
          })),
        },
      ],
    };
  }, [suiteResults]);

  const createTruthDisabled = running || !truthName.trim() || !truthVideoUrl.trim();
  const runDisabled = running || !runTruthId;

  const handleCreateTruth = async () => {
    setRunning(true);
    try {
      await createBenchmarkTruth({
        name: truthName.trim(),
        video_url: truthVideoUrl.trim(),
        expected_ocr_text: truthExpectedOcr,
        expected_categories: truthCategories
          .split(',')
          .map((value) => value.trim())
          .filter(Boolean),
        expected_brand: truthExpectedBrand.trim(),
        expected_category: truthExpectedCategory.trim(),
        expected_confidence: truthExpectedConfidence.trim() ? Number(truthExpectedConfidence) : null,
        expected_reasoning: truthExpectedReasoning,
      });
      const truthPayload = await getBenchmarkTruths();
      setTruths(truthPayload.truths || []);
      if (truthPayload.truths?.length && !runTruthId) {
        setRunTruthId(truthPayload.truths[0].truth_id);
      }
      setError('');
    } catch (err: any) {
      setError(err?.message || 'Failed to create benchmark truth');
    } finally {
      setRunning(false);
    }
  };

  const handleRunSuite = async () => {
    setRunning(true);
    try {
      const result = await runBenchmarkSuite({
        truth_id: runTruthId,
        categories: runCategories,
      });
      const suiteId = String(result?.suite_id || '');
      if (suiteId) {
        setSelectedSuiteId(suiteId);
      }
      const suitePayload = await getBenchmarkSuites();
      setSuites(suitePayload.suites || []);
      setError('');
    } catch (err: any) {
      setError(err?.message || 'Failed to start benchmark suite');
    } finally {
      setRunning(false);
    }
  };

  const handleOpenSuiteModal = (suite: BenchmarkSuiteSummary) => {
    setSuiteModalId(suite.suite_id);
    setSuiteModalName((suite.name || '').trim() || `Suite ${suite.suite_id}`);
    setSuiteModalDescription(suite.description || '');
    setSuiteModalOpen(true);
  };

  const handleSaveSuite = async () => {
    if (!suiteModalId) return;
    setRunning(true);
    try {
      await updateBenchmarkSuite(suiteModalId, {
        name: suiteModalName.trim(),
        description: suiteModalDescription.trim(),
      });
      await fetchBaseData();
      if (selectedSuiteId === suiteModalId) {
        await refreshSelectedSuite(suiteModalId);
      }
      setSuiteModalOpen(false);
      setError('');
    } catch (err: any) {
      setError(err?.message || 'Failed to update suite');
    } finally {
      setRunning(false);
    }
  };

  const handleDeleteSuite = async (suite: BenchmarkSuiteSummary) => {
    const confirmed = window.confirm(`Delete suite ${suite.suite_id}? This will delete its benchmark tests and jobs.`);
    if (!confirmed) return;
    setRunning(true);
    try {
      await deleteBenchmarkSuite(suite.suite_id);
      const payload = await getBenchmarkSuites();
      const nextSuites = payload.suites || [];
      setSuites(nextSuites);
      if (selectedSuiteId === suite.suite_id) {
        setSelectedSuiteId(nextSuites[0]?.suite_id || '');
      }
      setError('');
    } catch (err: any) {
      setError(err?.message || 'Failed to delete suite');
    } finally {
      setRunning(false);
    }
  };

  const handleOpenTestModal = (test: BenchmarkTruth) => {
    setTestModalId(test.test_id || test.truth_id);
    setTestModalSourceUrl(test.source_url || test.video_url || '');
    setTestModalExpectedCategory(test.expected_category || test.expected_categories?.[0] || '');
    setTestModalExpectedBrand(test.expected_brand || '');
    setTestModalExpectedConfidence(
      test.expected_confidence == null || Number.isNaN(test.expected_confidence)
        ? ''
        : String(test.expected_confidence),
    );
    setTestModalExpectedReasoning(test.expected_reasoning || '');
    setTestModalExpectedOcr(test.expected_ocr_text || '');
    setTestModalOpen(true);
  };

  const handleSaveTest = async () => {
    if (!testModalId) return;
    setRunning(true);
    try {
      await updateBenchmarkTest(testModalId, {
        source_url: testModalSourceUrl.trim(),
        expected_category: testModalExpectedCategory.trim(),
        expected_brand: testModalExpectedBrand.trim(),
        expected_confidence: testModalExpectedConfidence.trim() ? Number(testModalExpectedConfidence) : null,
        expected_reasoning: testModalExpectedReasoning,
        expected_ocr_text: testModalExpectedOcr,
      });
      if (selectedSuiteId) {
        await refreshSelectedSuite(selectedSuiteId);
      }
      setTestModalOpen(false);
      setError('');
    } catch (err: any) {
      setError(err?.message || 'Failed to update benchmark test');
    } finally {
      setRunning(false);
    }
  };

  const handleDeleteTest = async (test: BenchmarkTruth) => {
    const id = test.test_id || test.truth_id;
    const confirmed = window.confirm(`Delete test ${id}?`);
    if (!confirmed) return;
    setRunning(true);
    try {
      await deleteBenchmarkTest(id);
      if (selectedSuiteId) {
        await refreshSelectedSuite(selectedSuiteId);
      }
      setError('');
    } catch (err: any) {
      setError(err?.message || 'Failed to delete benchmark test');
    } finally {
      setRunning(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 p-8 text-gray-500 animate-pulse">
        <UpdateIcon className="animate-spin" /> Loading benchmark tools…
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="flex items-center gap-2">
        <RocketIcon className="h-6 w-6 text-primary-600" />
        <h2 className="text-3xl font-bold tracking-tight text-gray-900">Benchmarking</h2>
      </div>

      {error && (
        <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
          <ExclamationTriangleIcon className="h-4 w-4" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      <div className="rounded-xl border border-gray-200 bg-white p-4">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500">Host Hardware Profile</h3>
          <span className="text-[11px] text-gray-400">Detected at {profile?.timestamp || '—'}</span>
        </div>
        <div className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
          <div className="rounded border border-gray-200 bg-gray-50 p-2 text-gray-700">CPU (logical): {profile?.hardware.cpu_count_logical ?? '—'}</div>
          <div className="rounded border border-gray-200 bg-gray-50 p-2 text-gray-700">RAM: {formatGbFromMb(profile?.hardware.total_ram_mb)}</div>
          <div className="rounded border border-gray-200 bg-gray-50 p-2 text-gray-700">Accelerator: {profile?.hardware.accelerator || 'cpu'}</div>
          <div className="rounded border border-gray-200 bg-gray-50 p-2 text-gray-700">VRAM: {profile?.hardware.total_vram_mb ?? '—'} MB</div>
        </div>
        {(profile?.warnings || []).length > 0 && (
          <div className="mt-3 space-y-2">
            {profile!.warnings.map((warning, idx) => (
              <div
                key={`${warning.model}-${idx}`}
                className="rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-700"
              >
                {warning.message}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="space-y-3 rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
          <h3 className="text-sm font-semibold text-gray-900">Create Golden Video Truth</h3>
          <input
            value={truthName}
            onChange={(event) => setTruthName(event.target.value)}
            placeholder="Truth set name"
            className="h-10 w-full rounded border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          />
          <input
            value={truthVideoUrl}
            onChange={(event) => setTruthVideoUrl(event.target.value)}
            placeholder="Video URL or absolute server path"
            className="h-10 w-full rounded border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          />
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            <input
              value={truthExpectedBrand}
              onChange={(event) => setTruthExpectedBrand(event.target.value)}
              placeholder="Expected brand"
              className="h-10 w-full rounded border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
            />
            <input
              value={truthExpectedCategory}
              onChange={(event) => setTruthExpectedCategory(event.target.value)}
              placeholder="Expected category"
              className="h-10 w-full rounded border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
            />
          </div>
          <input
            value={truthExpectedConfidence}
            onChange={(event) => setTruthExpectedConfidence(event.target.value)}
            placeholder="Expected confidence (0..1)"
            className="h-10 w-full rounded border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          />
          <textarea
            value={truthCategories}
            onChange={(event) => setTruthCategories(event.target.value)}
            placeholder="Expected categories (comma separated)"
            className="h-20 w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          />
          <textarea
            value={truthExpectedOcr}
            onChange={(event) => setTruthExpectedOcr(event.target.value)}
            placeholder="Expected OCR corpus"
            className="h-20 w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          />
          <textarea
            value={truthExpectedReasoning}
            onChange={(event) => setTruthExpectedReasoning(event.target.value)}
            placeholder="Expected reasoning"
            className="h-20 w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          />
          <button
            type="button"
            disabled={createTruthDisabled}
            onClick={handleCreateTruth}
            className="h-10 rounded bg-primary-600 px-4 text-sm font-semibold text-white disabled:opacity-50"
          >
            Create Truth
          </button>
        </div>

        <div className="space-y-3 rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
          <h3 className="text-sm font-semibold text-gray-900">Run Benchmark Suite</h3>
          <select
            value={runTruthId}
            onChange={(event) => setRunTruthId(event.target.value)}
            className="h-10 w-full rounded border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          >
            <option value="">Select Golden Truth</option>
            {truths.map((truth) => (
              <option key={truth.truth_id} value={truth.truth_id}>
                {truth.name}
              </option>
            ))}
          </select>
          <input
            value={runCategories}
            onChange={(event) => setRunCategories(event.target.value)}
            placeholder="Optional categories override"
            className="h-10 w-full rounded border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          />
          <button
            type="button"
            disabled={runDisabled}
            onClick={handleRunSuite}
            className="h-10 rounded bg-emerald-600 px-4 text-sm font-semibold text-white disabled:opacity-50"
          >
            Launch Cartesian Benchmark
          </button>
          <div className="text-xs text-gray-500">
            Benchmarks enqueue permutations across scan strategy, OCR engine/mode, and provider-model combinations.
          </div>
        </div>
      </div>

      <div className="space-y-3 rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-900">Benchmark Suites</h3>
          <select
            value={selectedSuiteId}
            onChange={(event) => setSelectedSuiteId(event.target.value)}
            className="h-9 rounded border border-gray-300 bg-white px-2 text-xs text-gray-900 focus:border-primary-500 focus:outline-none"
          >
            <option value="">Select suite</option>
            {suites.map((suite) => (
              <option key={suite.suite_id} value={suite.suite_id}>
                {suite.name || suite.suite_id}
              </option>
            ))}
          </select>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[980px] text-left text-xs text-gray-700">
            <thead className="text-gray-500">
              <tr className="border-b border-gray-200">
                <th className="px-2 py-2">Suite</th>
                <th className="px-2 py-2">Description</th>
                <th className="px-2 py-2">Status</th>
                <th className="px-2 py-2">Tests</th>
                <th className="px-2 py-2">Jobs</th>
                <th className="px-2 py-2">Updated</th>
                <th className="px-2 py-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {suites.map((suite) => {
                const active = suite.suite_id === selectedSuiteId;
                return (
                  <tr
                    key={suite.suite_id}
                    className={`border-t border-gray-100 ${active ? 'bg-primary-50' : 'bg-transparent hover:bg-gray-50'}`}
                  >
                    <td className="px-2 py-2">
                      <button
                        type="button"
                        onClick={() => setSelectedSuiteId(suite.suite_id)}
                        className="text-left text-primary-600 hover:underline font-medium"
                      >
                        {suite.name || suite.suite_id}
                      </button>
                    </td>
                    <td className="px-2 py-2 text-gray-500" title={suite.description || ''}>
                      {truncate(suite.description, 70)}
                    </td>
                    <td className="px-2 py-2">{suite.status}</td>
                    <td className="px-2 py-2">{suite.test_count ?? 0}</td>
                    <td className="px-2 py-2">{suite.completed_jobs}/{suite.total_jobs}</td>
                    <td className="px-2 py-2 text-gray-400">{truncate(suite.updated_at, 22)}</td>
                    <td className="px-2 py-2">
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => handleOpenSuiteModal(suite)}
                          className="inline-flex items-center gap-1 rounded border border-gray-300 px-2 py-1 text-[11px] text-gray-700 hover:bg-gray-100"
                        >
                          <Pencil2Icon /> Edit
                        </button>
                        <button
                          type="button"
                          onClick={() => void handleDeleteSuite(suite)}
                          className="inline-flex items-center gap-1 rounded border border-red-200 px-2 py-1 text-[11px] text-red-600 hover:bg-red-50"
                        >
                          <TrashIcon /> Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
              {suites.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-2 py-3 text-gray-400">No benchmark suites available.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="space-y-3 rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-900">Suite Detail</h3>
        {suiteDetail ? (
          <>
            <div className="grid grid-cols-2 gap-3 text-sm text-gray-700 md:grid-cols-4">
              <div className="rounded border border-gray-200 bg-gray-50 p-2">Status: {suiteDetail.status}</div>
              <div className="rounded border border-gray-200 bg-gray-50 p-2">Total: {suiteDetail.total_jobs}</div>
              <div className="rounded border border-gray-200 bg-gray-50 p-2">Completed: {suiteDetail.completed_jobs}</div>
              <div className="rounded border border-gray-200 bg-gray-50 p-2">Failed: {suiteDetail.failed_jobs}</div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[1000px] text-left text-xs text-gray-700">
                <thead className="text-gray-500">
                  <tr className="border-b border-gray-200">
                    <th className="px-2 py-2">Test ID</th>
                    <th className="px-2 py-2">Source URL</th>
                    <th className="px-2 py-2">Expected Category</th>
                    <th className="px-2 py-2">Expected Brand</th>
                    <th className="px-2 py-2">Expected Confidence</th>
                    <th className="px-2 py-2">Expected Reasoning</th>
                    <th className="px-2 py-2">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {(suiteDetail.tests || []).map((test) => {
                    const id = test.test_id || test.truth_id;
                    return (
                      <tr key={id} className="border-t border-gray-100 hover:bg-gray-50">
                        <td className="px-2 py-2">{id}</td>
                        <td className="px-2 py-2 text-gray-500" title={test.source_url || test.video_url}>
                          {truncate(test.source_url || test.video_url, 54)}
                        </td>
                        <td className="px-2 py-2" title={test.expected_category || test.expected_categories?.join(', ')}>
                          {truncate(test.expected_category || test.expected_categories?.[0], 48)}
                        </td>
                        <td className="px-2 py-2">{truncate(test.expected_brand, 36)}</td>
                        <td className="px-2 py-2">{test.expected_confidence ?? '—'}</td>
                        <td className="px-2 py-2" title={test.expected_reasoning || ''}>
                          {truncate(test.expected_reasoning, 72)}
                        </td>
                        <td className="px-2 py-2">
                          <div className="flex items-center gap-2">
                            <button
                              type="button"
                              onClick={() => handleOpenTestModal(test)}
                              className="inline-flex items-center gap-1 rounded border border-gray-300 px-2 py-1 text-[11px] text-gray-700 hover:bg-gray-100"
                            >
                              <Pencil2Icon /> Edit
                            </button>
                            <button
                              type="button"
                              onClick={() => void handleDeleteTest(test)}
                              className="inline-flex items-center gap-1 rounded border border-red-200 px-2 py-1 text-[11px] text-red-600 hover:bg-red-50"
                            >
                              <TrashIcon /> Delete
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                  {(suiteDetail.tests || []).length === 0 && (
                    <tr>
                      <td colSpan={7} className="px-2 py-3 text-gray-400">No benchmark tests assigned to this suite.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="text-sm text-gray-400">Select a suite to inspect its golden tests.</div>
        )}
      </div>

      <div className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm">
        <div className="mb-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500">
            Duration vs Composite Accuracy
          </h3>
          <p className="text-xs text-gray-400">
            Each dot is a benchmark permutation. Lower X and higher Y is better (Pareto frontier).
          </p>
        </div>
        <ReactECharts option={scatterOption} style={{ height: 460, width: '100%' }} notMerge lazyUpdate />
      </div>

      <ModalShell title="Edit Benchmark Suite" open={suiteModalOpen} onClose={() => setSuiteModalOpen(false)}>
        <div className="space-y-3">
          <input
            value={suiteModalName}
            onChange={(event) => setSuiteModalName(event.target.value)}
            placeholder="Suite title"
            className="h-10 w-full rounded border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          />
          <textarea
            value={suiteModalDescription}
            onChange={(event) => setSuiteModalDescription(event.target.value)}
            placeholder="Suite description"
            className="h-24 w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          />
          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => void handleSaveSuite()}
              className="h-9 rounded bg-primary-600 px-4 text-sm font-semibold text-white disabled:opacity-50"
            >
              Save Suite
            </button>
          </div>
        </div>
      </ModalShell>

      <ModalShell title="Edit Benchmark Test" open={testModalOpen} onClose={() => setTestModalOpen(false)}>
        <div className="space-y-3">
          <input
            value={testModalSourceUrl}
            onChange={(event) => setTestModalSourceUrl(event.target.value)}
            placeholder="Source URL"
            className="h-10 w-full rounded border border-slate-700 bg-slate-900 px-3 text-sm text-slate-100"
          />
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            <input
              value={testModalExpectedCategory}
              onChange={(event) => setTestModalExpectedCategory(event.target.value)}
              placeholder="Expected category"
              className="h-10 w-full rounded border border-slate-700 bg-slate-900 px-3 text-sm text-slate-100"
            />
            <input
              value={testModalExpectedBrand}
              onChange={(event) => setTestModalExpectedBrand(event.target.value)}
              placeholder="Expected brand"
              className="h-10 w-full rounded border border-slate-700 bg-slate-900 px-3 text-sm text-slate-100"
            />
          </div>
          <input
            value={testModalExpectedConfidence}
            onChange={(event) => setTestModalExpectedConfidence(event.target.value)}
            placeholder="Expected confidence (0..1)"
            className="h-10 w-full rounded border border-slate-700 bg-slate-900 px-3 text-sm text-slate-100"
          />
          <textarea
            value={testModalExpectedReasoning}
            onChange={(event) => setTestModalExpectedReasoning(event.target.value)}
            placeholder="Expected reasoning"
            className="h-24 w-full rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100"
          />
          <textarea
            value={testModalExpectedOcr}
            onChange={(event) => setTestModalExpectedOcr(event.target.value)}
            placeholder="Expected OCR corpus"
            className="h-24 w-full rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100"
          />
          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => void handleSaveTest()}
              className="h-9 rounded bg-cyan-600 px-4 text-sm font-semibold text-white disabled:opacity-50"
            >
              Save Test
            </button>
          </div>
        </div>
      </ModalShell>
    </div>
  );
}
