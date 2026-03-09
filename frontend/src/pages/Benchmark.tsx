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
  deleteBenchmarkTruth,
  getBenchmarkModels,
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
  BenchmarkPathCount,
  BenchmarkSuiteDetail,
  BenchmarkSuiteResults,
  BenchmarkSuiteSummary,
  BenchmarkTruth,
  ModelCombo,
  SystemProfile,
} from '../lib/api';

const panelClass =
  'rounded-[30px] border border-slate-200/80 bg-white/82 shadow-[0_18px_45px_rgba(15,23,42,0.06)] backdrop-blur';
const controlClass =
  'h-10 w-full rounded-2xl border border-slate-200 bg-white/90 px-3 text-sm text-slate-700 shadow-sm transition-colors focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15';

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

// ── Medal helper ────────────────────────────────────────────────────────────
function medalFor(rank: number): string {
  if (rank === 1) return '🥇';
  if (rank === 2) return '🥈';
  if (rank === 3) return '🥉';
  return `#${rank}`;
}

// ── Mini bar component ───────────────────────────────────────────────────────
function MiniBar({ pct, color }: { pct: number; color: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="h-2 w-24 rounded-full bg-gray-100 overflow-hidden flex-shrink-0">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.min(100, Math.max(0, pct))}%`, background: color }}
        />
      </div>
      <span className="text-[11px] tabular-nums text-gray-600">{pct.toFixed(1)}%</span>
    </div>
  );
}

// ── Performance ranking table ─────────────────────────────────────────────
function PerformanceTable({ points }: { points: BenchmarkPoint[] }) {
  const sorted = useMemo(
    () =>
      [...points]
        .filter((p) => Number.isFinite(p.y_composite_accuracy_pct))
        .sort((a, b) => b.y_composite_accuracy_pct - a.y_composite_accuracy_pct),
    [points],
  );

  if (sorted.length === 0) {
    return (
      <div className="text-sm text-gray-400 py-4 text-center">
        No completed benchmark points yet.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[700px] text-left text-xs">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="px-3 py-2 text-gray-400 font-medium w-12">Rank</th>
            <th className="px-3 py-2 text-gray-400 font-medium">Model / Config</th>
            <th className="px-3 py-2 text-gray-400 font-medium w-24">Duration</th>
            <th className="px-3 py-2 text-gray-400 font-medium">Composite</th>
            <th className="px-3 py-2 text-gray-400 font-medium">Classification</th>
            <th className="px-3 py-2 text-gray-400 font-medium">OCR</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((point, i) => {
            const rank = i + 1;
            const isTop3 = rank <= 3;
            const rowBg =
              rank === 1
                ? 'bg-amber-50'
                : rank === 2
                  ? 'bg-gray-50/80'
                  : rank === 3
                    ? 'bg-orange-50/50'
                    : 'bg-white';
            return (
              <tr
                key={point.job_id}
                className={`border-t border-gray-100 transition-colors hover:bg-indigo-50/30 ${rowBg}`}
              >
                <td className="px-3 py-2.5 font-semibold text-center">
                  <span className={`text-base ${isTop3 ? '' : 'text-gray-400 text-xs'}`}>
                    {medalFor(rank)}
                  </span>
                </td>
                <td className="px-3 py-2.5 max-w-[280px]">
                  <span
                    className="block truncate font-medium text-gray-800"
                    title={point.label}
                  >
                    {point.label}
                  </span>
                </td>
                <td className="px-3 py-2.5 tabular-nums text-gray-500">
                  {formatSeconds(point.x_duration_seconds)}
                </td>
                <td className="px-3 py-2.5">
                  <MiniBar pct={point.y_composite_accuracy_pct} color="#6366f1" />
                </td>
                <td className="px-3 py-2.5">
                  <MiniBar pct={(point.classification_accuracy || 0) * 100} color="#10b981" />
                </td>
                <td className="px-3 py-2.5">
                  <MiniBar pct={(point.ocr_accuracy || 0) * 100} color="#f59e0b" />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PathMetricList({
  title,
  subtitle,
  items,
  denominator,
  accent,
}: {
  title: string;
  subtitle: string;
  items: BenchmarkPathCount[];
  denominator: number;
  accent: string;
}) {
  if (!items.length) {
    return (
      <div className="rounded-[24px] border border-slate-200/80 bg-slate-50/80 p-4">
        <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">{title}</div>
        <p className="mt-2 text-xs leading-5 text-slate-500">{subtitle}</p>
        <div className="mt-4 text-sm text-slate-400">No saved trace data yet.</div>
      </div>
    );
  }

  return (
    <div className="rounded-[24px] border border-slate-200/80 bg-slate-50/80 p-4">
      <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">{title}</div>
      <p className="mt-2 text-xs leading-5 text-slate-500">{subtitle}</p>
      <div className="mt-4 space-y-3">
        {items.map((item) => {
          const pct = denominator > 0 ? (item.count / denominator) * 100 : 0;
          return (
            <div key={`${title}-${item.attempt_type}`} className="space-y-1.5">
              <div className="flex items-center justify-between gap-3">
                <span className="truncate text-sm font-medium text-slate-700" title={item.title}>
                  {item.title}
                </span>
                <span className="shrink-0 text-xs font-semibold text-slate-500">
                  {item.count} job{item.count === 1 ? '' : 's'}
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-white">
                <div
                  className="h-full rounded-full transition-all"
                  style={{ width: `${Math.min(100, Math.max(0, pct))}%`, background: accent }}
                />
              </div>
              <div className="text-[11px] uppercase tracking-[0.18em] text-slate-400">
                {pct.toFixed(1)}% of traced jobs
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
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

  // Model multi-select
  const [availableModels, setAvailableModels] = useState<ModelCombo[]>([]);
  const [selectedModelKeys, setSelectedModelKeys] = useState<Set<string>>(new Set());

  // Manual model entry (for llama-server etc.)
  const [manualProvider, setManualProvider] = useState<string>('Llama Server');
  const [manualModelName, setManualModelName] = useState<string>('');

  // Express mode
  const [expressMode, setExpressMode] = useState(false);

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

  // Load available models for selector
  useEffect(() => {
    getBenchmarkModels().then((combos) => {
      setAvailableModels(combos);
      setSelectedModelKeys(new Set(combos.map((c) => c.key)));
    });
  }, []);

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
  const tracedJobCount = suiteResults?.path_metrics?.jobs_with_trace || 0;

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
      // Build explicit model_combos from the selected chips
      const selectedCombos = availableModels
        .filter((m) => selectedModelKeys.has(m.key))
        .map((m) => ({ provider: m.provider, model: m.model }));

      const result = await runBenchmarkSuite({
        truth_id: runTruthId,
        categories: runCategories,
        express_mode: expressMode,
        ...(selectedCombos.length > 0 ? { model_combos: selectedCombos } : {}),
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

  // Feature 1: delete truth from Run Suite panel
  const handleDeleteTruth = async (truth: BenchmarkTruth) => {
    const confirmed = window.confirm(`Delete golden truth "${truth.name || truth.truth_id}"?`);
    if (!confirmed) return;
    setRunning(true);
    try {
      await deleteBenchmarkTruth(truth.truth_id);
      const payload = await getBenchmarkTruths();
      const next = payload.truths || [];
      setTruths(next);
      if (runTruthId === truth.truth_id) {
        setRunTruthId(next[0]?.truth_id || '');
      }
      setError('');
    } catch (err: any) {
      setError(err?.message || 'Failed to delete truth');
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

  // Model chip toggle
  const toggleModel = (key: string) => {
    setSelectedModelKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const addManualModel = () => {
    const name = manualModelName.trim();
    if (!name) return;
    const key = `${manualProvider}::${name}`;
    if (availableModels.some((m) => m.key === key)) {
      // Already exists — just make sure it's selected
      setSelectedModelKeys((prev) => new Set([...prev, key]));
    } else {
      const combo: ModelCombo = { provider: manualProvider, model: name, key };
      setAvailableModels((prev) => [...prev, combo]);
      setSelectedModelKeys((prev) => new Set([...prev, key]));
    }
    setManualModelName('');
  };

  const removeModel = (key: string) => {
    setAvailableModels((prev) => prev.filter((m) => m.key !== key));
    setSelectedModelKeys((prev) => { const next = new Set(prev); next.delete(key); return next; });
  };

  const selectAllModels = () => setSelectedModelKeys(new Set(availableModels.map((m) => m.key)));
  const clearAllModels = () => setSelectedModelKeys(new Set());

  if (loading) {
    return (
      <div className={`${panelClass} flex items-center gap-3 p-8 text-slate-500 animate-pulse`}>
        <UpdateIcon className="animate-spin" /> Loading benchmark tools…
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className={`${panelClass} overflow-hidden p-6`}>
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px] lg:items-end">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-primary-200 bg-primary-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.24em] text-primary-700">
              <RocketIcon className="h-3.5 w-3.5" />
              Controlled trials
            </div>
            <h2 className="mt-4 text-3xl font-black tracking-[-0.05em] text-slate-950">Benchmark lab</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
              Run repeatable suites across OCR, model hosts, and runtime modes. This screen is tuned for comparative readouts rather than day-to-day queue operations.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-[22px] border border-slate-200/80 bg-slate-50/80 px-4 py-4 shadow-sm">
              <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">Truth sets</div>
              <div className="mt-2 text-lg font-black tracking-[-0.04em] text-slate-950">{truths.length}</div>
              <div className="text-xs text-slate-500">Golden references loaded</div>
            </div>
            <div className="rounded-[22px] border border-slate-200/80 bg-slate-950 px-4 py-4 text-slate-50 shadow-[0_18px_36px_rgba(15,23,42,0.18)]">
              <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">Available models</div>
              <div className="mt-2 text-lg font-black tracking-[-0.04em] text-white">{availableModels.length}</div>
              <div className="text-xs text-slate-400">Selectable benchmark contenders</div>
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-3 rounded-[22px] border border-red-200 bg-red-50/90 p-4 text-red-700 shadow-sm">
          <ExclamationTriangleIcon className="h-4 w-4" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      <div className={`${panelClass} p-5`}>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-[0.22em] text-slate-500">Host hardware profile</h3>
          <span className="text-[11px] text-slate-400">Detected at {profile?.timestamp || '—'}</span>
        </div>
        <div className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
          <div className="rounded-[22px] border border-slate-200 bg-slate-50/80 p-3 text-slate-700">CPU (logical): {profile?.hardware.cpu_count_logical ?? '—'}</div>
          <div className="rounded-[22px] border border-slate-200 bg-slate-50/80 p-3 text-slate-700">RAM: {formatGbFromMb(profile?.hardware.total_ram_mb)}</div>
          <div className="rounded-[22px] border border-slate-200 bg-slate-50/80 p-3 text-slate-700">Accelerator: {profile?.hardware.accelerator || 'cpu'}</div>
          <div className="rounded-[22px] border border-slate-200 bg-slate-50/80 p-3 text-slate-700">VRAM: {profile?.hardware.total_vram_mb ?? '—'} MB</div>
        </div>
        {(profile?.warnings || []).length > 0 && (
          <div className="mt-3 space-y-2">
            {profile!.warnings.map((warning, idx) => (
              <div
                key={`${warning.model}-${idx}`}
                className="rounded-[18px] border border-amber-200 bg-amber-50 p-3 text-xs text-amber-700"
              >
                {warning.message}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* Create Golden Video Truth */}
        <div className={`${panelClass} space-y-3 p-5`}>
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">Golden input</div>
            <h3 className="mt-2 text-xl font-black tracking-[-0.04em] text-slate-950">Create truth reference</h3>
            <p className="mt-1 text-sm text-slate-500">Define the expected brand, category, OCR, and reasoning for one benchmark asset.</p>
          </div>
          <input
            value={truthName}
            onChange={(event) => setTruthName(event.target.value)}
            placeholder="Truth set name"
            className={controlClass}
          />
          <input
            value={truthVideoUrl}
            onChange={(event) => setTruthVideoUrl(event.target.value)}
            placeholder="Video URL or absolute server path"
            className={controlClass}
          />
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            <input
              value={truthExpectedBrand}
              onChange={(event) => setTruthExpectedBrand(event.target.value)}
              placeholder="Expected brand"
              className={controlClass}
            />
            <input
              value={truthExpectedCategory}
              onChange={(event) => setTruthExpectedCategory(event.target.value)}
              placeholder="Expected category"
              className={controlClass}
            />
          </div>
          <input
            value={truthExpectedConfidence}
            onChange={(event) => setTruthExpectedConfidence(event.target.value)}
            placeholder="Expected confidence (0..1)"
            className={controlClass}
          />
          <textarea
            value={truthCategories}
            onChange={(event) => setTruthCategories(event.target.value)}
            placeholder="Expected categories (comma separated)"
            className="h-20 w-full rounded-2xl border border-slate-200 bg-white/90 px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15"
          />
          <textarea
            value={truthExpectedOcr}
            onChange={(event) => setTruthExpectedOcr(event.target.value)}
            placeholder="Expected OCR corpus"
            className="h-20 w-full rounded-2xl border border-slate-200 bg-white/90 px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15"
          />
          <textarea
            value={truthExpectedReasoning}
            onChange={(event) => setTruthExpectedReasoning(event.target.value)}
            placeholder="Expected reasoning"
            className="h-20 w-full rounded-2xl border border-slate-200 bg-white/90 px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors focus:border-primary-400 focus:ring-2 focus:ring-primary-500/15"
          />
          <button
            type="button"
            disabled={createTruthDisabled}
            onClick={handleCreateTruth}
            className="inline-flex h-11 items-center justify-center rounded-2xl bg-[linear-gradient(135deg,#4f46e5,#2563eb)] px-4 text-sm font-bold uppercase tracking-[0.18em] text-white shadow-[0_18px_36px_rgba(79,70,229,0.22)] transition-transform duration-200 hover:-translate-y-0.5 disabled:opacity-50"
          >
            Create Truth
          </button>
        </div>

        {/* Run Benchmark Suite */}
        <div className={`${panelClass} space-y-3 p-5`}>
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">Execution</div>
            <h3 className="mt-2 text-xl font-black tracking-[-0.04em] text-slate-950">Run comparison suite</h3>
            <p className="mt-1 text-sm text-slate-500">Select the truth set, the contender models, and the execution profile you want to compare.</p>
          </div>

          {/* ── Feature 1: Deletable truth list ────────────────────────────── */}
          <div>
            <p className="mb-1.5 text-xs font-medium uppercase tracking-[0.18em] text-slate-400">Select golden truth</p>
            {truths.length === 0 ? (
              <div className="rounded-[18px] border border-dashed border-slate-300 py-3 text-center text-xs text-slate-400">
                No golden truths yet — create one on the left.
              </div>
            ) : (
              <div className="space-y-1 max-h-48 overflow-y-auto rounded-[20px] border border-slate-200 bg-slate-50/60 p-1.5">
                {truths.map((truth) => {
                  const isSelected = runTruthId === truth.truth_id;
                  return (
                    <div
                      key={truth.truth_id}
                      className={`flex items-center justify-between px-3 py-2 cursor-pointer transition-colors ${
                        isSelected
                          ? 'bg-primary-50 border-l-2 border-primary-500'
                          : 'hover:bg-gray-50 border-l-2 border-transparent'
                      }`}
                      onClick={() => setRunTruthId(truth.truth_id)}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <span
                          className={`h-2 w-2 rounded-full flex-shrink-0 ${
                            isSelected ? 'bg-primary-500' : 'bg-gray-300'
                          }`}
                        />
                        <span
                          className={`text-sm truncate ${
                            isSelected ? 'font-semibold text-primary-700' : 'text-gray-700'
                          }`}
                          title={truth.name}
                        >
                          {truth.name || truth.truth_id}
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          void handleDeleteTruth(truth);
                        }}
                        disabled={running}
                        className="ml-2 flex-shrink-0 rounded p-1 text-gray-400 hover:bg-red-50 hover:text-red-500 transition-colors disabled:opacity-40"
                        title="Delete this truth"
                      >
                        <TrashIcon className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* ── Feature 2: Model multi-select chips ────────────────────────── */}
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <p className="text-xs font-medium text-gray-500">Competing Models</p>
              {availableModels.length > 0 && (
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={selectAllModels}
                    className="text-[11px] text-primary-600 hover:underline"
                  >
                    All
                  </button>
                  <span className="text-gray-300">|</span>
                  <button
                    type="button"
                    onClick={clearAllModels}
                    className="text-[11px] text-gray-400 hover:underline"
                  >
                    None
                  </button>
                </div>
              )}
            </div>
            {availableModels.length === 0 ? (
              <div className="rounded border border-dashed border-gray-300 py-3 text-center text-xs text-gray-400">
                No models detected — add manually below or all available will be used.
              </div>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {availableModels.map((combo) => {
                  const active = selectedModelKeys.has(combo.key);
                  return (
                    <button
                      key={combo.key}
                      type="button"
                      onClick={() => toggleModel(combo.key)}
                      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[11px] font-medium transition-all ${
                        active
                          ? 'border-primary-400 bg-primary-50 text-primary-700 shadow-sm'
                          : 'border-gray-300 bg-white text-gray-400 hover:border-gray-400'
                      }`}
                    >
                      <span
                        className={`h-1.5 w-1.5 rounded-full ${active ? 'bg-primary-500' : 'bg-gray-300'}`}
                      />
                      <span className="font-normal text-gray-500">{combo.provider}</span>
                      <span className="font-semibold">{combo.model}</span>
                      <span
                        role="button"
                        tabIndex={0}
                        onClick={(e) => { e.stopPropagation(); removeModel(combo.key); }}
                        onKeyDown={(e) => { if (e.key === 'Enter') { e.stopPropagation(); removeModel(combo.key); } }}
                        className="ml-0.5 text-gray-400 hover:text-red-500"
                        title="Remove"
                      >
                        ×
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
            {/* Manual model entry row */}
            <div className="mt-2 flex items-center gap-1.5">
              <select
                value={manualProvider}
                onChange={(e) => setManualProvider(e.target.value)}
                className="h-9 rounded-xl border border-slate-200 bg-white px-2 text-xs text-slate-700 focus:border-primary-400 focus:outline-none"
              >
                <option>Llama Server</option>
                <option>Ollama</option>
                <option>OpenAI</option>
                <option>Anthropic</option>
                <option>Groq</option>
              </select>
              <input
                value={manualModelName}
                onChange={(e) => setManualModelName(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') addManualModel(); }}
                placeholder="model name (e.g. llama3.3:70b)"
                className="h-9 flex-1 rounded-xl border border-slate-200 bg-white px-2 text-xs text-slate-700 focus:border-primary-400 focus:outline-none"
              />
              <button
                type="button"
                onClick={addManualModel}
                disabled={!manualModelName.trim()}
                className="h-9 rounded-xl border border-primary-200 bg-primary-50 px-2.5 text-xs font-semibold text-primary-700 hover:bg-primary-100 disabled:opacity-40"
              >
                + Add
              </button>
            </div>
            {availableModels.length > 0 && (
              <p className="mt-1.5 text-[11px] text-gray-400">
                {selectedModelKeys.size} of {availableModels.length} selected — only these will run
              </p>
            )}
          </div>

          <input
            value={runCategories}
            onChange={(event) => setRunCategories(event.target.value)}
            placeholder="Optional categories override"
            className={controlClass}
          />

          {/* Express mode toggle */}
          <label className="flex cursor-pointer items-center gap-2.5 select-none">
            <div
              onClick={() => setExpressMode((v) => !v)}
              className={`relative h-5 w-9 rounded-full transition-colors ${
                expressMode ? 'bg-emerald-500' : 'bg-gray-300'
              }`}
            >
              <span
                className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${
                  expressMode ? 'translate-x-4' : 'translate-x-0.5'
                }`}
              />
            </div>
            <span className="text-sm text-gray-700 font-medium">
              Express mode
            </span>
            <span className="text-xs text-gray-400">(faster scan, fewer frames)</span>
          </label>

          <button
            type="button"
            disabled={runDisabled}
            onClick={handleRunSuite}
            className="inline-flex h-11 items-center justify-center rounded-2xl bg-[linear-gradient(135deg,#059669,#0f766e)] px-4 text-sm font-bold uppercase tracking-[0.18em] text-white shadow-[0_18px_36px_rgba(5,150,105,0.18)] transition-transform duration-200 hover:-translate-y-0.5 disabled:opacity-50"
          >
            Launch Benchmark
          </button>
          <div className="text-xs text-slate-500">
            Runs permutations across scan strategy, OCR engine/mode, and the selected models above.
          </div>
        </div>
      </div>

      {/* Benchmark Suites table */}
      <div className={`${panelClass} space-y-3 p-5`}>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">Results archive</div>
            <h3 className="mt-2 text-xl font-black tracking-[-0.04em] text-slate-950">Benchmark suites</h3>
          </div>
          <select
            value={selectedSuiteId}
            onChange={(event) => setSelectedSuiteId(event.target.value)}
            className="h-10 rounded-2xl border border-slate-200 bg-white px-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-700 focus:border-primary-400 focus:outline-none"
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

      {/* Suite Detail */}
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

      {suiteResults?.path_metrics && (
        <div className={`${panelClass} p-5`}>
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold uppercase tracking-[0.22em] text-slate-500">
                Path telemetry
              </h3>
              <p className="mt-1 text-sm text-slate-500">
                Derived from saved processing traces only. This does not add any work to live job execution.
              </p>
            </div>
            <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
              {tracedJobCount} traced job{tracedJobCount === 1 ? '' : 's'}
            </span>
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            <PathMetricList
              title="Accepted paths"
              subtitle="Where jobs finished after the fallback ladder settled on a final answer."
              items={suiteResults.path_metrics.accepted_paths || []}
              denominator={tracedJobCount}
              accent="linear-gradient(135deg, #4f46e5, #2563eb)"
            />
            <PathMetricList
              title="Transit counts"
              subtitle="How many jobs traversed each stage, including retries that were later rejected."
              items={suiteResults.path_metrics.transit_paths || []}
              denominator={tracedJobCount}
              accent="linear-gradient(135deg, #0f766e, #14b8a6)"
            />
          </div>
        </div>
      )}

      {/* ── Scatter chart + Feature 3: Performance ranking table ─────────── */}
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

        {/* Performance ranking table */}
        {(suiteResults?.points || []).length > 0 && (
          <div className="mt-6">
            <div className="mb-3 flex items-center gap-2">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-500">
                Performance Ranking
              </h3>
              <span className="rounded-full bg-indigo-50 px-2 py-0.5 text-[11px] font-medium text-indigo-600">
                {suiteResults!.points.length} runs
              </span>
            </div>
            <PerformanceTable points={suiteResults!.points} />
          </div>
        )}
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
            className="h-10 w-full rounded border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          />
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            <input
              value={testModalExpectedCategory}
              onChange={(event) => setTestModalExpectedCategory(event.target.value)}
              placeholder="Expected category"
              className="h-10 w-full rounded border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
            />
            <input
              value={testModalExpectedBrand}
              onChange={(event) => setTestModalExpectedBrand(event.target.value)}
              placeholder="Expected brand"
              className="h-10 w-full rounded border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
            />
          </div>
          <input
            value={testModalExpectedConfidence}
            onChange={(event) => setTestModalExpectedConfidence(event.target.value)}
            placeholder="Expected confidence (0..1)"
            className="h-10 w-full rounded border border-gray-300 bg-white px-3 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          />
          <textarea
            value={testModalExpectedReasoning}
            onChange={(event) => setTestModalExpectedReasoning(event.target.value)}
            placeholder="Expected reasoning"
            className="h-24 w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
          />
          <textarea
            value={testModalExpectedOcr}
            onChange={(event) => setTestModalExpectedOcr(event.target.value)}
            placeholder="Expected OCR corpus"
            className="h-24 w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-primary-500 focus:outline-none"
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
