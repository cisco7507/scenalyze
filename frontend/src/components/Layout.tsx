import { useMemo, useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { LayersIcon, BarChartIcon, ComponentInstanceIcon, CodeIcon, ClockIcon, RocketIcon } from '@radix-ui/react-icons';
import { cn } from '../lib/utils';
import { DebugConsole } from './DebugConsole';
import scenalyzeMark from '../assets/scenalyze-mark.png';

const pageMeta = {
  '/': {
    eyebrow: 'System Pulse',
    title: 'Operations overview',
    description: 'Cluster health, backlog pressure, and throughput signals in one place.',
  },
  '/jobs': {
    eyebrow: 'Work Queue',
    title: 'Job command center',
    description: 'Launch analysis, monitor progress, and intervene quickly when edge cases appear.',
  },
  '/analytics': {
    eyebrow: 'Latency & Yield',
    title: 'Analytics radar',
    description: 'Watch duration percentiles, tail risk, and execution patterns across the cluster.',
  },
  '/benchmark': {
    eyebrow: 'Controlled Trials',
    title: 'Benchmark lab',
    description: 'Compare model stacks, OCR settings, and execution profiles under repeatable suites.',
  },
} as const;

export function Layout() {
  const location = useLocation();
  const [debugConsoleOpen, setDebugConsoleOpen] = useState(false);

  const navItems = [
    { name: 'Overview', path: '/', icon: LayersIcon },
    { name: 'Jobs', path: '/jobs', icon: ComponentInstanceIcon },
    { name: 'Analytics', path: '/analytics', icon: BarChartIcon },
    { name: 'Benchmarking', path: '/benchmark', icon: ClockIcon },
  ];

  const activeMeta = useMemo(() => {
    const entry = Object.entries(pageMeta).find(([path]) =>
      path === '/'
        ? location.pathname === '/'
        : location.pathname === path || location.pathname.startsWith(`${path}/`),
    );
    return entry?.[1] ?? pageMeta['/'];
  }, [location.pathname]);

  return (
    <div className="flex h-screen bg-[radial-gradient(circle_at_top_left,rgba(96,165,250,0.14),transparent_28%),radial-gradient(circle_at_top_right,rgba(129,140,248,0.12),transparent_26%),linear-gradient(180deg,#f8fafc_0%,#eef2f7_100%)] text-slate-900 font-sans">
      <aside className="flex w-[18.5rem] shrink-0 flex-col border-r border-slate-200/80 bg-[linear-gradient(180deg,rgba(255,255,255,0.95)_0%,rgba(244,247,255,0.95)_100%)] px-4 py-4 shadow-[0_12px_40px_rgba(15,23,42,0.06)] backdrop-blur-xl">
        <Link
          to="/"
          className="group rounded-[26px] border border-white/80 bg-[linear-gradient(145deg,rgba(255,255,255,0.98)_0%,rgba(242,247,255,0.96)_48%,rgba(234,240,255,0.94)_100%)] px-4 py-4 shadow-[0_14px_40px_rgba(99,102,241,0.10)] transition-transform duration-200 hover:-translate-y-0.5"
          title="Scenalyze"
        >
          <div className="flex items-center gap-3">
            <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-[22px] bg-white/95 shadow-[inset_0_0_0_1px_rgba(148,163,184,0.18),0_8px_20px_rgba(79,70,229,0.10)]">
              <img src={scenalyzeMark} alt="Scenalyze mark" className="h-11 w-11 object-contain" />
            </div>
            <div className="min-w-0">
              <div className="text-[1.72rem] font-black tracking-[-0.05em] leading-none text-slate-950">
                <span className="bg-gradient-to-r from-sky-400 via-blue-500 to-violet-600 bg-clip-text text-transparent">S</span>
                cenalyze
              </div>
              <div className="mt-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">
                Intelligence meets video
              </div>
            </div>
          </div>
          <p className="mt-3 text-[12px] leading-relaxed text-slate-500">
            Every frame tells a story.
          </p>
        </Link>

        <div className="px-2 pt-6">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-400">Navigation</div>
          <nav className="space-y-1.5">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path));
              return (
                <Link
                  key={item.name}
                  to={item.path}
                  className={cn(
                    'flex items-center justify-between rounded-2xl border px-3.5 py-3 text-sm font-semibold transition-all duration-200',
                    isActive
                      ? 'border-primary-200 bg-[linear-gradient(135deg,rgba(99,102,241,0.14),rgba(56,189,248,0.10))] text-primary-900 shadow-[0_10px_24px_rgba(79,70,229,0.10)]'
                      : 'border-transparent text-slate-500 hover:border-slate-200 hover:bg-white/80 hover:text-slate-900',
                  )}
                >
                  <span className="flex items-center gap-3">
                    <span className={cn('flex h-9 w-9 items-center justify-center rounded-xl border', isActive ? 'border-primary-200 bg-white/90 text-primary-700' : 'border-slate-200 bg-white/70 text-slate-400')}>
                      <Icon className="h-4 w-4" />
                    </span>
                    {item.name}
                  </span>
                  {isActive ? <span className="h-2.5 w-2.5 rounded-full bg-primary-500 shadow-[0_0_0_4px_rgba(99,102,241,0.14)]" /> : null}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="mt-auto px-2 pt-6">
          <div className="rounded-[24px] border border-slate-200/80 bg-white/75 px-4 py-4 shadow-[0_10px_30px_rgba(15,23,42,0.05)] backdrop-blur">
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">
              <RocketIcon className="h-4 w-4 text-primary-500" />
              Control Center
            </div>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">
              Built for operators who need to classify ads fast, inspect edge cases, and explain why the system chose each path.
            </p>
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="border-b border-slate-200/80 bg-white/70 backdrop-blur-xl">
          <div className="mx-auto flex max-w-7xl items-start justify-between gap-6 px-8 py-6">
            <div className="min-w-0">
              <div className="text-[11px] font-semibold uppercase tracking-[0.26em] text-slate-400">{activeMeta.eyebrow}</div>
              <h1 className="mt-2 text-[2rem] font-black tracking-[-0.05em] text-slate-950">{activeMeta.title}</h1>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">{activeMeta.description}</p>
            </div>
            <button
              type="button"
              onClick={() => setDebugConsoleOpen((current) => !current)}
              className="inline-flex shrink-0 items-center gap-2 rounded-full border border-slate-300 bg-white/80 px-4 py-2.5 text-xs font-semibold uppercase tracking-[0.2em] text-slate-700 shadow-sm transition-colors hover:border-slate-400 hover:bg-white"
            >
              <CodeIcon className="h-4 w-4" />
              Terminal
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-auto">
          <main className="mx-auto min-h-full max-w-7xl px-8 py-8">
            <Outlet />
          </main>
        </div>

        <DebugConsole open={debugConsoleOpen} onClose={() => setDebugConsoleOpen(false)} />
      </div>
    </div>
  );
}
