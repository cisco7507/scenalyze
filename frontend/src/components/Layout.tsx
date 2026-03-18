import { useMemo, useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { LayersIcon, BarChartIcon, ComponentInstanceIcon, CodeIcon, ClockIcon, LightningBoltIcon, MagnifyingGlassIcon } from '@radix-ui/react-icons';
import { DebugConsole } from './DebugConsole';
import { BrandLogo } from './BrandLogo';

const pageMeta = {
  '/': {
    eyebrow: 'Operations Overview',
    title: 'Bell-aligned control surface for live classifier operations.',
    description: 'Track cluster health, queue pressure, and output quality from a single interface tuned for Bell media operations.',
  },
  '/jobs': {
    eyebrow: 'Pipeline Queue',
    title: 'Launch, triage, and review analysis runs with one operational workflow.',
    description: 'Submit new jobs, watch pipeline stages, and jump straight into evidence when a run needs operator attention.',
  },
  '/analytics': {
    eyebrow: 'Latency And Yield',
    title: 'Tail-aware analytics for cluster health and execution risk.',
    description: 'See percentile drift, trace-path mix, and the operating signals that show when slow paths are taking over.',
  },
  '/taxonomy': {
    eyebrow: 'FreeWheel Taxonomy',
    title: 'Explore the loaded category hierarchy without leaving the control surface.',
    description: 'Search groups, inspect canonical paths, and follow how industry bridges connect to the deeper taxonomy used by the mapper.',
  },
  '/benchmark': {
    eyebrow: 'Controlled Trials',
    title: 'Benchmark suites for repeatable model and OCR comparisons.',
    description: 'Run controlled evaluations across providers, OCR profiles, and scan strategies without leaving the product shell.',
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
    { name: 'Taxonomy', path: '/taxonomy', icon: MagnifyingGlassIcon },
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
    <div className="min-h-screen bg-[linear-gradient(180deg,#f7f8f5_0%,#eef4f8_100%)] text-slate-900">
      <header className="sticky top-0 z-40">
        <div className="bell-topbar">
          <div className="mx-auto flex max-w-[88rem] flex-wrap items-center justify-between gap-3 px-6 py-2.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-white/84">
            <div className="flex flex-wrap items-center gap-3">
              <span>Bell media operations</span>
              <span className="hidden text-white/40 md:inline">/</span>
              <span className="hidden md:inline">Shared-nothing routing</span>
            </div>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => setDebugConsoleOpen((current) => !current)}
                className="inline-flex items-center gap-2 rounded-full border border-white/25 bg-white/10 px-3 py-1.5 text-[11px] font-bold tracking-[0.16em] text-white transition-colors hover:bg-white/16"
              >
                <CodeIcon className="h-3.5 w-3.5" />
                Logs
              </button>
            </div>
          </div>
        </div>

        <div className="bell-mainnav">
          <div className="mx-auto flex max-w-[88rem] flex-col gap-4 px-6 py-4 xl:flex-row xl:items-center xl:justify-between">
            <Link to="/" className="block" title="Scenalyze">
              <BrandLogo />
            </Link>

            <nav className="flex flex-1 flex-wrap items-center gap-2 xl:justify-center">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path));
                return (
                  <Link
                    key={item.name}
                    to={item.path}
                    data-active={isActive ? 'true' : 'false'}
                    className="bell-nav-link"
                  >
                    <Icon className="mr-2 h-4 w-4" />
                    {item.name}
                  </Link>
                );
              })}
            </nav>
          </div>
        </div>

        <div className="border-b border-slate-200/90 bg-white/88 backdrop-blur">
          <div className="mx-auto flex max-w-[88rem] flex-col gap-4 px-6 py-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="min-w-0">
              <div className="bell-kicker">{activeMeta.eyebrow}</div>
              <h1 className="mt-2 text-[1.65rem] font-bold text-primary-700">{activeMeta.title}</h1>
              <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-600">{activeMeta.description}</p>
            </div>
            <div className="bell-data-pill self-start lg:self-auto">
              <LightningBoltIcon className="h-3.5 w-3.5 text-primary-500" />
              Operational UI live
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[88rem] px-6 py-8">
        <Outlet />
      </main>

      <DebugConsole open={debugConsoleOpen} onClose={() => setDebugConsoleOpen(false)} />
    </div>
  );
}
