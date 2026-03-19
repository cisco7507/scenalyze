import { useEffect, useState } from 'react';
import { getClusterNodes, getClusterJobs, getMetrics, setClusterNodeMaintenance } from '../lib/api';
import type { ClusterNode, JobStatus, Metrics } from '../lib/api';
import { Share1Icon, ExclamationTriangleIcon, UpdateIcon, LightningBoltIcon } from '@radix-ui/react-icons';

function NodeBadge({
  name,
  url,
  isUp,
  isSelf,
  isMaintenance,
  isAccepting,
  controlsAvailable,
  busy,
  onToggleMaintenance,
}: {
  name: string;
  url: string;
  isUp: boolean;
  isSelf: boolean;
  isMaintenance: boolean;
  isAccepting: boolean;
  controlsAvailable: boolean;
  busy: boolean;
  onToggleMaintenance: () => void;
}) {
  return (
    <tr className="transition-colors hover:bg-primary-50/50">
      <td className="px-6 py-4 font-semibold text-slate-800">
        <div className="flex items-center gap-2">
          <span>{name}</span>
          {isSelf ? (
            <span className="rounded-full border border-primary-200 bg-primary-50 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-primary-700">
              Self
            </span>
          ) : null}
        </div>
      </td>
      <td className="px-6 py-4 text-xs font-mono text-slate-500">{url}</td>
      <td className="px-6 py-4">
        <div className="flex flex-wrap items-center justify-end gap-2">
        {isUp ? (
          <span className="inline-flex items-center gap-2 rounded-full border border-primary-200 bg-primary-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-primary-700">
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary-300 opacity-70" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-primary-500" />
            </span>
            Online
          </span>
        ) : (
          <span className="inline-flex items-center gap-2 rounded-full border border-rose-200 bg-rose-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-rose-700">
            <ExclamationTriangleIcon className="h-3.5 w-3.5" />
            Unreachable
          </span>
        )}
        {isUp ? (
          isAccepting ? (
            <span className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-emerald-700">
              Accepting
            </span>
          ) : isMaintenance ? (
            <span className="inline-flex items-center gap-2 rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-amber-700">
              Maintenance
            </span>
          ) : (
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-600">
              Admission paused
            </span>
          )
        ) : null}
        {!isUp || !controlsAvailable ? null : (
          <button
            type="button"
            onClick={onToggleMaintenance}
            disabled={busy}
            className={`inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] transition-colors ${
              isMaintenance
                ? 'border-primary-200 bg-white text-primary-700 hover:border-primary-300 hover:bg-primary-50'
                : 'border-amber-200 bg-white text-amber-700 hover:border-amber-300 hover:bg-amber-50'
            } disabled:cursor-not-allowed disabled:opacity-50`}
          >
            {busy ? 'Updating…' : isMaintenance ? 'Return to production' : 'Enter maintenance'}
          </button>
        )}
        </div>
      </td>
    </tr>
  );
}

function StatCard({ label, value, detail, tone }: { label: string; value: string | number; detail: string; tone: 'blue' | 'amber' | 'emerald' | 'rose'; }) {
  const toneMap = {
    blue: 'from-primary-50 to-white border-primary-200/80 text-primary-900',
    amber: 'from-[rgba(247,244,237,1)] to-white border-[#e5d9c7] text-slate-900',
    emerald: 'from-[rgba(221,238,250,0.75)] to-white border-primary-200/80 text-primary-900',
    rose: 'from-rose-50 to-white border-rose-200/80 text-rose-900',
  } as const;

  return (
    <div className={`rounded-[1.9rem] border bg-[linear-gradient(135deg,var(--tw-gradient-from),var(--tw-gradient-to))] p-5 shadow-[0_18px_45px_rgba(0,55,120,0.08)] ${toneMap[tone]}`}>
      <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">{label}</div>
      <div className="mt-3 text-4xl font-bold">{value}</div>
      <div className="mt-2 text-sm text-slate-600">{detail}</div>
    </div>
  );
}

export function Overview() {
  const [nodes, setNodes] = useState<ClusterNode | null>(null);
  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState('');
  const [nodeErr, setNodeErr] = useState('');
  const [maintenanceBusy, setMaintenanceBusy] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let unmounted = false;

    const poll = async () => {
      try {
        const j = await getClusterJobs();
        if (!unmounted) {
          setJobs(j);
          setError('');
        }
      } catch (err: any) {
        if (!unmounted) setError(err.message || 'Failed to fetch jobs');
      }

      try {
        const n = await getClusterNodes();
        if (!unmounted) {
          setNodes(n);
          setNodeErr('');
        }
      } catch (err: any) {
        if (!unmounted) setNodeErr(err.message || 'Cannot reach API');
      }

      try {
        const m = await getMetrics();
        if (!unmounted) setMetrics(m);
      } catch {}

      if (!unmounted) setTimeout(poll, 3000);
    };

    poll();
    return () => {
      unmounted = true;
    };
  }, []);

  const processing = jobs.filter((j) => j.status === 'processing').length;
  const queued = jobs.filter((j) => j.status === 'queued' || j.status === 're-queued').length;
  const completed = jobs.filter((j) => j.status === 'completed').length;
  const failed = jobs.filter((j) => j.status === 'failed').length;
  const offlineNodes = nodes ? Object.values(nodes.status).filter((v) => !v).length : 0;
  const onlineNodes = nodes ? Object.values(nodes.status).filter(Boolean).length : 0;
  const totalNodes = nodes ? Object.keys(nodes.status).length : 0;
  const maintenanceApiAvailable = Boolean(
    nodes &&
      Object.keys(nodes.maintenance || {}).length > 0 &&
      Object.keys(nodes.accepting_new_jobs || {}).length > 0,
  );
  const maintenanceNodes = nodes && maintenanceApiAvailable ? Object.values(nodes.maintenance || {}).filter(Boolean).length : 0;

  const refreshCluster = async () => {
    const [n, j, m] = await Promise.all([
      getClusterNodes(),
      getClusterJobs(),
      getMetrics().catch(() => null),
    ]);
    setNodes(n);
    setJobs(j);
    if (m) setMetrics(m);
  };

  const toggleMaintenance = async (nodeName: string, nextEnabled: boolean) => {
    setMaintenanceBusy((current) => ({ ...current, [nodeName]: true }));
    try {
      await setClusterNodeMaintenance(nodeName, nextEnabled);
      await refreshCluster();
    } catch (err: any) {
      const message = String(err?.message || `Failed to update maintenance mode for ${nodeName}`);
      if (/not found/i.test(message)) {
        setNodeErr('Maintenance controls require restarting the API nodes so the new maintenance endpoints are loaded.');
      } else {
        setNodeErr(message);
      }
    } finally {
      setMaintenanceBusy((current) => ({ ...current, [nodeName]: false }));
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <section className="bell-hero">
        <div className="relative z-10 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <div className="bell-badge">
              <Share1Icon className="h-3.5 w-3.5" />
              Live Cluster View
            </div>
            <h2 className="mt-4 max-w-2xl text-[3rem] font-bold text-primary-700">See the fleet before you inspect the jobs.</h2>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">
              This page surfaces availability, queue pressure, and output quality signals so an operator can spot imbalance before diving into individual jobs
            </p>
          </div>
          <div className="grid min-w-[18rem] grid-cols-2 gap-3 lg:w-[22rem]">
            <div className="rounded-[1.7rem] border border-white/90 bg-white/88 px-4 py-4 shadow-[0_14px_28px_rgba(0,55,120,0.08)]">
              <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Nodes online</div>
              <div className="mt-2 text-3xl font-bold text-slate-950">{onlineNodes}/{totalNodes || '—'}</div>
              <div className="mt-2 text-sm text-slate-500">Shared-nothing workers with proxy-to-owner routing.</div>
            </div>
            <div className="rounded-[1.7rem] border border-primary-700/30 bg-primary-700 px-4 py-4 text-white shadow-[0_20px_42px_rgba(0,55,120,0.22)]">
              <div className="text-[11px] uppercase tracking-[0.22em] text-white/65">API uptime</div>
              <div className="mt-2 text-3xl font-bold">{metrics ? `${Math.floor(metrics.uptime_seconds / 60)}m` : '—'}</div>
              <div className="mt-2 text-sm text-white/72">Node {metrics?.node || '—'} is driving this control surface.</div>
            </div>
          </div>
        </div>
      </section>

      {nodeErr ? (
        <div className="rounded-[24px] border border-rose-200 bg-rose-50/90 p-4 text-rose-700 shadow-sm">
          <div className="flex items-start gap-3">
            <ExclamationTriangleIcon className="mt-0.5 h-5 w-5 shrink-0" />
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.2em]">API unreachable</p>
              <p className="mt-1 text-sm text-rose-700/90">{nodeErr}</p>
            </div>
          </div>
        </div>
      ) : null}

      {!nodeErr && offlineNodes > 0 ? (
        <div className="rounded-[24px] border border-[#e5d9c7] bg-[rgba(247,244,237,0.96)] p-4 text-slate-700 shadow-sm">
          <div className="flex items-start gap-3">
            <ExclamationTriangleIcon className="mt-0.5 h-5 w-5 shrink-0 text-[#a1632b]" />
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-[#a1632b]">Partial visibility</p>
              <p className="mt-1 text-sm text-slate-700">{offlineNodes} node{offlineNodes > 1 ? 's are' : ' is'} unreachable. Jobs owned by those nodes may not appear until connectivity returns</p>
            </div>
          </div>
        </div>
      ) : null}

      {error && !nodeErr ? (
        <div className="rounded-[24px] border border-rose-200 bg-rose-50/90 p-4 text-rose-700 shadow-sm">
          <div className="flex items-center gap-3 text-sm">
            <ExclamationTriangleIcon className="h-4 w-4" />
            {error}
          </div>
        </div>
      ) : null}

      <section className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        <StatCard label="Processing" value={processing} detail="Jobs actively moving through OCR, vision, or LLM stages." tone="blue" />
        <StatCard label="Queued" value={queued} detail="Work waiting to be claimed or re-queued after recovery." tone="amber" />
        <StatCard label="Completed" value={completed} detail="Jobs that finished cleanly and persisted their result payload." tone="emerald" />
        <StatCard label="Failed" value={failed} detail="Runs that need operator review or a pipeline retry." tone="rose" />
      </section>

      {metrics ? (
        <section className="grid gap-3 md:grid-cols-4">
          {[
            { label: 'Total Completed (DB)', value: metrics.jobs_completed },
            { label: 'Total Failed (DB)', value: metrics.jobs_failed },
            { label: 'Submitted This Session', value: metrics.jobs_submitted_this_process },
            { label: 'API Uptime', value: `${Math.floor(metrics.uptime_seconds / 60)}m ${metrics.uptime_seconds % 60}s` },
          ].map((item) => (
            <div key={item.label} className="bell-panel px-4 py-4">
              <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">{item.label}</div>
              <div className="mt-3 text-2xl font-bold text-slate-950">{item.value}</div>
            </div>
          ))}
        </section>
      ) : null}

      <section className="bell-panel overflow-hidden">
        <div className="flex items-center justify-between border-b border-slate-200/80 bg-primary-50/75 px-6 py-4">
          <div>
            <h3 className="text-lg font-bold text-slate-900">Worker fleet</h3>
            <p className="mt-1 text-sm text-slate-500">Each node is a claim-capable executor with deterministic owner routing. Maintenance mode drains local work while blocking new admissions.</p>
          </div>
          <div className="bell-data-pill">
            <LightningBoltIcon className="h-3.5 w-3.5 text-primary-500" />
            {onlineNodes}/{totalNodes || '—'} online{maintenanceApiAvailable ? ` · ${maintenanceNodes} in maintenance` : ''}
          </div>
        </div>
        {!maintenanceApiAvailable ? (
          <div className="border-t border-slate-200/80 bg-amber-50/80 px-6 py-3 text-sm text-amber-800">
            Restart the API nodes to enable maintenance controls on this fleet view.
          </div>
        ) : null}
        <div className="overflow-x-auto">
          <table className="w-full min-w-[880px] text-left text-sm">
            <thead className="bg-slate-50/80 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">
              <tr>
                <th className="px-6 py-4">Node</th>
                <th className="px-6 py-4">Internal URL</th>
                <th className="px-6 py-4 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {nodes ? (
                Object.entries(nodes.nodes).map(([name, url]) => (
                  (() => {
                    const hasMaintenanceState = Object.prototype.hasOwnProperty.call(nodes.maintenance || {}, name);
                    const hasAcceptingState = Object.prototype.hasOwnProperty.call(nodes.accepting_new_jobs || {}, name);
                    const isMaintenance = hasMaintenanceState ? Boolean(nodes.maintenance?.[name]) : false;
                    const isAccepting = hasAcceptingState ? Boolean(nodes.accepting_new_jobs?.[name]) : Boolean(nodes.status[name]);
                    return (
                  <NodeBadge
                    key={name}
                    name={name}
                    url={url as string}
                    isUp={nodes.status[name]}
                    isSelf={name === nodes.self}
                    isMaintenance={isMaintenance}
                    isAccepting={isAccepting}
                    controlsAvailable={maintenanceApiAvailable}
                    busy={Boolean(maintenanceBusy[name])}
                    onToggleMaintenance={() => toggleMaintenance(name, !isMaintenance)}
                  />
                    );
                  })()
                ))
              ) : (
                <tr>
                  <td colSpan={3} className="px-6 py-12 text-center text-slate-400">
                    {nodeErr ? (
                      <span className="inline-flex items-center gap-2 text-rose-400">
                        <ExclamationTriangleIcon /> {nodeErr}
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-2">
                        <UpdateIcon className="animate-spin" /> Loading nodes…
                      </span>
                    )}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
