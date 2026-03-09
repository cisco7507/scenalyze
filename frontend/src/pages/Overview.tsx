import { useEffect, useState } from 'react';
import { getClusterNodes, getClusterJobs, getMetrics } from '../lib/api';
import type { ClusterNode, JobStatus, Metrics } from '../lib/api';
import { Share1Icon, ExclamationTriangleIcon, UpdateIcon, LightningBoltIcon } from '@radix-ui/react-icons';

function NodeBadge({ name, url, isUp, isSelf }: { name: string; url: string; isUp: boolean; isSelf: boolean }) {
  return (
    <tr className="transition-colors hover:bg-slate-50/80">
      <td className="px-6 py-4 font-semibold text-slate-800">
        <div className="flex items-center gap-2">
          <span>{name}</span>
          {isSelf ? (
            <span className="rounded-full border border-primary-200 bg-primary-50 px-2 py-0.5 text-[10px] font-mono font-semibold uppercase tracking-wider text-primary-700">
              Self
            </span>
          ) : null}
        </div>
      </td>
      <td className="px-6 py-4 text-xs font-mono text-slate-500">{url}</td>
      <td className="px-6 py-4 text-right">
        {isUp ? (
          <span className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-emerald-700">
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-70" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
            </span>
            Online
          </span>
        ) : (
          <span className="inline-flex items-center gap-2 rounded-full border border-rose-200 bg-rose-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-rose-700">
            <ExclamationTriangleIcon className="h-3.5 w-3.5" />
            Unreachable
          </span>
        )}
      </td>
    </tr>
  );
}

function StatCard({ label, value, detail, tone }: { label: string; value: string | number; detail: string; tone: 'blue' | 'amber' | 'emerald' | 'rose'; }) {
  const toneMap = {
    blue: 'from-sky-50 to-blue-50 border-sky-200/80 text-sky-900',
    amber: 'from-amber-50 to-orange-50 border-amber-200/80 text-amber-900',
    emerald: 'from-emerald-50 to-teal-50 border-emerald-200/80 text-emerald-900',
    rose: 'from-rose-50 to-pink-50 border-rose-200/80 text-rose-900',
  } as const;

  return (
    <div className={`rounded-[26px] border bg-[linear-gradient(135deg,var(--tw-gradient-from),var(--tw-gradient-to))] p-5 shadow-[0_18px_45px_rgba(15,23,42,0.06)] ${toneMap[tone]}`}>
      <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">{label}</div>
      <div className="mt-3 text-4xl font-black tracking-[-0.05em]">{value}</div>
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

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <section className="rounded-[30px] border border-slate-200/80 bg-[linear-gradient(135deg,rgba(255,255,255,0.98)_0%,rgba(240,246,255,0.96)_55%,rgba(232,239,255,0.92)_100%)] p-7 shadow-[0_20px_55px_rgba(15,23,42,0.08)]">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 rounded-full border border-primary-200 bg-white/80 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.24em] text-primary-700">
              <Share1Icon className="h-3.5 w-3.5" />
              Live Cluster View
            </div>
            <h2 className="mt-4 text-4xl font-black tracking-[-0.06em] text-slate-950">See the fleet before you inspect the jobs.</h2>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">
              This page surfaces availability, queue pressure, and output quality signals so an operator can spot imbalance before diving into individual jobs
            </p>
          </div>
          <div className="grid min-w-[18rem] grid-cols-2 gap-3 lg:w-[22rem]">
            <div className="rounded-[22px] border border-white/80 bg-white/75 px-4 py-4 shadow-sm">
              <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Nodes online</div>
              <div className="mt-2 text-3xl font-black tracking-[-0.05em] text-slate-950">{onlineNodes}/{totalNodes || '—'}</div>
              <div className="mt-2 text-sm text-slate-500">Shared-nothing workers with proxy-to-owner routing.</div>
            </div>
            <div className="rounded-[22px] border border-white/80 bg-slate-950 px-4 py-4 text-slate-50 shadow-[0_20px_45px_rgba(15,23,42,0.18)]">
              <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">API uptime</div>
              <div className="mt-2 text-3xl font-black tracking-[-0.05em]">{metrics ? `${Math.floor(metrics.uptime_seconds / 60)}m` : '—'}</div>
              <div className="mt-2 text-sm text-slate-400">Node {metrics?.node || '—'} is driving this control surface.</div>
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
        <div className="rounded-[24px] border border-amber-200 bg-amber-50/90 p-4 text-amber-800 shadow-sm">
          <div className="flex items-start gap-3">
            <ExclamationTriangleIcon className="mt-0.5 h-5 w-5 shrink-0" />
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.2em]">Partial visibility</p>
              <p className="mt-1 text-sm text-amber-800/90">{offlineNodes} node{offlineNodes > 1 ? 's are' : ' is'} unreachable. Jobs owned by those nodes may not appear until connectivity returns</p>
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
            <div key={item.label} className="rounded-[22px] border border-slate-200/80 bg-white/80 px-4 py-4 shadow-sm backdrop-blur">
              <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">{item.label}</div>
              <div className="mt-3 text-2xl font-black tracking-[-0.05em] text-slate-950">{item.value}</div>
            </div>
          ))}
        </section>
      ) : null}

      <section className="overflow-hidden rounded-[30px] border border-slate-200/80 bg-white/82 shadow-[0_18px_45px_rgba(15,23,42,0.06)] backdrop-blur">
        <div className="flex items-center justify-between border-b border-slate-200/80 bg-slate-50/70 px-6 py-4">
          <div>
            <h3 className="text-lg font-black tracking-[-0.03em] text-slate-900">Worker fleet</h3>
            <p className="mt-1 text-sm text-slate-500">Each node is a claim-capable executor with deterministic owner routing.</p>
          </div>
          <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
            <LightningBoltIcon className="h-3.5 w-3.5 text-primary-500" />
            {onlineNodes}/{totalNodes || '—'} online
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-sm">
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
                  <NodeBadge
                    key={name}
                    name={name}
                    url={url as string}
                    isUp={nodes.status[name]}
                    isSelf={name === nodes.self}
                  />
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
