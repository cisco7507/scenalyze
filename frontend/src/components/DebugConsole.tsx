import { useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import {
  Cross2Icon,
  MagnifyingGlassIcon,
  ReloadIcon,
  TrashIcon,
} from '@radix-ui/react-icons';
import { API_BASE_URL } from '../lib/api';
import { cn } from '../lib/utils';

type ConnectionState = 'connecting' | 'open' | 'error';
type LogLevel = 'ERROR' | 'WARNING' | 'INFO' | 'DEBUG' | 'OTHER';

interface DebugConsoleProps {
  open: boolean;
  onClose: () => void;
}

interface LogLine {
  id: number;
  text: string;
  level: LogLevel;
  hasTimeout: boolean;
  hasException: boolean;
  parsed: ParsedLogLine;
}

interface ParsedLogLine {
  timestamp: string;
  level: LogLevel;
  jobId: string;
  stage: string;
  module: string;
  message: string;
  parsed: boolean;
}

const MAX_LINES = 1500;
const CONNECT_TIMEOUT_MS = 5000;
const POLL_INTERVAL_MS = 2000;
const LOG_LINE_REGEX =
  /^(?<timestamp>\S+)\s+(?<level>[A-Z]+)\s+job_id=(?<job_id>\S+)\s+stage=(?<stage>\S+)\s+(?<module>[A-Za-z0-9_.-]+)\s+(?<message>.*)$/;
const UVICORN_LINE_REGEX = /^(?<level>INFO|WARNING|ERROR|DEBUG):\s+(?<message>.*)$/;

function detectLevel(line: string): LogLevel {
  if (/\bERROR\b/i.test(line)) return 'ERROR';
  if (/\bWARNING\b/i.test(line)) return 'WARNING';
  if (/\bINFO\b/i.test(line)) return 'INFO';
  if (/\bDEBUG\b/i.test(line)) return 'DEBUG';
  return 'OTHER';
}

function parseLogLine(text: string): ParsedLogLine {
  const cleanText = text.replace(/\x1b\[[0-9;]*m/g, '');
  const match = cleanText.match(LOG_LINE_REGEX);
  const fallbackLevel = detectLevel(cleanText);

  if (!match || !match.groups) {
    const uvicornMatch = cleanText.match(UVICORN_LINE_REGEX);
    if (uvicornMatch?.groups) {
      return {
        timestamp: '',
        level: detectLevel(uvicornMatch.groups.level),
        jobId: '-',
        stage: '-',
        module: 'uvicorn',
        message: uvicornMatch.groups.message || cleanText,
        parsed: true,
      };
    }

    return {
      timestamp: '',
      level: fallbackLevel,
      jobId: '',
      stage: '',
      module: '',
      message: cleanText,
      parsed: false,
    };
  }

  const parsedLevel = detectLevel(match.groups.level || fallbackLevel);
  return {
    timestamp: match.groups.timestamp || '',
    level: parsedLevel,
    jobId: match.groups.job_id || '',
    stage: match.groups.stage || '',
    module: match.groups.module || '',
    message: match.groups.message || '',
    parsed: true,
  };
}

function buildLogLine(id: number, text: string): LogLine {
  const parsed = parseLogLine(text);
  return {
    id,
    text,
    level: parsed.level,
    hasTimeout: /\btimeout\b|\btimed out\b/i.test(text),
    hasException: /\bexception\b|\btraceback\b/i.test(text),
    parsed,
  };
}

function formatJobId(jobId: string): string {
  if (!jobId || jobId === '-') return '';
  if (jobId.length <= 30) return jobId;
  return `${jobId.slice(0, 20)}…${jobId.slice(-8)}`;
}

function levelClass(level: LogLevel): string {
  if (level === 'ERROR') return 'border-red-500/40 bg-red-900/30 text-red-200';
  if (level === 'WARNING') return 'border-amber-500/40 bg-amber-900/30 text-amber-200';
  if (level === 'INFO') return 'border-emerald-500/40 bg-emerald-900/30 text-emerald-200';
  if (level === 'DEBUG') return 'border-sky-500/40 bg-sky-900/30 text-sky-200';
  return 'border-slate-600/50 bg-slate-800/60 text-slate-200';
}

function stageClass(stage: string): string {
  const value = (stage || '').toLowerCase();
  if (!value || value === '-') return 'border-slate-700 bg-slate-900/50 text-slate-400';
  if (value.includes('ingest') || value.includes('download')) return 'border-violet-500/40 bg-violet-900/30 text-violet-200';
  if (value.includes('frame')) return 'border-indigo-500/40 bg-indigo-900/30 text-indigo-200';
  if (value.includes('ocr')) return 'border-cyan-500/40 bg-cyan-900/30 text-cyan-200';
  if (value.includes('vision')) return 'border-fuchsia-500/40 bg-fuchsia-900/30 text-fuchsia-200';
  if (value.includes('llm')) return 'border-pink-500/40 bg-pink-900/30 text-pink-200';
  if (value.includes('persist')) return 'border-teal-500/40 bg-teal-900/30 text-teal-200';
  if (value.includes('fail')) return 'border-red-500/40 bg-red-900/30 text-red-200';
  if (value.includes('complete')) return 'border-emerald-500/40 bg-emerald-900/30 text-emerald-200';
  return 'border-slate-600/50 bg-slate-800/60 text-slate-300';
}

function renderMessage(message: string): ReactNode {
  const parts = message.split(/(\b[a-zA-Z_][a-zA-Z0-9_:-]*=[^,\s)]+)/g);
  if (parts.length === 1) return message;
  return parts.map((part, index) => {
    if (!part) return null;
    if (/^[a-zA-Z_][a-zA-Z0-9_:-]*=[^,\s)]+$/.test(part)) {
      return (
        <span
          key={`${part}-${index}`}
          className="rounded border border-slate-600/60 bg-slate-800/70 px-1 py-0.5 text-[11px] text-cyan-200"
        >
          {part}
        </span>
      );
    }
    return <span key={`${part}-${index}`}>{part}</span>;
  });
}

export function DebugConsole({ open, onClose }: DebugConsoleProps) {
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [query, setQuery] = useState('');
  const [hideInfo, setHideInfo] = useState(false);
  const [hideDebug, setHideDebug] = useState(false);
  const [connection, setConnection] = useState<ConnectionState>('connecting');
  const [stickToBottom, setStickToBottom] = useState(true);
  const lineCounter = useRef(1);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  const clearPollTimer = () => {
    if (pollTimerRef.current != null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const [reconnectTrigger, setReconnectTrigger] = useState(0);

  const appendLogs = (rawLines: string[]) => {
    if (!rawLines.length) return;
    setLogs((prev) => {
      const next = rawLines
        .filter((line) => typeof line === 'string' && line.trim().length > 0)
        .map((line) => {
          const item = buildLogLine(lineCounter.current, line);
          lineCounter.current += 1;
          return item;
        });
      if (!next.length) return prev;
      const merged = [...prev, ...next];
      return merged.length > MAX_LINES ? merged.slice(-MAX_LINES) : merged;
    });
  };

  useEffect(() => {
    if (!open) return undefined;

    const apiRoot = API_BASE_URL.replace(/\/$/, '');
    const endpoint = `${apiRoot}/admin/logs/stream`;
    const recentEndpoint = `${apiRoot}/admin/logs/recent?limit=300`;
    let gotAnyPayload = false;
    let disposed = false;
    let connectTimeoutId: number | null = null;

    const bootstrapFromRecent = async () => {
      try {
        const response = await fetch(recentEndpoint);
        if (!response.ok) return;
        const payload = (await response.json()) as { lines?: string[] };
        appendLogs(payload.lines ?? []);
      } catch {
        // Ignore recent-fetch failures; SSE may still connect.
      }
    };

    const startFallbackPolling = () => {
      if (pollTimerRef.current != null) return;
      pollTimerRef.current = window.setInterval(() => {
        void bootstrapFromRecent();
      }, POLL_INTERVAL_MS);
    };

    setConnection('connecting');
    void bootstrapFromRecent();
    const source = new EventSource(endpoint);

    source.onopen = () => {
      if (disposed) return;
      setConnection('open');
      clearPollTimer();
      if (connectTimeoutId != null) {
        window.clearTimeout(connectTimeoutId);
        connectTimeoutId = null;
      }
    };
    source.onerror = () => {
      if (disposed) return;
      setConnection('error');
      startFallbackPolling();
    };

    source.addEventListener('bootstrap', (event: MessageEvent) => {
      try {
        const payload = JSON.parse(event.data) as { lines?: string[] };
        appendLogs(payload.lines ?? []);
        gotAnyPayload = true;
        setConnection('open');
        clearPollTimer();
      } catch {
        // Ignore malformed bootstrap payloads.
      }
    });

    source.addEventListener('log', (event: MessageEvent) => {
      try {
        const payload = JSON.parse(event.data) as { line?: string };
        if (payload.line) {
          appendLogs([payload.line]);
          gotAnyPayload = true;
          setConnection('open');
          clearPollTimer();
        }
      } catch {
        // Ignore malformed log payloads.
      }
    });

    source.onmessage = (event: MessageEvent) => {
      try {
        const payload = JSON.parse(event.data) as { line?: string; lines?: string[] };
        if (payload.lines?.length) {
          appendLogs(payload.lines);
          gotAnyPayload = true;
          setConnection('open');
          clearPollTimer();
          return;
        }
        if (payload.line) {
          appendLogs([payload.line]);
          gotAnyPayload = true;
          setConnection('open');
          clearPollTimer();
        }
      } catch {
        if (event.data) {
          appendLogs([event.data]);
          gotAnyPayload = true;
          setConnection('open');
          clearPollTimer();
        }
      }
    };

    connectTimeoutId = window.setTimeout(() => {
      if (disposed) return;
      if (!gotAnyPayload && source.readyState !== EventSource.OPEN) {
        setConnection('error');
        startFallbackPolling();
      }
    }, CONNECT_TIMEOUT_MS);

    return () => {
      disposed = true;
      if (connectTimeoutId != null) {
        window.clearTimeout(connectTimeoutId);
      }
      clearPollTimer();
      source.close();
      setConnection('connecting');
    };
  }, [open, reconnectTrigger]);

  useEffect(() => {
    if (!open || !stickToBottom) return;
    const node = scrollerRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [logs, open, stickToBottom]);

  const filteredLogs = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return logs.filter((line) => {
      if (hideDebug && line.level === 'DEBUG') return false;
      if (hideInfo && line.level === 'INFO') return false;
      if (!needle) return true;
      return line.text.toLowerCase().includes(needle);
    });
  }, [logs, query, hideDebug, hideInfo]);

  const connectionLabel = connection === 'open' ? 'Live' : connection === 'error' ? 'Retrying' : 'Connecting';
  const connectionClass =
    connection === 'open'
      ? 'text-emerald-300 bg-emerald-900/40 border-emerald-600/40'
      : connection === 'error'
        ? 'text-amber-200 bg-amber-900/40 border-amber-600/40'
        : 'text-slate-200 bg-slate-800/50 border-slate-600/40';

  return (
    <div
      className={cn(
        'fixed inset-x-0 bottom-0 z-[70] transition-transform duration-300 ease-out',
        open ? 'translate-y-0' : 'translate-y-full'
      )}
    >
      <div className="mx-3 mb-3 rounded-xl border border-primary-600/30 bg-slate-800/96 shadow-2xl backdrop-blur-sm lg:mx-6">
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-700/60 px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-1.5 rounded-full bg-primary-400 animate-pulse" />
            <div className="text-sm font-semibold tracking-wide text-slate-100">Server Logs</div>
          </div>
          <div className={cn('rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase', connectionClass)}>
            {connectionLabel}
          </div>
          <div className="relative ml-auto min-w-[220px] flex-1 max-w-md">
            <MagnifyingGlassIcon className="pointer-events-none absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-400" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Filter logs (job_id, error, timeout...)"
              className="h-8 w-full rounded-md border border-slate-600 bg-slate-700/80 pl-8 pr-2 text-xs text-slate-100 placeholder:text-slate-400 focus:border-primary-500 focus:outline-none"
            />
          </div>
          <label className="flex items-center gap-1 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={hideInfo}
              onChange={(event) => setHideInfo(event.target.checked)}
              className="h-3.5 w-3.5 accent-primary-500"
            />
            hide INFO
          </label>
          <label className="flex items-center gap-1 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={hideDebug}
              onChange={(event) => setHideDebug(event.target.checked)}
              className="h-3.5 w-3.5 accent-primary-500"
            />
            hide DEBUG
          </label>
          <button
            type="button"
            onClick={() => {
              setLogs([]);
              lineCounter.current = 1;
              setStickToBottom(true);
              setReconnectTrigger((x) => x + 1);
            }}
            className="inline-flex h-8 items-center gap-1 rounded-md border border-slate-600 bg-slate-700/80 px-2 text-xs font-semibold text-slate-200 hover:bg-slate-600/80"
          >
            <TrashIcon className="h-3.5 w-3.5" />
            Clear
          </button>
          <button
            type="button"
            onClick={onClose}
            className="inline-flex h-8 items-center gap-1 rounded-md border border-slate-600 bg-slate-700/80 px-2 text-xs font-semibold text-slate-200 hover:bg-slate-600/80"
          >
            <Cross2Icon className="h-3.5 w-3.5" />
            Close
          </button>
        </div>

        <div
          ref={scrollerRef}
          onScroll={(event) => {
            const target = event.currentTarget;
            const nearBottom = target.scrollHeight - target.scrollTop - target.clientHeight < 24;
            setStickToBottom(nearBottom);
          }}
          className="h-[44vh] overflow-y-auto px-2 py-2 font-mono text-xs"
        >
          {filteredLogs.length === 0 ? (
            <div className="flex h-full items-center justify-center gap-2 text-slate-400">
              <ReloadIcon className="h-4 w-4 animate-spin" />
              Waiting for logs...
            </div>
          ) : (
            <div className="space-y-1">
              {filteredLogs.map((line) => {
                const parsed = line.parsed;
                const rowBaseClass = line.hasTimeout
                  ? 'border-orange-500/40 bg-orange-950/25'
                  : line.level === 'ERROR' || line.hasException
                    ? 'border-red-500/35 bg-red-950/25'
                    : line.level === 'WARNING'
                      ? 'border-amber-500/35 bg-amber-950/20'
                      : 'border-slate-800 bg-slate-950/40';

                if (!parsed.parsed) {
                  return (
                    <div
                      key={line.id}
                      className={cn(
                        'rounded border px-2 py-1 leading-5 whitespace-pre-wrap break-words text-slate-200',
                        rowBaseClass
                      )}
                    >
                      {line.text}
                    </div>
                  );
                }

                return (
                  <div
                    key={line.id}
                    className={cn('rounded border px-2 py-1 transition-colors', rowBaseClass)}
                  >
                    <div className="grid min-w-[980px] grid-cols-[170px_60px_100px_minmax(0,1fr)] items-start gap-3">
                      <div className="truncate pt-[2px] text-[11px] text-slate-500">{parsed.timestamp}</div>
                      <div>
                        <span
                          className={cn(
                            'inline-flex rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide',
                            levelClass(parsed.level)
                          )}
                        >
                          {parsed.level}
                        </span>
                      </div>
                      <div className="flex justify-center">
                        {parsed.stage && parsed.stage !== '-' && (
                          <span
                            className={cn(
                              'inline-flex w-full justify-center rounded border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide',
                              stageClass(parsed.stage)
                            )}
                          >
                            {parsed.stage}
                          </span>
                        )}
                      </div>
                      <div className="flex flex-col gap-0.5">
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] text-slate-400">{parsed.module}</span>
                          {parsed.jobId && parsed.jobId !== '-' && (
                            <span className="inline-flex rounded border border-sky-500/40 bg-sky-900/40 px-1.5 align-middle text-[9px] text-sky-200">
                              job: {formatJobId(parsed.jobId)}
                            </span>
                          )}
                        </div>
                        <div className="whitespace-pre-wrap break-words leading-5 text-slate-200">
                          {renderMessage(parsed.message)}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
