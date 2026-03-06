import {
  Fragment,
  useEffect,
  useState,
  useRef,
  useCallback,
  useMemo,
} from "react";
import type { ReactElement } from "react";
import { useParams, Link } from "react-router-dom";
import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import {
  getJob,
  getJobResult,
  getJobEvents,
  getJobArtifacts,
  getJobVideoUrl,
  exportResultsCSV,
  copyToClipboard,
} from "../lib/api";
import type {
  JobStatus,
  ResultRow,
  JobArtifacts,
  SignalVectorPlot,
  SignalVectorPlotPoint,
} from "../lib/api";
import {
  ArrowLeftIcon,
  FileTextIcon,
  MagicWandIcon,
  DownloadIcon,
  CheckCircledIcon,
  ExclamationTriangleIcon,
  CopyIcon,
} from "@radix-ui/react-icons";
import { HelpTooltip } from "../components/HelpTooltip";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function toApiUrl(url?: string | null): string {
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  return `${API_BASE}${url}`;
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function formatMatchMethod(value: unknown): string {
  if (typeof value !== "string") return "";
  const trimmed = value.trim();
  if (!trimmed) return "";
  const normalized = trimmed.toLowerCase();
  if (normalized === "none" || normalized === "pending") return "";
  return normalized
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatSummaryMatch(method: unknown, score: unknown): string {
  if (typeof method !== "string") return "—";
  const normalized = method.trim().toLowerCase();
  if (!normalized || normalized === "none" || normalized === "pending")
    return "—";

  const label =
    normalized === "semantic"
      ? "Semantic"
      : normalized === "exact"
        ? "Exact"
        : normalized === "embeddings"
          ? "Embed."
          : normalized === "vision"
            ? "Vision"
            : formatMatchMethod(method) || "—";

  const scoreValue = toNumber(score);
  if (scoreValue === null) return label;
  return `${label} (${scoreValue.toFixed(2)})`;
}

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await copyToClipboard(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <button
      onClick={handleCopy}
      title={`Copy ${label}`}
      className="flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider rounded border transition-colors bg-gray-100 border-gray-300 text-gray-600 hover:bg-gray-200 hover:text-gray-900 active:scale-95"
    >
      <CopyIcon className="w-3 h-3" />
      {copied ? "Copied!" : label}
    </button>
  );
}

type ArtifactTab = "video" | "signals" | "ocr" | "frames";
type VideoSource = { type: "local" | "youtube" | "remote"; url: string };
type ScratchTool = "OCR" | "SEARCH" | "VISION" | "FINAL" | "ERROR";
type ReasoningTermType = "brand" | "url" | "evidence";
type ReasoningTerm = { text: string; type: ReasoningTermType };
type HighlightedReasoningPart =
  | string
  | { text: string; type: ReasoningTermType };
type FrameConfidenceTone = {
  stripClass: string;
  badgeClass: string;
  textLabel: string;
};

const PIPELINE_STAGES = [
  "claim",
  "ingest",
  "frame_extract",
  "ocr",
  "vision",
  "llm",
  "persist",
  "completed",
] as const;
const AGENT_STAGES = [
  "claim",
  "ingest",
  "frame_extract",
  "ocr",
  "vision",
  "llm",
  "persist",
  "completed",
] as const;
const COMMON_SIGNAL_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "has",
  "have",
  "he",
  "her",
  "his",
  "in",
  "is",
  "it",
  "its",
  "of",
  "on",
  "or",
  "she",
  "so",
  "the",
  "their",
  "them",
  "then",
  "there",
  "they",
  "this",
  "to",
  "too",
  "was",
  "we",
  "were",
  "what",
  "when",
  "where",
  "which",
  "who",
  "will",
  "with",
  "would",
  "but",
  "if",
  "not",
  "no",
  "yes",
  "that",
  "than",
  "also",
  "been",
  "being",
  "both",
  "each",
  "had",
  "may",
  "most",
  "must",
  "likely",
  "likely meant",
  "however",
  "therefore",
  "thus",
]);

function extractFrameTimestampKey(frame: {
  timestamp?: number | null;
  label?: string;
}): string | null {
  if (typeof frame.timestamp === "number" && Number.isFinite(frame.timestamp)) {
    return frame.timestamp.toFixed(1);
  }
  if (typeof frame.label === "string") {
    const match = frame.label.match(/([\d.]+)\s*s/i);
    if (match) {
      const parsed = Number.parseFloat(match[1]);
      if (Number.isFinite(parsed)) return parsed.toFixed(1);
    }
  }
  return null;
}

function formatStageName(stage: string): string {
  return stage.replace(/_/g, " ");
}

function classifyReasoningTerm(term: string, brandText: string): ReasoningTerm {
  const cleanTerm = term.trim();
  const termLower = cleanTerm.toLowerCase();
  const brandLower = brandText.trim().toLowerCase();
  if (brandLower && termLower === brandLower)
    return { text: cleanTerm, type: "brand" };
  if (/\.\w{2,4}$/i.test(cleanTerm)) return { text: cleanTerm, type: "url" };
  return { text: cleanTerm, type: "evidence" };
}

function isValidSignalPill(text: string): boolean {
  const trimmed = text.trim();
  if (trimmed.length > 50) return false;
  if (trimmed.length < 2) return false;
  if (/^[—\-,;:)\.\!\?'’]/.test(trimmed)) return false;
  if (/[,;:\('’]$/.test(trimmed)) return false;
  if (/[\[\]]/.test(trimmed)) return false;
  const wordCount = trimmed.split(/\s+/).length;
  if (wordCount > 10) return false;
  if (COMMON_SIGNAL_WORDS.has(trimmed.toLowerCase())) return false;
  return true;
}

function normalizeSignalPillText(text: string): string {
  const trimmed = text.trim().replace(/\s+/g, " ");
  const spacedDomain = trimmed.match(
    /^([a-z0-9-]+)\s+(com|net|org|co|io|ai|ca|us|uk|edu|gov)$/i,
  );
  if (spacedDomain) {
    return `${spacedDomain[1].toLowerCase()}.${spacedDomain[2].toLowerCase()}`;
  }
  return trimmed;
}

function sanitizeInlineReasoningFragment(text: string): string {
  return text
    .replace(/\[[A-Z][A-Z0-9 _-]{1,30}\]/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function getFrameConfidenceTone(score: number | null): FrameConfidenceTone {
  if (score == null) {
    return {
      stripClass: "bg-gray-300",
      badgeClass: "bg-gray-100 text-gray-700 border border-gray-300",
      textLabel: "Unknown",
    };
  }
  if (score >= 0.7) {
    return {
      stripClass: "bg-emerald-500",
      badgeClass: "bg-emerald-100 text-emerald-800 border border-emerald-300",
      textLabel: "High",
    };
  }
  if (score >= 0.4) {
    return {
      stripClass: "bg-amber-500",
      badgeClass: "bg-amber-100 text-amber-800 border border-amber-300",
      textLabel: "Medium",
    };
  }
  return {
    stripClass: "bg-red-400",
    badgeClass: "bg-red-100 text-red-800 border border-red-300",
    textLabel: "Low",
  };
}

function truncateCategory(value: string, max = 20): string {
  const normalized = (value || "").trim();
  if (normalized.length <= max) return normalized;
  return `${normalized.slice(0, max - 1)}…`;
}

function normalizeStage(raw: string): string {
  const lower = raw.toLowerCase().trim();
  const aliases: Record<string, string> = {
    frameextract: "frame_extract",
    "frame extract": "frame_extract",
    complete: "completed",
    done: "completed",
  };
  return aliases[lower] || lower;
}

function reasoningPillClass(type: ReasoningTermType): string {
  if (type === "brand")
    return "bg-gray-200 text-gray-900 font-semibold px-2.5 py-1 rounded-full text-xs";
  if (type === "url")
    return "bg-cyan-50 text-cyan-700 border border-cyan-200 px-2.5 py-1 rounded-full text-xs font-mono";
  return "bg-amber-50 text-amber-700 border border-amber-200 px-2.5 py-1 rounded-full text-xs";
}

function reasoningInlineClass(type: ReasoningTermType): string {
  if (type === "brand")
    return "bg-gray-200 text-gray-900 font-semibold px-1 rounded";
  if (type === "url") return "bg-cyan-50 text-cyan-700 px-1 rounded font-mono";
  return "bg-amber-50 text-amber-700 px-1 rounded";
}

function HelpHeading({
  label,
  help,
}: {
  label: string;
  help?: string;
}) {
  return (
    <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-gray-400 font-bold">
      <span>{label}</span>
      {help ? <HelpTooltip content={help} widthClassName="w-72" /> : null}
    </div>
  );
}

function vectorPointKindLabel(kind?: string): string {
  if (kind === "query") return "Query";
  if (kind === "selected") return "Final category";
  if (kind === "leader") return "Top visual match";
  if (kind === "background") return "Taxonomy backdrop";
  return "Nearby category";
}

function buildSignalPlotOption(plot: SignalVectorPlot | null): EChartsOption {
  const emptyOption: EChartsOption = {
    animation: false,
    xAxis: { show: false, min: -1, max: 1 },
    yAxis: { show: false, min: -1, max: 1 },
    series: [],
  };
  if (!plot || !Array.isArray(plot.points) || plot.points.length === 0) {
    return emptyOption;
  }

  const grouped = {
    query: [] as SignalVectorPlotPoint[],
    selected: [] as SignalVectorPlotPoint[],
    leader: [] as SignalVectorPlotPoint[],
    neighbor: [] as SignalVectorPlotPoint[],
    background: [] as SignalVectorPlotPoint[],
  };

  for (const point of plot.points) {
    if (point.kind === "query") grouped.query.push(point);
    else if (point.kind === "selected") grouped.selected.push(point);
    else if (point.kind === "leader") grouped.leader.push(point);
    else if (point.kind === "background") grouped.background.push(point);
    else grouped.neighbor.push(point);
  }

  const toSeriesData = (points: SignalVectorPlotPoint[]) =>
    points.map((point) => ({
      value: [point.x, point.y],
      pointLabel: point.label,
      categoryId: point.category_id ?? "",
      score: point.score,
      kind: point.kind ?? "neighbor",
    }));

  const fullBounds = plot.full_bounds;
  const focusBounds = plot.focus_bounds || fullBounds;

  return {
    backgroundColor: "transparent",
    animationDuration: 500,
    grid: { left: 12, right: 12, top: 12, bottom: 12 },
    dataZoom: fullBounds
      ? [
          {
            type: "inside",
            xAxisIndex: [0],
            filterMode: "none",
            startValue: focusBounds?.x_min,
            endValue: focusBounds?.x_max,
            zoomOnMouseWheel: true,
            moveOnMouseMove: true,
            moveOnMouseWheel: false,
          },
          {
            type: "inside",
            yAxisIndex: [0],
            filterMode: "none",
            startValue: focusBounds?.y_min,
            endValue: focusBounds?.y_max,
            zoomOnMouseWheel: true,
            moveOnMouseMove: true,
            moveOnMouseWheel: false,
          },
        ]
      : [],
    tooltip: {
      trigger: "item",
      backgroundColor: "rgba(2, 6, 23, 0.96)",
      borderColor: "rgba(56, 189, 248, 0.25)",
      textStyle: { color: "#e2e8f0" },
      formatter: (params: any) => {
        const data = params?.data || {};
        const lines = [
          `<strong>${String(data.pointLabel || "Point")}</strong>`,
          vectorPointKindLabel(data.kind),
        ];
        if (data.categoryId) lines.push(`ID: ${data.categoryId}`);
        if (typeof data.score === "number") lines.push(`Score: ${data.score.toFixed(4)}`);
        return lines.join("<br/>");
      },
    },
    xAxis: {
      type: "value",
      show: false,
      splitLine: { show: false },
      axisLine: { show: false },
      axisTick: { show: false },
      min: fullBounds?.x_min,
      max: fullBounds?.x_max,
      scale: true,
    },
    yAxis: {
      type: "value",
      show: false,
      splitLine: { show: false },
      axisLine: { show: false },
      axisTick: { show: false },
      min: fullBounds?.y_min,
      max: fullBounds?.y_max,
      scale: true,
    },
    series: [
      {
        name: "Nebula",
        type: "scatter",
        symbolSize: 6,
        silent: false,
        data: toSeriesData(grouped.background),
        itemStyle: {
          color: "rgba(71, 85, 105, 0.22)",
          borderColor: "rgba(148, 163, 184, 0.18)",
          borderWidth: 0.5,
        },
        emphasis: { scale: 1.05 },
      },
      {
        name: "Nearby categories",
        type: "scatter",
        symbolSize: 14,
        data: toSeriesData(grouped.neighbor),
        itemStyle: {
          color: "rgba(148, 163, 184, 0.55)",
          borderColor: "rgba(226, 232, 240, 0.35)",
          borderWidth: 1,
        },
        emphasis: { scale: 1.15 },
      },
      {
        name: "Top visual match",
        type: "scatter",
        symbol: "diamond",
        symbolSize: 18,
        data: toSeriesData(grouped.leader),
        itemStyle: {
          color: "#f59e0b",
          borderColor: "#fde68a",
          borderWidth: 2,
        },
        label: {
          show: true,
          position: "top",
          color: "#fef3c7",
          fontSize: 11,
          formatter: (params: any) => params?.data?.pointLabel || "",
        },
      },
      {
        name: "Final category",
        type: "scatter",
        symbol: "roundRect",
        symbolSize: 20,
        data: toSeriesData(grouped.selected),
        itemStyle: {
          color: "#6366f1",
          borderColor: "#c7d2fe",
          borderWidth: 2,
          shadowBlur: 14,
          shadowColor: "rgba(99, 102, 241, 0.45)",
        },
        label: {
          show: true,
          position: "top",
          color: "#e0e7ff",
          fontSize: 11,
          formatter: (params: any) => params?.data?.pointLabel || "",
        },
      },
      {
        name: "Query",
        type: "scatter",
        symbol: "circle",
        symbolSize: 24,
        data: toSeriesData(grouped.query),
        itemStyle: {
          color: "#22d3ee",
          borderColor: "#cffafe",
          borderWidth: 2,
          shadowBlur: 18,
          shadowColor: "rgba(34, 211, 238, 0.45)",
        },
        label: {
          show: true,
          position: "bottom",
          color: "#cffafe",
          fontSize: 11,
          formatter: (params: any) => params?.data?.pointLabel || "",
        },
      },
    ],
  };
}

function parseToolSegment(line: string): {
  tool: ScratchTool | null;
  query: string;
  finalFields: Record<string, string>;
} {
  const toolMatch = line.match(
    /\[TOOL:\s*(OCR|SEARCH|VISION|FINAL|ERROR)\b([^\]]*)\]/i,
  );
  if (!toolMatch) return { tool: null, query: "", finalFields: {} };

  const tool = toolMatch[1].toUpperCase() as ScratchTool;
  const segment = toolMatch[0];
  const queryMatch = segment.match(/query\s*=\s*["']([^"']+)["']/i);

  const finalFields: Record<string, string> = {};
  if (tool === "FINAL") {
    const quoted = /(\w+)\s*=\s*"([^"]*)"/g;
    let match = quoted.exec(segment);
    while (match) {
      finalFields[match[1].toLowerCase()] = match[2].trim();
      match = quoted.exec(segment);
    }
    const unquoted = /(\w+)\s*=\s*([^,\]\|]+)/g;
    match = unquoted.exec(segment);
    while (match) {
      const key = match[1].toLowerCase();
      if (!(key in finalFields)) finalFields[key] = match[2].trim();
      match = unquoted.exec(segment);
    }
  }

  return { tool, query: queryMatch?.[1]?.trim() || "", finalFields };
}

function toolTone(tool: ScratchTool | null): {
  icon: string;
  badge: string;
  border: string;
  text: string;
} {
  switch (tool) {
    case "OCR":
      return {
        icon: "📝",
        badge: "bg-cyan-50 border-cyan-200 text-cyan-700",
        border: "border-cyan-300",
        text: "text-cyan-700",
      };
    case "SEARCH":
      return {
        icon: "🔍",
        badge: "bg-amber-50 border-amber-200 text-amber-700",
        border: "border-amber-300",
        text: "text-amber-700",
      };
    case "VISION":
      return {
        icon: "👁️",
        badge: "bg-fuchsia-50 border-fuchsia-200 text-fuchsia-700",
        border: "border-fuchsia-300",
        text: "text-fuchsia-700",
      };
    case "FINAL":
      return {
        icon: "✅",
        badge: "bg-emerald-50 border-emerald-200 text-emerald-700",
        border: "border-emerald-300",
        text: "text-emerald-700",
      };
    case "ERROR":
      return {
        icon: "❌",
        badge: "bg-red-50 border-red-200 text-red-700",
        border: "border-red-300",
        text: "text-red-700",
      };
    default:
      return {
        icon: "•",
        badge: "bg-gray-100 border-gray-300 text-gray-700",
        border: "border-gray-300",
        text: "text-gray-700",
      };
  }
}

function renderScratchboardEvent(event: string, index: number): ReactElement {
  const lines = event.split("\n");
  let currentTool: ScratchTool | null = null;
  const renderedLines: ReactElement[] = [];

  lines.forEach((rawLine, lineIndex) => {
    const trimmed = rawLine.trim();
    const key = `${index}-${lineIndex}`;

    if (!trimmed) {
      renderedLines.push(<div key={key} className="h-1" />);
      return;
    }

    if (/^---\s*Step\s+\d+\s*---/i.test(trimmed)) {
      renderedLines.push(
        <div
          key={key}
          className="text-gray-400 uppercase tracking-wider text-[10px] border-b border-gray-200 pb-1 mb-2 mt-4"
        >
          {trimmed}
        </div>,
      );
      return;
    }

    if (trimmed.includes("✅ FINAL CONCLUSION")) {
      renderedLines.push(
        <div
          key={key}
          className="bg-emerald-50 border border-emerald-200 rounded px-3 py-2 text-emerald-700 font-semibold"
        >
          {trimmed}
        </div>,
      );
      return;
    }

    if (/^🤔\s*Thought:/i.test(trimmed)) {
      renderedLines.push(
        <div key={key} className="italic text-gray-400">
          {trimmed}
        </div>,
      );
      return;
    }

    if (/^Action:/i.test(trimmed)) {
      const actionText = trimmed.replace(/^Action:\s*/i, "");
      const parsed = parseToolSegment(actionText);
      if (parsed.tool) currentTool = parsed.tool;
      const tone = toolTone(parsed.tool);
      const trailingText = actionText.replace(/\[TOOL:[^\]]+\]/i, "").trim();

      renderedLines.push(
        <div key={key} className="text-gray-700 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-semibold text-gray-800">Action:</span>
            {parsed.tool ? (
              <span
                className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${tone.badge}`}
              >
                <span>{tone.icon}</span>
                <span>{parsed.tool}</span>
              </span>
            ) : (
              <span className="text-gray-700">{actionText}</span>
            )}
            {trailingText && (
              <span className="text-gray-500">{trailingText}</span>
            )}
            {parsed.tool === "SEARCH" && parsed.query && (
              <span className="bg-amber-50 text-amber-700 border border-amber-200 px-2 py-0.5 rounded text-[10px] font-mono">
                {parsed.query}
              </span>
            )}
          </div>
          {parsed.tool === "FINAL" &&
            Object.keys(parsed.finalFields).length > 0 && (
              <div className="ml-6 grid gap-1 text-[10px] text-emerald-700">
                {parsed.finalFields.brand && (
                  <div>
                    <span className="text-gray-400 uppercase mr-1">Brand:</span>
                    {parsed.finalFields.brand}
                  </div>
                )}
                {parsed.finalFields.category && (
                  <div>
                    <span className="text-gray-400 uppercase mr-1">
                      Category:
                    </span>
                    {parsed.finalFields.category}
                  </div>
                )}
                {parsed.finalFields.reason && (
                  <div>
                    <span className="text-gray-400 uppercase mr-1">
                      Reason:
                    </span>
                    {parsed.finalFields.reason}
                  </div>
                )}
              </div>
            )}
        </div>,
      );
      return;
    }

    if (/^(Result:|Observation:)/i.test(trimmed)) {
      const parsed = parseToolSegment(trimmed);
      const tone = toolTone(parsed.tool || currentTool);
      renderedLines.push(
        <div
          key={key}
          className={`ml-2 pl-3 border-l-2 ${tone.border} text-gray-500 whitespace-pre-wrap`}
        >
          {trimmed}
        </div>,
      );
      return;
    }

    renderedLines.push(
      <div key={key} className="text-gray-700 whitespace-pre-wrap">
        {rawLine}
      </div>,
    );
  });

  return (
    <div className="border-b border-gray-200 pb-2 mb-2 last:border-0">
      {renderedLines}
    </div>
  );
}

export function JobDetail() {
  const { id } = useParams<{ id: string }>();
  const [job, setJob] = useState<JobStatus | null>(null);
  const [result, setResult] = useState<ResultRow[] | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [artifacts, setArtifacts] = useState<JobArtifacts | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");
  const [artifactTab, setArtifactTab] = useState<ArtifactTab>("signals");
  const [vectorSpace, setVectorSpace] = useState<"mapper" | "visual">("mapper");
  const [videoSource, setVideoSource] = useState<VideoSource | null>(null);
  const [videoAvailable, setVideoAvailable] = useState(false);
  const [videoError, setVideoError] = useState("");
  const [showAllReasoningTerms, setShowAllReasoningTerms] = useState(false);
  const [showFullReasoning, setShowFullReasoning] = useState(false);

  const scratchboardRef = useRef<HTMLDivElement>(null);
  const historyRef = useRef<HTMLDivElement>(null);
  const autoSelectVideoRef = useRef(false);
  const firstRow = result?.[0];
  const brandText =
    typeof firstRow?.Brand === "string" ? firstRow.Brand.trim() : "";
  const reasoningRaw = firstRow
    ? (firstRow.Reasoning ??
      (firstRow as any).reasoning ??
      firstRow["Reasoning"])
    : "";
  const reasoningText =
    typeof reasoningRaw === "string" ? reasoningRaw.trim() : "";
  const isRecoveredReasoning = reasoningText
    .toLowerCase()
    .startsWith("(recovered)");
  const reasoningDisplayText = useMemo(() => {
    if (!reasoningText) return "";
    if (showFullReasoning || reasoningText.length <= 500) return reasoningText;
    return `${reasoningText.slice(0, 220).trimEnd()}...`;
  }, [reasoningText, showFullReasoning]);
  const quotedTermsAll = useMemo<ReasoningTerm[]>(() => {
    if (!reasoningText) return [];
    const regex = /'([^']+)'/g;
    const seen = new Set<string>();
    const orderedTerms: string[] = [];
    let match = regex.exec(reasoningText);
    while (match) {
      const clean = match[1].trim();
      const key = clean.toLowerCase();
      if (clean && !seen.has(key)) {
        seen.add(key);
        orderedTerms.push(clean);
      }
      match = regex.exec(reasoningText);
    }
    const canonicalMap = new Map<string, string>();
    orderedTerms.filter(isValidSignalPill).forEach((term) => {
      const normalized = normalizeSignalPillText(term);
      const key = normalized.toLowerCase();
      const existing = canonicalMap.get(key);
      if (!existing) {
        canonicalMap.set(key, normalized);
        return;
      }
      if (
        existing === existing.toUpperCase() &&
        normalized !== normalized.toUpperCase()
      ) {
        canonicalMap.set(key, normalized);
      }
    });

    return Array.from(canonicalMap.values()).map((term) =>
      classifyReasoningTerm(term, brandText),
    );
  }, [reasoningText, brandText]);
  const visibleQuotedTerms = showAllReasoningTerms
    ? quotedTermsAll
    : quotedTermsAll.slice(0, 6);
  const hiddenQuotedTermsCount = Math.max(
    0,
    quotedTermsAll.length - visibleQuotedTerms.length,
  );
  const highlightedReasoning = useMemo<HighlightedReasoningPart[]>(() => {
    if (!reasoningDisplayText) return [];
    if (quotedTermsAll.length === 0) return [reasoningDisplayText];
    const termType = new Map<string, ReasoningTermType>();
    quotedTermsAll.forEach((term) =>
      termType.set(term.text.toLowerCase(), term.type),
    );

    const parts: HighlightedReasoningPart[] = [];
    const regex = /'([^']+)'/g;
    let lastIndex = 0;
    let match = regex.exec(reasoningDisplayText);
    while (match) {
      if (match.index > lastIndex) {
        parts.push(reasoningDisplayText.slice(lastIndex, match.index));
      }
      const term = match[1];
      const normalizedTerm = normalizeSignalPillText(term);
      const termKind = termType.get(normalizedTerm.toLowerCase());
      if (termKind) {
        parts.push({ text: normalizedTerm, type: termKind });
      } else {
        const sanitized = sanitizeInlineReasoningFragment(term);
        if (sanitized) parts.push(sanitized);
      }
      lastIndex = regex.lastIndex;
      match = regex.exec(reasoningDisplayText);
    }
    if (lastIndex < reasoningDisplayText.length) {
      parts.push(reasoningDisplayText.slice(lastIndex));
    }
    return parts;
  }, [reasoningDisplayText, quotedTermsAll]);
  const ocrText = artifacts?.ocr_text?.text || "";
  const agentScratchboardEvents = useMemo(
    () =>
      events
        .filter((evt) => evt.includes(" agent:\n") || evt.includes(" agent: "))
        .map((evt) => {
          if (evt.includes(" agent:\n"))
            return evt.split(" agent:\n")[1] ?? evt;
          if (evt.includes(" agent: ")) return evt.split(" agent: ")[1] ?? evt;
          return evt;
        }),
    [events],
  );
  const ocrByTimestamp = useMemo(() => {
    const map = new Map<string, string>();
    for (const line of (ocrText || "").split("\n")) {
      const match = line.match(/^\[([\d.]+)s\]\s*(.*)$/);
      if (!match) continue;
      const ts = Number.parseFloat(match[1]);
      if (!Number.isFinite(ts)) continue;
      map.set(ts.toFixed(1), match[2] || "");
    }
    return map;
  }, [ocrText]);
  const stageSequenceForEvents =
    job?.mode === "agent" ? AGENT_STAGES : PIPELINE_STAGES;
  const stageMessages = useMemo(() => {
    const map = new Map<string, string>();
    for (const evt of events) {
      const withoutTimestamp = evt.replace(
        /^\d{4}-\d{2}-\d{2}T[\d:.+\-]+Z?\s*/,
        "",
      );
      const colonIdx = withoutTimestamp.indexOf(":");
      if (colonIdx <= 0) continue;
      const stageRaw = withoutTimestamp.slice(0, colonIdx).trim();
      const stage = normalizeStage(stageRaw);
      if (
        !stageSequenceForEvents.includes(
          stage as (typeof stageSequenceForEvents)[number],
        )
      )
        continue;
      const detail = withoutTimestamp.slice(colonIdx + 1).trim();
      if (!detail) continue;
      map.set(stage, detail);
    }
    return map;
  }, [events, stageSequenceForEvents]);

  const updateVideoSource = useCallback((currentJob: JobStatus) => {
    const rawUrl = (currentJob.url || "").trim();
    if (!rawUrl) {
      setVideoSource(null);
      setVideoAvailable(false);
      return;
    }

    const youtubeMatch = rawUrl.match(
      /(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w-]+)/i,
    );
    if (youtubeMatch?.[1]) {
      setVideoSource({
        type: "youtube",
        url: `https://www.youtube.com/embed/${youtubeMatch[1]}`,
      });
      setVideoAvailable(true);
      return;
    }

    if (rawUrl.startsWith("http://") || rawUrl.startsWith("https://")) {
      setVideoSource({ type: "remote", url: rawUrl });
      setVideoAvailable(true);
      return;
    }

    setVideoSource({ type: "local", url: getJobVideoUrl(currentJob.job_id) });
    setVideoAvailable(true);
  }, []);

  const refreshJobSnapshot = useCallback(
    async (forceTerminalFetch = false) => {
      if (!id) return null;

      const currentJob = await getJob(id);
      setJob(currentJob);
      setError("");
      setVideoError("");
      updateVideoSource(currentJob);

      const isTerminal =
        currentJob.status === "completed" || currentJob.status === "failed";

      if (isTerminal || forceTerminalFetch) {
        try {
          const resultPayload = await getJobResult(id);
          setResult(resultPayload.result || null);
        } catch {
          // no-op
        }
      }

      try {
        const artifactsPayload = await getJobArtifacts(id);
        setArtifacts(artifactsPayload.artifacts || null);
      } catch {
        // no-op
      }

      if (
        currentJob.status === "processing" ||
        isTerminal ||
        forceTerminalFetch
      ) {
        try {
          const eventsPayload = await getJobEvents(id);
          setEvents(eventsPayload.events || []);
        } catch {
          // no-op
        }
      }

      return currentJob;
    },
    [id, updateVideoSource],
  );

  useEffect(() => {
    setShowAllReasoningTerms(false);
    setShowFullReasoning(false);
  }, [reasoningText]);

  useEffect(() => {
    if (videoAvailable && !autoSelectVideoRef.current) {
      setArtifactTab("video");
      autoSelectVideoRef.current = true;
    }
  }, [videoAvailable]);

  useEffect(() => {
    autoSelectVideoRef.current = false;
    setArtifactTab("signals");
  }, [id]);

  useEffect(() => {
    if (scratchboardRef.current) {
      scratchboardRef.current.scrollTop = scratchboardRef.current.scrollHeight;
    }
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [events]);

  useEffect(() => {
    let cancelled = false;
    if (!id) return;

    setLoading(true);
    setJob(null);
    setResult(null);
    setEvents([]);
    setArtifacts(null);
    setVideoSource(null);
    setVideoAvailable(false);

    refreshJobSnapshot(true)
      .catch((err: any) => {
        if (cancelled) return;
        setError(err.message || "Failed to load job");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [id, refreshJobSnapshot]);

  useEffect(() => {
    if (!id || !job) return;
    if (job.status === "completed" || job.status === "failed") return;

    const streamUrl = `${API_BASE}/jobs/${id}/stream`;
    const eventSource = new EventSource(streamUrl);
    let closed = false;

    eventSource.addEventListener("update", (evt) => {
      try {
        const parsed = JSON.parse((evt as MessageEvent).data);
        setJob((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            status: parsed.status ?? prev.status,
            stage: parsed.stage ?? prev.stage,
            stage_detail: parsed.stage_detail ?? prev.stage_detail,
            progress:
              typeof parsed.progress === "number"
                ? parsed.progress
                : prev.progress,
            error: parsed.error ?? prev.error,
            updated_at: parsed.updated_at ?? prev.updated_at,
          };
        });
        if (Array.isArray(parsed.events)) {
          setEvents(parsed.events);
        }
      } catch {
        // no-op
      }
    });

    eventSource.addEventListener("complete", () => {
      if (closed) return;
      closed = true;
      eventSource.close();
      refreshJobSnapshot(true).catch(() => {
        // no-op
      });
    });

    eventSource.onerror = () => {
      if (closed) return;
      closed = true;
      eventSource.close();
    };

    return () => {
      closed = true;
      eventSource.close();
    };
  }, [id, job?.status, refreshJobSnapshot]);

  const handleExportCSV = useCallback(() => {
    if (result) exportResultsCSV(result, `job-${id}-results.csv`);
  }, [result, id]);

  if (loading && !job) {
    return (
      <div className="p-8 text-gray-500 flex items-center gap-2 animate-pulse">
        Loading job…
      </div>
    );
  }

  if (error && !job) {
    return (
      <div className="p-8 text-red-700 bg-red-50 border border-red-200 rounded-lg max-w-xl mx-auto flex flex-col items-center py-12">
        <ExclamationTriangleIcon className="w-12 h-12 mb-4" />
        <h2 className="text-xl font-bold mb-2">Could Not Load Job</h2>
        <p className="text-red-600 mb-6 text-sm text-center">{error}</p>
        <Link
          to="/jobs"
          className="text-sm text-gray-700 bg-gray-100 hover:bg-gray-200 px-4 py-2 rounded transition-colors flex items-center gap-2"
        >
          <ArrowLeftIcon /> Back to Jobs
        </Link>
      </div>
    );
  }

  if (!job) return null;

  const progressPercent = Math.round(job.progress ?? 0);

  const frameItems = artifacts?.latest_frames || [];
  const perFrameVision = Array.isArray(artifacts?.per_frame_vision)
    ? artifacts.per_frame_vision
    : [];
  const visionBoard = artifacts?.vision_board;
  const mapperArtifact =
    artifacts && typeof artifacts.category_mapper === "object"
      ? artifacts.category_mapper
      : null;
  const frameCount = frameItems.length;
  const frameVisionByIndex = new Map<
    number,
    { frame_index: number; top_category: string; top_score: number }
  >();
  for (const item of perFrameVision) {
    const index = Number(item?.frame_index);
    const score = toNumber(item?.top_score);
    if (!Number.isFinite(index) || score == null) continue;
    if (!item?.top_category || typeof item.top_category !== "string") continue;
    frameVisionByIndex.set(index, {
      frame_index: index,
      top_category: item.top_category,
      top_score: score,
    });
  }
  let bestFrameIndex: number | null = null;
  let bestScore = -1;
  frameVisionByIndex.forEach((value, index) => {
    if (value.top_score > bestScore) {
      bestScore = value.top_score;
      bestFrameIndex = index;
    }
  });
  const stages = job.mode === "agent" ? AGENT_STAGES : PIPELINE_STAGES;
  const currentStage = (job.stage || "").trim();
  const currentIdx = stages.indexOf(currentStage as (typeof stages)[number]);

  const categoryText =
    typeof firstRow?.Category === "string" ? firstRow.Category.trim() : "";
  const categoryIdRaw =
    firstRow?.["Category ID"] ?? (firstRow as any)?.category_id;
  const categoryIdText =
    typeof categoryIdRaw === "string"
      ? categoryIdRaw.trim()
      : String(categoryIdRaw ?? "").trim();

  const confidenceValue = toNumber(firstRow?.Confidence);
  const confidenceSummaryDisplay =
    confidenceValue === null ? "—" : confidenceValue.toFixed(2);
  const confidenceSummaryTextColor =
    confidenceValue === null
      ? "text-gray-500"
      : confidenceValue >= 0.8
        ? "text-emerald-700"
        : confidenceValue >= 0.5
          ? "text-amber-700"
          : "text-red-700";
  const confidenceSummaryDotColor =
    confidenceValue === null
      ? "bg-gray-400"
      : confidenceValue >= 0.8
        ? "bg-emerald-500"
        : confidenceValue >= 0.5
          ? "bg-amber-500"
          : "bg-red-500";

  const matchMethodRaw = firstRow
    ? (firstRow as any).category_match_method
    : "";
  const summaryMatchDisplay = formatSummaryMatch(
    matchMethodRaw,
    firstRow ? (firstRow as any).category_match_score : null,
  );
  const mapperCategoryText =
    typeof mapperArtifact?.category === "string" && mapperArtifact.category.trim()
      ? mapperArtifact.category.trim()
      : categoryText;
  const mapperCategoryIdText =
    typeof mapperArtifact?.category_id === "string" &&
    mapperArtifact.category_id.trim()
      ? mapperArtifact.category_id.trim()
      : categoryIdText;
  const mapperMethodDisplay =
    formatMatchMethod(mapperArtifact?.method || matchMethodRaw) || "—";
  const mapperScoreValue = toNumber(
    mapperArtifact?.score ?? (firstRow ? (firstRow as any).category_match_score : null),
  );
  const mapperScoreDisplay =
    mapperScoreValue === null ? "—" : mapperScoreValue.toFixed(4);
  const mapperConfidenceValue = toNumber(
    mapperArtifact?.confidence ?? firstRow?.Confidence,
  );
  const mapperConfidenceDisplay =
    mapperConfidenceValue === null ? "—" : mapperConfidenceValue.toFixed(2);
  const summaryFrameDisplay = artifacts ? String(frameCount) : "—";
  const mapperVectorPlot =
    mapperArtifact && typeof mapperArtifact.vector_plot === "object"
      ? mapperArtifact.vector_plot
      : null;
  const visualVectorPlot =
    visionBoard && typeof visionBoard.vector_plot === "object"
      ? visionBoard.vector_plot
      : null;
  const vectorPlotSpaces = [
    mapperVectorPlot ? "mapper" : null,
    visualVectorPlot ? "visual" : null,
  ].filter(Boolean) as Array<"mapper" | "visual">;
  const effectiveVectorSpace =
    vectorPlotSpaces.includes(vectorSpace)
      ? vectorSpace
      : vectorPlotSpaces[0] || "mapper";
  const activeVectorPlot =
    effectiveVectorSpace === "visual" ? visualVectorPlot : mapperVectorPlot;
  const vectorPlotOption = buildSignalPlotOption(activeVectorPlot);

  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-24 animate-in fade-in duration-500">
      <div className="flex items-center gap-4 text-sm text-gray-500 mb-2">
        <Link
          to="/jobs"
          className="hover:text-primary-600 flex items-center gap-1 transition-colors"
        >
          <ArrowLeftIcon /> Jobs
        </Link>
        <span>/</span>
        <span className="font-mono text-gray-700 truncate max-w-sm">
          {job.job_id}
        </span>
      </div>

      <div className="bg-white border border-gray-200 rounded-xl p-8 shadow-sm flex flex-col gap-6 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gray-100">
          <div
            className="h-full bg-gradient-to-r from-primary-500 to-primary-400 transition-all duration-1000 ease-in-out"
            style={{ width: `${progressPercent}%` }}
          />
        </div>

        <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="text-3xl font-bold text-gray-900 tracking-tight flex items-center gap-3">
                {job.mode === "agent" ? (
                  <MagicWandIcon className="text-primary-500" />
                ) : (
                  <FileTextIcon className="text-primary-500" />
                )}
                {job.mode.charAt(0).toUpperCase() + job.mode.slice(1)} Job
              </h1>
              {(() => {
                const statusClass =
                  job.status === "completed"
                    ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                    : job.status === "failed"
                      ? "bg-red-50 text-red-700 border-red-200"
                      : job.status === "processing"
                        ? "bg-blue-50 text-blue-700 border-blue-200 animate-pulse"
                        : job.status === "re-queued"
                          ? "bg-orange-50 text-orange-700 border-orange-200"
                          : "bg-amber-50 text-amber-700 border-amber-200";
                const statusLabel =
                  job.status === "re-queued"
                    ? "waiting (recovered)"
                    : job.status;
                return (
                  <span
                    className={`px-2.5 py-1 rounded-md text-xs font-semibold uppercase tracking-wider border backdrop-blur-md ${
                      statusClass
                    }`}
                  >
                    {statusLabel}{" "}
                    {job.status === "processing" && `${progressPercent}%`}
                  </span>
                );
              })()}
            </div>

            <div className="flex flex-wrap gap-2">
              <CopyButton text={job.job_id} label="Copy Job ID" />
              {result && (
                <>
                  <CopyButton
                    text={JSON.stringify(result, null, 2)}
                    label="Copy JSON"
                  />
                  <button
                    onClick={handleExportCSV}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider rounded border transition-colors bg-emerald-100 border-emerald-300 text-emerald-700 hover:bg-emerald-200 active:scale-95"
                  >
                    <DownloadIcon className="w-3 h-3" /> Export CSV
                  </button>
                </>
              )}
            </div>

            <div className="text-sm text-gray-500 break-all max-w-3xl font-mono opacity-80 bg-gray-50/80 p-2 rounded border border-gray-200">
              {job.url}
            </div>

            {job.status !== "completed" && job.status !== "failed" && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                <div className="bg-gray-100 border border-gray-200 rounded p-3">
                  <div className="uppercase tracking-wider text-gray-400 mb-1">
                    Current Stage
                  </div>
                  <div className="text-gray-800 font-mono">
                    {job.stage || "unknown"}
                  </div>
                </div>
                <div className="bg-gray-100 border border-gray-200 rounded p-3">
                  <div className="uppercase tracking-wider text-gray-400 mb-1">
                    Stage Detail
                  </div>
                  <div className="text-gray-700">{job.stage_detail || "—"}</div>
                </div>
              </div>
            )}
          </div>

          <div className="flex flex-col items-end gap-1 text-sm text-gray-400 shrink-0">
            <span className="flex items-center gap-2 bg-gray-50 px-3 py-1.5 rounded-md border border-gray-200 shadow-sm">
              Created:{" "}
              <span className="text-gray-700 font-mono text-xs">
                {job.created_at}
              </span>
            </span>
            <span className="flex items-center gap-2 bg-gray-50 px-3 py-1.5 rounded-md border border-gray-200 shadow-sm">
              Updated:{" "}
              <span className="text-gray-700 font-mono text-xs">
                {job.updated_at}
              </span>
            </span>
          </div>
        </div>

        {job.error && (
          <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg text-sm shadow-inner flex flex-col gap-2">
            <div className="flex items-center gap-2 font-bold">
              <ExclamationTriangleIcon /> Execution Failure
            </div>
            <pre className="font-mono text-xs whitespace-pre-wrap px-2 opacity-80">
              {job.error}
            </pre>
          </div>
        )}
      </div>

      {job.status === "completed" && firstRow && firstRow.Brand !== "Err" && (
        <div className="bg-emerald-50/50 border border-emerald-200 rounded-xl px-6 py-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-3 md:gap-0">
          <div className="flex items-center gap-3 min-w-0">
            <CheckCircledIcon className="w-5 h-5 text-emerald-600 shrink-0" />
            <span
              title={brandText || "Unknown Brand"}
              className="text-lg md:text-xl font-bold text-gray-900 max-w-[14rem] md:max-w-xs truncate"
            >
              {brandText || "Unknown Brand"}
            </span>
            <span className="text-gray-400 shrink-0">→</span>
            <span
              title={categoryText || "Unknown Category"}
              className="text-lg md:text-xl font-bold text-emerald-700 max-w-[14rem] md:max-w-sm truncate"
            >
              {categoryText || "Unknown Category"}
            </span>
            {categoryIdText && (
              <span className="text-[10px] font-mono text-gray-400 bg-gray-100 px-2 py-0.5 rounded shrink-0">
                ID: {categoryIdText}
              </span>
            )}
          </div>

          <div className="flex items-stretch self-stretch md:self-auto shrink-0">
            <div className="px-3 md:px-4 border-l border-gray-300/70 text-center">
              <div className="text-[9px] uppercase tracking-wider text-gray-400 mb-0.5">
                Confidence
              </div>
              <div
                className={`inline-flex items-center justify-center gap-1 text-sm font-bold ${confidenceSummaryTextColor}`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${confidenceSummaryDotColor}`}
                  aria-hidden
                />
                <span>{confidenceSummaryDisplay}</span>
              </div>
            </div>
            <div className="px-3 md:px-4 border-l border-gray-300/70 text-center">
              <div className="text-[9px] uppercase tracking-wider text-gray-400 mb-0.5">
                Match
              </div>
              <div className="text-sm font-mono text-cyan-700">
                {summaryMatchDisplay}
              </div>
            </div>
            <div className="px-3 md:px-4 border-l border-gray-300/70 text-center">
              <div className="text-[9px] uppercase tracking-wider text-gray-400 mb-0.5">
                Frames
              </div>
              <div className="text-sm font-mono text-gray-700">
                {summaryFrameDisplay}
              </div>
            </div>
          </div>
        </div>
      )}

      {firstRow && firstRow.Brand !== "Err" && (
        <div className="animate-in slide-in-from-bottom-4 duration-500 fill-mode-forwards">
          <div className="bg-white border border-gray-200 border-l-[3px] border-l-primary-500 rounded-xl p-6">
            <div className="flex items-center justify-between gap-3 mb-3">
              <h3 className="text-xs uppercase tracking-wider text-gray-400 font-bold">
                💡 LLM Reasoning
              </h3>
              <CopyButton
                text={reasoningText || "No reasoning provided by the LLM."}
                label="Copy Reasoning"
              />
            </div>

            {isRecoveredReasoning && (
              <div className="mb-3 inline-flex items-center gap-1 text-amber-700 border border-amber-200 bg-amber-50 rounded px-2 py-1 text-xs">
                <span>🔍</span>
                <span>Web-assisted recovery</span>
              </div>
            )}

            {visibleQuotedTerms.length > 0 && (
              <div className="mb-4">
                <div className="flex flex-wrap gap-2">
                  {visibleQuotedTerms.map((term, idx) => (
                    <span
                      key={`${term.text}-${idx}`}
                      role="status"
                      className={reasoningPillClass(term.type)}
                    >
                      {term.text}
                    </span>
                  ))}
                  {hiddenQuotedTermsCount > 0 && !showAllReasoningTerms && (
                    <button
                      type="button"
                      onClick={() => setShowAllReasoningTerms(true)}
                      className="px-2.5 py-1 rounded-full text-xs border border-gray-300 text-gray-700 bg-gray-100 hover:bg-gray-200 transition-colors"
                    >
                      +{hiddenQuotedTermsCount} more
                    </button>
                  )}
                  {quotedTermsAll.length > 6 && showAllReasoningTerms && (
                    <button
                      type="button"
                      onClick={() => setShowAllReasoningTerms(false)}
                      className="px-2.5 py-1 rounded-full text-xs border border-gray-300 text-gray-700 bg-gray-100 hover:bg-gray-200 transition-colors"
                    >
                      Show less
                    </button>
                  )}
                </div>
                <div className="border-b border-gray-200 mt-4" />
              </div>
            )}

            {reasoningText ? (
              <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                {highlightedReasoning.map((part, idx) =>
                  typeof part === "string" ? (
                    <span key={idx}>{part}</span>
                  ) : (
                    <span key={idx} className={reasoningInlineClass(part.type)}>
                      {part.text}
                    </span>
                  ),
                )}
              </p>
            ) : (
              <p className="text-gray-400 italic text-sm">
                No reasoning provided by the LLM.
              </p>
            )}

            {reasoningText.length > 500 && (
              <button
                type="button"
                onClick={() => setShowFullReasoning((current) => !current)}
                className="mt-3 text-xs text-cyan-700 hover:text-cyan-800 underline underline-offset-2"
              >
                {showFullReasoning ? "Show less" : "Show more"}
              </button>
            )}
          </div>
        </div>
      )}

      <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-200 bg-gray-50">
          {videoAvailable && videoSource && (
            <button
              type="button"
              onClick={() => setArtifactTab("video")}
              className={`px-3 py-1.5 text-xs rounded border ${artifactTab === "video" ? "bg-primary-600 border-primary-500 text-white" : "bg-white border-gray-300 text-gray-700"}`}
            >
              ▶ Video
            </button>
          )}
          <button
            type="button"
            onClick={() => setArtifactTab("signals")}
            title="Final mapper output and supporting vision matches"
            className={`px-3 py-1.5 text-xs rounded border ${artifactTab === "signals" ? "bg-primary-600 border-primary-500 text-white" : "bg-gray-50 border-gray-200 text-gray-700"}`}
          >
            Signals
          </button>
          <button
            type="button"
            onClick={() => setArtifactTab("ocr")}
            className={`px-3 py-1.5 text-xs rounded border ${artifactTab === "ocr" ? "bg-primary-600 border-primary-500 text-white" : "bg-gray-50 border-gray-200 text-gray-700"}`}
          >
            OCR Output
          </button>
          <button
            type="button"
            onClick={() => setArtifactTab("frames")}
            className={`px-3 py-1.5 text-xs rounded border ${artifactTab === "frames" ? "bg-primary-600 border-primary-500 text-white" : "bg-gray-50 border-gray-200 text-gray-700"}`}
          >
            Latest Frames
          </button>
        </div>

        {artifactTab === "video" && videoSource && (
          <div className="p-4">
            {videoSource.type === "local" && (
              <div className="space-y-3">
                <video
                  controls
                  preload="metadata"
                  className="w-full max-h-[500px] rounded-lg border border-gray-300 bg-black"
                  onError={() =>
                    setVideoError(
                      "Source video could not be streamed (missing file or unavailable).",
                    )
                  }
                >
                  <source src={videoSource.url} />
                  Your browser does not support the video element.
                </video>
                {videoError && (
                  <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded px-3 py-2">
                    {videoError}
                  </div>
                )}
              </div>
            )}
            {videoSource.type === "youtube" && (
              <iframe
                src={videoSource.url}
                className="w-full aspect-video rounded-lg border border-gray-300"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                title="Source video"
              />
            )}
            {videoSource.type === "remote" && (
              <div className="text-center py-12 text-gray-500">
                <p className="mb-3">Video is hosted externally.</p>
                <a
                  href={videoSource.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary-600 hover:text-primary-700 underline text-sm"
                >
                  Open in new tab →
                </a>
              </div>
            )}
          </div>
        )}

        {artifactTab === "signals" && (
          <div className="p-4 space-y-4">
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1.4fr)]">
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <HelpHeading
                      label="Final Mapper Output"
                      help="This is the final canonical taxonomy category after the service maps the model's raw category choice onto the official FreeWheel taxonomy."
                    />
                    <div className="text-xs text-gray-500">
                      Canonical taxonomy category selected after mapping.
                    </div>
                  </div>
                  {mapperCategoryIdText && (
                    <span
                      title={`Canonical taxonomy ID: ${mapperCategoryIdText}`}
                      className="inline-flex items-center px-2 py-1 rounded-full border border-primary-200 bg-primary-50 text-primary-700 text-[11px] font-mono font-semibold leading-none"
                    >
                      ID {mapperCategoryIdText}
                    </span>
                  )}
                </div>
                <div
                  title={mapperCategoryText || "No mapped category available."}
                  className="text-lg font-semibold text-gray-900 break-words"
                >
                  {mapperCategoryText || "No mapped category available."}
                </div>
                <div className="grid grid-cols-3 gap-3 text-xs">
                  <div
                    title={`Mapper method: ${mapperMethodDisplay}`}
                    className="bg-white border border-gray-200 rounded-lg px-3 py-2"
                  >
                    <HelpHeading
                      label="Method"
                      help="How the final category was chosen. For example, Embeddings means the raw category text was matched against taxonomy labels in embedding space."
                    />
                    <div className="font-medium text-gray-800">
                      {mapperMethodDisplay}
                    </div>
                  </div>
                  <div
                    title={
                      mapperScoreValue === null
                        ? "No mapper score available."
                        : `Mapper score: ${mapperScoreValue.toFixed(6)}`
                    }
                    className="bg-white border border-gray-200 rounded-lg px-3 py-2"
                  >
                    <HelpHeading
                      label="Mapper Score"
                      help="Strength of the final taxonomy match. Higher means the mapper considered the chosen canonical category a closer fit to the source label or evidence."
                    />
                    <div className="font-mono text-cyan-700">
                      {mapperScoreDisplay}
                    </div>
                  </div>
                  <div
                    title={
                      mapperConfidenceValue === null
                        ? "No LLM confidence available."
                        : `LLM confidence: ${mapperConfidenceValue.toFixed(6)}`
                    }
                    className="bg-white border border-gray-200 rounded-lg px-3 py-2"
                  >
                    <HelpHeading
                      label="LLM Confidence"
                      help="The classification model's own confidence in the brand/category answer before the final taxonomy mapping step."
                    />
                    <div className="font-mono text-gray-800">
                      {mapperConfidenceDisplay}
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-3">
                <div>
                  <HelpHeading
                    label="Vision Matches"
                    help="Visual encoder category scores from the sampled frames. These are supporting signals only; they help explain what the visual model saw but do not replace the final mapped category."
                  />
                  <div className="text-xs text-gray-500">
                    Raw visual category scores used as supporting evidence.
                  </div>
                </div>
                {(visionBoard?.top_matches || []).length > 0 ? (
                  <div className="grid gap-2">
                    {(visionBoard?.top_matches || []).map((m, idx) => (
                      <div
                        key={idx}
                        title={`${m.label} · score ${Number(m.score).toFixed(6)}${m.category_id != null ? ` · category ID ${m.category_id}` : ""}`}
                        className="flex items-center justify-between text-xs bg-white border border-gray-200 rounded px-3 py-2"
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="text-gray-800 truncate">{m.label}</span>
                          {m.category_id != null && (
                            <span
                              title={`Category ID: ${m.category_id}`}
                              className="shrink-0 inline-flex items-center px-1.5 py-0.5 rounded-full border border-primary-200 bg-primary-50 text-primary-700 text-[10px] font-mono font-semibold leading-none"
                            >
                              #{m.category_id}
                            </span>
                          )}
                        </div>
                        <span className="font-mono text-cyan-700 ml-3 shrink-0">
                          {Number(m.score).toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-gray-500">
                    No vision matches available.
                  </div>
                )}
              </div>
            </div>

            {activeVectorPlot && (
              <div className="rounded-2xl border border-slate-800 bg-slate-950 p-4 shadow-xl">
                <div className="mb-3 flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
                        Vector View
                      </h3>
                      <HelpTooltip
                        content="A local 2D projection of the category neighborhood. Query shows the source signal, final category is the mapped answer, and nearby points are the closest alternatives in that embedding space."
                        widthClassName="w-80"
                      />
                    </div>
                    <p className="text-xs text-slate-400">
                      {activeVectorPlot.subtitle ||
                        "Projected neighborhood around the chosen category."}
                    </p>
                  </div>
                  {vectorPlotSpaces.length > 1 && (
                    <div className="inline-flex rounded-lg border border-slate-700 bg-slate-900 p-1">
                      <button
                        type="button"
                        onClick={() => setVectorSpace("mapper")}
                        className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                          effectiveVectorSpace === "mapper"
                            ? "bg-indigo-500 text-white"
                            : "text-slate-300 hover:bg-slate-800"
                        }`}
                      >
                        Mapper Space
                      </button>
                      <button
                        type="button"
                        onClick={() => setVectorSpace("visual")}
                        className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                          effectiveVectorSpace === "visual"
                            ? "bg-cyan-500 text-slate-950"
                            : "text-slate-300 hover:bg-slate-800"
                        }`}
                      >
                        Visual Space
                      </button>
                    </div>
                  )}
                </div>

                <div className="mb-3 flex flex-wrap gap-2 text-[11px] text-slate-300">
                  <span className="rounded-full border border-cyan-400/40 bg-cyan-400/10 px-2.5 py-1">
                    Query: {activeVectorPlot.query_label || "signal"}
                  </span>
                  <span className="rounded-full border border-indigo-400/40 bg-indigo-400/10 px-2.5 py-1">
                    Final: {activeVectorPlot.selected_label || mapperCategoryText || "—"}
                  </span>
                  {activeVectorPlot.backend && (
                    <span className="rounded-full border border-slate-600 bg-slate-900 px-2.5 py-1">
                      Backend: {activeVectorPlot.backend}
                    </span>
                  )}
                  <span className="rounded-full border border-slate-600 bg-slate-900 px-2.5 py-1 text-slate-400">
                    Scroll to zoom, drag to pan
                  </span>
                </div>

                <ReactECharts
                  option={vectorPlotOption}
                  style={{ height: 360, width: "100%" }}
                  notMerge
                  lazyUpdate
                />
              </div>
            )}

            {visionBoard?.image_url && (
              <img
                src={toApiUrl(visionBoard.image_url)}
                alt="Vision board"
                className="max-h-96 rounded border border-gray-300"
              />
            )}
            {visionBoard?.plot_url && (
              <a
                href={toApiUrl(visionBoard.plot_url)}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-primary-600 underline"
              >
                Open vision board metadata
              </a>
            )}
          </div>
        )}

        {artifactTab === "ocr" && (
          <div className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-xs text-gray-500">OCR output</div>
              <CopyButton text={ocrText} label="Copy OCR" />
            </div>
            <div className="max-h-80 overflow-auto text-xs font-mono whitespace-pre-wrap text-gray-700 bg-gray-50 border border-gray-200 rounded p-3">
              {ocrText || "No OCR text available."}
            </div>
          </div>
        )}

        {artifactTab === "frames" && (
          <div className="p-4">
            {frameItems.length > 0 ? (
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {frameItems.map((frame, idx) => {
                  const frameLabel =
                    frame.label ||
                    (typeof frame.timestamp === "number"
                      ? `${frame.timestamp.toFixed(1)}s`
                      : `Frame ${idx + 1}`);
                  const frameTsKey = extractFrameTimestampKey(frame);
                  const frameOcrText = frameTsKey
                    ? ocrByTimestamp.get(frameTsKey)
                    : "";
                  const frameVision = frameVisionByIndex.get(idx);
                  const frameScore = frameVision
                    ? toNumber(frameVision.top_score)
                    : null;
                  const frameTone = getFrameConfidenceTone(frameScore);
                  const frameCategory = frameVision?.top_category || "";
                  const isBestFrame =
                    bestFrameIndex === idx && frameVision != null;
                  const frameTooltip = frameVision
                    ? `Frame ${idx + 1}: ${frameCategory} (${(frameScore ?? 0).toFixed(2)})`
                    : undefined;
                  return (
                    <div
                      key={idx}
                      className="aspect-video bg-gray-50 rounded border border-gray-200 overflow-hidden relative group"
                    >
                      <img
                        src={toApiUrl(frame.url)}
                        alt={frameLabel}
                        className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                      />
                      {frameVision && (
                        <div
                          className={`absolute inset-y-0 left-0 w-1 ${frameTone.stripClass}`}
                          aria-hidden
                        />
                      )}
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent p-2 text-[10px] font-mono text-emerald-400">
                        {frameLabel}
                      </div>
                      {frameVision && (
                        <div
                          className={`absolute bottom-1 left-2 px-1.5 py-0.5 rounded text-[10px] font-mono ${frameTone.badgeClass}`}
                          title={frameTooltip}
                        >
                          {(frameScore ?? 0).toFixed(2)} ·{" "}
                          {truncateCategory(frameCategory)} ·{" "}
                          {frameTone.textLabel}
                        </div>
                      )}
                      {isBestFrame && (
                        <div className="absolute top-1 right-1 px-1.5 py-0.5 rounded bg-emerald-600 text-white text-[10px] font-semibold">
                          ★ Best
                        </div>
                      )}
                      {frameOcrText && (
                        <div className="absolute inset-0 bg-black/85 opacity-0 group-hover:opacity-100 transition-opacity duration-200 p-3 flex items-center justify-center">
                          <p className="text-[10px] text-cyan-300 font-mono leading-relaxed text-center line-clamp-6">
                            {frameOcrText}
                          </p>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-xs text-gray-500">
                No latest frames available.
              </div>
            )}
          </div>
        )}
      </div>

      {job.mode === "agent" && agentScratchboardEvents.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm flex flex-col animate-in slide-in-from-bottom-4 duration-500 delay-100 fill-mode-forwards">
          <div className="bg-gray-50 px-4 py-3 border-b border-gray-200 font-semibold text-fuchsia-700 flex items-center gap-2">
            <MagicWandIcon className="text-fuchsia-600" /> Agent Scratchboard
          </div>
          <div
            className="p-4 h-96 overflow-y-auto space-y-2 font-mono text-xs text-gray-700"
            ref={scratchboardRef}
          >
            {agentScratchboardEvents.map((evt, i) => (
              <Fragment key={i}>{renderScratchboardEvent(evt, i)}</Fragment>
            ))}
          </div>
        </div>
      )}

      {(events.length > 0 || job.stage) && (
        <div className="bg-gray-50 border border-gray-200 rounded-xl overflow-hidden shadow-inner flex flex-col animate-in slide-in-from-bottom-4 duration-500 delay-100 fill-mode-forwards">
          <div className="px-4 py-4 border-b border-gray-200 bg-gray-50 overflow-x-auto">
            <div className="min-w-[680px] w-full px-1">
              <div className="flex items-center w-full mb-2">
                {stages.map((stage, idx) => {
                  const isDone =
                    currentIdx > idx ||
                    job.status === "completed" ||
                    (job.status === "failed" && currentIdx > idx);
                  const isCurrent =
                    currentIdx === idx && job.status === "processing";
                  const isFailed =
                    job.status === "failed" && currentIdx === idx;
                  const stageLabel = formatStageName(stage);
                  const dotTitle =
                    isCurrent && job.stage_detail
                      ? `${stageLabel}: ${job.stage_detail}`
                      : stageLabel;
                  return (
                    <Fragment key={stage}>
                      {idx > 0 && (
                        <div
                          className={`flex-1 h-0.5 transition-colors duration-500 ${isDone || isCurrent ? "bg-emerald-500" : "bg-gray-100"}`}
                        />
                      )}
                      <div
                        title={dotTitle}
                        className={`w-3 h-3 rounded-full border-2 shrink-0 transition-colors duration-500 ${
                          isDone
                            ? "bg-emerald-500 border-emerald-400"
                            : isCurrent
                              ? "bg-blue-500 border-blue-400 animate-pulse"
                              : isFailed
                                ? "bg-red-500 border-red-400"
                                : "bg-gray-100 border-gray-300"
                        }`}
                      />
                    </Fragment>
                  );
                })}
              </div>

              <div className="flex w-full mb-3">
                {stages.map((stage, idx) => {
                  const isDone =
                    currentIdx > idx ||
                    job.status === "completed" ||
                    (job.status === "failed" && currentIdx > idx);
                  const isCurrent =
                    currentIdx === idx && job.status === "processing";
                  const isFailed =
                    job.status === "failed" && currentIdx === idx;
                  return (
                    <div
                      key={stage}
                      className={`text-[9px] uppercase tracking-wider text-center transition-colors duration-500 px-1 ${
                        idx === 0 ? "w-3 shrink-0" : "flex-1 min-w-0"
                      } ${
                        isDone
                          ? "text-emerald-500"
                          : isCurrent
                            ? "text-blue-400"
                            : isFailed
                              ? "text-red-400"
                              : "text-gray-400"
                      }`}
                    >
                      {stage.replace("_", " ")}
                    </div>
                  );
                })}
              </div>

              <div className="flex w-full border-t border-gray-200 pt-3">
                {stages.map((stage, idx) => {
                  const message = stageMessages.get(stage);
                  const isDone =
                    currentIdx > idx ||
                    job.status === "completed" ||
                    (job.status === "failed" && currentIdx > idx);
                  const isCurrent =
                    currentIdx === idx && job.status === "processing";
                  return (
                    <div
                      key={stage}
                      className={`text-center px-1 ${idx === 0 ? "w-3 shrink-0" : "flex-1 min-w-0"}`}
                    >
                      {message && (
                        <div
                          className={`text-[10px] leading-tight truncate transition-all duration-300 ${
                            isCurrent
                              ? "text-blue-700 font-medium"
                              : isDone
                                ? "text-gray-400"
                                : "text-gray-400"
                          }`}
                          title={message}
                        >
                          {message}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          <details className="bg-gray-50 border-t border-gray-200 overflow-hidden shadow-sm group">
            <summary className="px-6 py-4 font-semibold text-gray-500 group-hover:bg-gray-100/50 transition-colors list-none flex items-center gap-2 cursor-pointer">
              <MagicWandIcon className="text-fuchsia-400" />
              <span>Event History</span>
              <span className="text-xs text-gray-400 font-normal ml-2">
                ({events.length} events)
              </span>
            </summary>
            {events.length > 0 ? (
              <div
                className="p-4 max-h-96 overflow-y-auto space-y-2 font-mono text-xs text-gray-500 border-t border-gray-200"
                ref={historyRef}
              >
                {events.map((evt, i) => (
                  <div
                    key={i}
                    className="border-b border-gray-200 pb-2 mb-2 last:border-0 whitespace-pre-wrap"
                  >
                    {evt}
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 text-xs text-gray-400 border-t border-gray-200">
                No events yet.
              </div>
            )}
          </details>
        </div>
      )}

      <details className="bg-white border border-gray-200 rounded-xl overflow-hidden cursor-pointer shadow-sm group">
        <summary className="px-6 py-4 font-semibold text-gray-500 group-hover:bg-gray-100/50 transition-colors list-none flex items-center gap-2">
          <DownloadIcon /> Raw JSON Context
        </summary>
        <div className="p-6 bg-gray-50 border-t border-gray-200 font-mono text-xs text-emerald-400/70 overflow-x-auto">
          <pre>
            {JSON.stringify(
              { settings: job.settings, result, artifacts },
              null,
              2,
            )}
          </pre>
        </div>
      </details>
    </div>
  );
}
