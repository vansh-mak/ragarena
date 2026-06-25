import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ResponsiveContainer,
} from "recharts";
import { getLeaderboard, type PipelineScore } from "../api/eval";
import { useRun } from "../context/RunContext";

// ── aggregation helpers ────────────────────────────────────────────────────

interface PipelineAgg {
  pipeline_id: string;
  composite_score: number;
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  context_recall: number;
  judge_correctness: number;
  judge_completeness: number;
  judge_groundedness: number;
  latency_score: number;
}

function avg(nums: number[]): number {
  if (!nums.length) return 0;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function aggregateByPipeline(scores: PipelineScore[]): PipelineAgg[] {
  const groups: Record<string, PipelineScore[]> = {};
  for (const s of scores) {
    (groups[s.pipeline_id] ??= []).push(s);
  }
  return Object.entries(groups)
    .map(([pipeline_id, rows]) => ({
      pipeline_id,
      composite_score: avg(rows.map((r) => r.composite_score)),
      faithfulness: avg(rows.map((r) => r.faithfulness)),
      answer_relevancy: avg(rows.map((r) => r.answer_relevancy)),
      context_precision: avg(rows.map((r) => r.context_precision)),
      context_recall: avg(rows.map((r) => r.context_recall)),
      judge_correctness: avg(rows.map((r) => r.judge_correctness)),
      judge_completeness: avg(rows.map((r) => r.judge_completeness)),
      judge_groundedness: avg(rows.map((r) => r.judge_groundedness)),
      latency_score: avg(rows.map((r) => r.latency_score)),
    }))
    .sort((a, b) => b.composite_score - a.composite_score);
}

// ── sub-components ─────────────────────────────────────────────────────────

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        padding: "16px 20px",
        flex: 1,
        minWidth: 0,
      }}
    >
      <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700, color: "#111827", wordBreak: "break-all" }}>
        {value}
      </div>
    </div>
  );
}

function ProgressBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
        <span style={{ color: "#374151" }}>{label}</span>
        <span style={{ color: "#6b7280" }}>{pct}%</span>
      </div>
      <div style={{ background: "#e5e7eb", borderRadius: 4, height: 8 }}>
        <div
          style={{
            width: `${pct}%`,
            background: "#6366f1",
            borderRadius: 4,
            height: 8,
            transition: "width 0.3s",
          }}
        />
      </div>
    </div>
  );
}

function Skeleton() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          style={{ height: 40, background: "#e5e7eb", borderRadius: 6, animation: "pulse 1.5s infinite" }}
        />
      ))}
    </div>
  );
}

// ── main page ──────────────────────────────────────────────────────────────

export default function Leaderboard() {
  const { currentRunId, setCurrentRunId } = useRun();
  const [inputId, setInputId] = useState(currentRunId ?? "");
  const [selectedPipeline, setSelectedPipeline] = useState<string | null>(null);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["leaderboard", currentRunId],
    queryFn: () => getLeaderboard(currentRunId!),
    enabled: !!currentRunId,
  });

  const pipelines = data ? aggregateByPipeline(data.pipeline_scores) : [];
  const winner = pipelines[0] ?? null;
  const selected = pipelines.find((p) => p.pipeline_id === selectedPipeline) ?? winner;
  const uniqueQueries = data
    ? new Set(data.pipeline_scores.map((s) => s.query_id)).size
    : 0;
  const bestLatency = pipelines.reduce(
    (best, p) => (p.latency_score > (best?.latency_score ?? -1) ? p : best),
    null as PipelineAgg | null
  );
  const bestRelevancy = pipelines.reduce(
    (best, p) => (p.answer_relevancy > (best?.answer_relevancy ?? -1) ? p : best),
    null as PipelineAgg | null
  );

  function handleLoad() {
    const trimmed = inputId.trim();
    if (trimmed) {
      setCurrentRunId(trimmed);
      setSelectedPipeline(null);
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto" }}>
      <h2 style={{ margin: "0 0 20px", color: "#111827" }}>Leaderboard</h2>

      {/* Run selector */}
      <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
        <input
          value={inputId}
          onChange={(e) => setInputId(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleLoad()}
          placeholder="Paste run ID…"
          style={{
            flex: 1,
            padding: "8px 12px",
            border: "1px solid #d1d5db",
            borderRadius: 6,
            fontSize: 14,
            fontFamily: "monospace",
          }}
        />
        <button
          onClick={handleLoad}
          style={{
            padding: "8px 20px",
            background: "#6366f1",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            fontWeight: 600,
          }}
        >
          Load
        </button>
      </div>

      {!currentRunId && (
        <div style={{ color: "#6b7280", textAlign: "center", marginTop: 80 }}>
          Enter a run ID above to load results.
        </div>
      )}

      {currentRunId && isLoading && <Skeleton />}
      {currentRunId && isError && (
        <div style={{ color: "#ef4444" }}>Failed to load results. Check the run ID.</div>
      )}

      {data && pipelines.length > 0 && (
        <>
          {/* Metric cards */}
          <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
            <MetricCard label="Top Pipeline" value={winner?.pipeline_id ?? "—"} />
            <MetricCard label="Total Queries" value={String(uniqueQueries)} />
            <MetricCard label="Best Latency" value={bestLatency?.pipeline_id ?? "—"} />
            <MetricCard label="Best Relevancy" value={bestRelevancy?.pipeline_id ?? "—"} />
          </div>

          {/* Charts row */}
          <div style={{ display: "flex", gap: 16, marginBottom: 24 }}>
            {/* Bar chart — 60% */}
            <div
              style={{
                flex: 3,
                background: "#fff",
                border: "1px solid #e5e7eb",
                borderRadius: 8,
                padding: 20,
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 16, color: "#111827" }}>
                Composite Score by Pipeline
              </div>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart
                  data={pipelines}
                  layout="vertical"
                  margin={{ left: 20, right: 24, top: 0, bottom: 0 }}
                >
                  <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 12 }} />
                  <YAxis type="category" dataKey="pipeline_id" width={110} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
                  <Bar
                    dataKey="composite_score"
                    radius={[0, 4, 4, 0]}
                    cursor="pointer"
                    onClick={(d) => setSelectedPipeline(d.pipeline_id)}
                  >
                    {pipelines.map((p) => (
                      <Cell
                        key={p.pipeline_id}
                        fill={
                          p.pipeline_id === winner?.pipeline_id
                            ? "#6366f1"
                            : p.pipeline_id === selected?.pipeline_id
                            ? "#a5b4fc"
                            : "#d1d5db"
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Detail panel — 40% */}
            <div
              style={{
                flex: 2,
                background: "#fff",
                border: "1px solid #e5e7eb",
                borderRadius: 8,
                padding: 20,
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 4, color: "#111827" }}>
                {selected?.pipeline_id ?? "—"}
              </div>
              <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 16 }}>
                Ragas metrics (avg across queries)
              </div>
              {selected ? (
                <>
                  <ProgressBar label="Faithfulness" value={selected.faithfulness} />
                  <ProgressBar label="Answer Relevancy" value={selected.answer_relevancy} />
                  <ProgressBar label="Context Precision" value={selected.context_precision} />
                  <ProgressBar label="Context Recall" value={selected.context_recall} />
                </>
              ) : (
                <div style={{ color: "#9ca3af", fontSize: 14 }}>Click a bar to inspect.</div>
              )}
            </div>
          </div>

          {/* Score table */}
          <div
            style={{
              background: "#fff",
              border: "1px solid #e5e7eb",
              borderRadius: 8,
              overflow: "hidden",
            }}
          >
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ background: "#f9fafb", borderBottom: "1px solid #e5e7eb" }}>
                  {["Pipeline", "Composite", "Correctness", "Completeness", "Groundedness", "Latency", "Relevancy"].map(
                    (h) => (
                      <th
                        key={h}
                        style={{
                          padding: "10px 14px",
                          textAlign: "left",
                          fontWeight: 600,
                          color: "#374151",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {h}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {pipelines.map((p, i) => (
                  <tr
                    key={p.pipeline_id}
                    style={{
                      background: i === 0 ? "#eef2ff" : i % 2 === 0 ? "#f9fafb" : "#fff",
                      borderBottom: "1px solid #f3f4f6",
                      cursor: "pointer",
                    }}
                    onClick={() => setSelectedPipeline(p.pipeline_id)}
                  >
                    <td style={{ padding: "10px 14px", fontWeight: i === 0 ? 700 : 400, color: "#111827" }}>
                      {i === 0 ? "🥇 " : ""}{p.pipeline_id}
                    </td>
                    <td style={{ padding: "10px 14px", fontWeight: 600, color: "#6366f1" }}>
                      {(p.composite_score * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "10px 14px", color: "#374151" }}>{p.judge_correctness.toFixed(1)}</td>
                    <td style={{ padding: "10px 14px", color: "#374151" }}>{p.judge_completeness.toFixed(1)}</td>
                    <td style={{ padding: "10px 14px", color: "#374151" }}>{p.judge_groundedness.toFixed(1)}</td>
                    <td style={{ padding: "10px 14px", color: "#374151" }}>{(p.latency_score * 100).toFixed(0)}%</td>
                    <td style={{ padding: "10px 14px", color: "#374151" }}>{(p.answer_relevancy * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
