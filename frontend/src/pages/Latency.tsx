import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { getLeaderboard, type PipelineScore } from "../api/eval";
import { useRun } from "../context/RunContext";

// ── aggregation ────────────────────────────────────────────────────────────

interface PipelineAgg {
  pipeline_id: string;
  latency_score: number;
  composite_score: number;
}

function avg(nums: number[]): number {
  return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
}

function aggregate(scores: PipelineScore[]): PipelineAgg[] {
  const groups: Record<string, PipelineScore[]> = {};
  for (const s of scores) (groups[s.pipeline_id] ??= []).push(s);
  return Object.entries(groups)
    .map(([pipeline_id, rows]) => ({
      pipeline_id,
      latency_score: avg(rows.map((r) => r.latency_score)),
      composite_score: avg(rows.map((r) => r.composite_score)),
    }))
    .sort((a, b) => b.latency_score - a.latency_score);
}

// ── main page ──────────────────────────────────────────────────────────────

export default function Latency() {
  const { currentRunId } = useRun();

  const { data, isLoading, isError } = useQuery({
    queryKey: ["leaderboard", currentRunId],
    queryFn: () => getLeaderboard(currentRunId!),
    enabled: !!currentRunId,
  });

  if (!currentRunId)
    return (
      <div style={{ color: "#6b7280", textAlign: "center", marginTop: 80 }}>
        Load a run on the Leaderboard page first.
      </div>
    );
  if (isLoading) return <div style={{ color: "#6b7280" }}>Loading…</div>;
  if (isError || !data) return <div style={{ color: "#ef4444" }}>Failed to load results.</div>;

  const pipelines = aggregate(data.pipeline_scores);
  const winner = pipelines[0];

  const chartData = pipelines.map((p) => ({
    ...p,
    latency_pct: parseFloat((p.latency_score * 100).toFixed(1)),
    composite_pct: parseFloat((p.composite_score * 100).toFixed(1)),
  }));

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>
      <h2 style={{ margin: "0 0 4px", color: "#111827" }}>Latency</h2>
      <p style={{ margin: "0 0 24px", color: "#6b7280", fontSize: 14 }}>
        Latency score: higher = faster pipeline (normalised across all pipelines in this run).
      </p>

      {/* Chart 1: Latency score */}
      <div
        style={{
          background: "#fff",
          border: "1px solid #e5e7eb",
          borderRadius: 8,
          padding: "20px 16px 8px",
          marginBottom: 20,
        }}
      >
        <div style={{ fontWeight: 600, color: "#111827", marginBottom: 16 }}>
          Latency Score by Pipeline
          <span style={{ fontWeight: 400, fontSize: 12, color: "#9ca3af", marginLeft: 8 }}>
            Higher = faster
          </span>
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={chartData} margin={{ top: 0, right: 16, bottom: 40, left: 0 }}>
            <XAxis
              dataKey="pipeline_id"
              tick={{ fontSize: 11 }}
              angle={-20}
              textAnchor="end"
              interval={0}
            />
            <YAxis
              tickFormatter={(v) => `${v}%`}
              domain={[0, 100]}
              tick={{ fontSize: 12 }}
            />
            <Tooltip formatter={(v: number) => [`${v}%`, "Latency score"]} />
            <Bar dataKey="latency_pct" radius={[4, 4, 0, 0]}>
              {chartData.map((p) => (
                <Cell
                  key={p.pipeline_id}
                  fill={p.pipeline_id === winner?.pipeline_id ? "#6366f1" : "#94a3b8"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Chart 2: Latency vs Composite grouped */}
      <div
        style={{
          background: "#fff",
          border: "1px solid #e5e7eb",
          borderRadius: 8,
          padding: "20px 16px 8px",
        }}
      >
        <div style={{ fontWeight: 600, color: "#111827", marginBottom: 16 }}>
          Latency Score vs Composite Score
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData} margin={{ top: 0, right: 16, bottom: 40, left: 0 }}>
            <XAxis
              dataKey="pipeline_id"
              tick={{ fontSize: 11 }}
              angle={-20}
              textAnchor="end"
              interval={0}
            />
            <YAxis
              tickFormatter={(v) => `${v}%`}
              domain={[0, 100]}
              tick={{ fontSize: 12 }}
            />
            <Tooltip formatter={(v: number, name: string) => [`${v}%`, name]} />
            <Legend
              verticalAlign="top"
              formatter={(value) =>
                value === "latency_pct" ? "Latency Score" : "Composite Score"
              }
            />
            <Bar dataKey="latency_pct" fill="#6366f1" radius={[4, 4, 0, 0]} name="latency_pct" />
            <Bar dataKey="composite_pct" fill="#a5b4fc" radius={[4, 4, 0, 0]} name="composite_pct" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
