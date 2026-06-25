import { useQuery } from "@tanstack/react-query";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Label,
} from "recharts";
import { getLeaderboard, type PipelineScore } from "../api/eval";
import { useRun } from "../context/RunContext";

// ── aggregation (avg across queries) ──────────────────────────────────────

interface PipelinePoint {
  pipeline_id: string;
  composite_score: number;
  latency_score: number;
  answer_relevancy: number;
}

function avg(nums: number[]): number {
  return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
}

function aggregate(scores: PipelineScore[]): PipelinePoint[] {
  const groups: Record<string, PipelineScore[]> = {};
  for (const s of scores) (groups[s.pipeline_id] ??= []).push(s);
  return Object.entries(groups)
    .map(([pipeline_id, rows]) => ({
      pipeline_id,
      composite_score: avg(rows.map((r) => r.composite_score)),
      latency_score: avg(rows.map((r) => r.latency_score)),
      answer_relevancy: avg(rows.map((r) => r.answer_relevancy)),
    }))
    .sort((a, b) => b.composite_score - a.composite_score);
}

// ── custom tooltip ─────────────────────────────────────────────────────────

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d: PipelinePoint = payload[0].payload;
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        padding: "10px 14px",
        fontSize: 13,
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
      }}
    >
      <div style={{ fontWeight: 700, color: "#111827", marginBottom: 6 }}>{d.pipeline_id}</div>
      <div style={{ color: "#6b7280" }}>Composite: <b style={{ color: "#6366f1" }}>{(d.composite_score * 100).toFixed(1)}%</b></div>
      <div style={{ color: "#6b7280" }}>Latency score: {(d.latency_score * 100).toFixed(1)}%</div>
      <div style={{ color: "#6b7280" }}>Relevancy: {(d.answer_relevancy * 100).toFixed(1)}%</div>
    </div>
  );
}

// ── quadrant label ─────────────────────────────────────────────────────────

function QuadLabel({
  x, y, text,
}: {
  x: number | string;
  y: number | string;
  text: string;
}) {
  return (
    <text x={x} y={y} fontSize={11} fill="#9ca3af" textAnchor="middle">
      {text}
    </text>
  );
}

// ── main page ──────────────────────────────────────────────────────────────

export default function CostQuality() {
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

  const points = aggregate(data.pipeline_scores);
  const winner = points[0];

  const COLORS = [
    "#6366f1", "#64748b", "#0ea5e9", "#10b981",
    "#f59e0b", "#ef4444", "#8b5cf6",
  ];

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>
      <h2 style={{ margin: "0 0 4px", color: "#111827" }}>Cost vs Quality</h2>
      <p style={{ margin: "0 0 24px", color: "#6b7280", fontSize: 14 }}>
        X = latency score (higher = faster). Y = composite quality score. Top-right is ideal.
      </p>

      <div
        style={{
          background: "#fff",
          border: "1px solid #e5e7eb",
          borderRadius: 8,
          padding: "24px 16px 16px",
          position: "relative",
        }}
      >
        {/* Quadrant labels — positioned absolutely over the chart */}
        <div style={{ position: "relative" }}>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 50 }}>
              <XAxis
                type="number"
                dataKey="latency_score"
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                tick={{ fontSize: 12 }}
              >
                <Label value="Latency Score (higher = faster)" offset={-10} position="insideBottom" fontSize={13} fill="#6b7280" />
              </XAxis>
              <YAxis
                type="number"
                dataKey="composite_score"
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                tick={{ fontSize: 12 }}
              >
                <Label value="Composite Score" angle={-90} position="insideLeft" offset={10} fontSize={13} fill="#6b7280" />
              </YAxis>
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine x={0.5} stroke="#e5e7eb" strokeDasharray="4 4" />
              <ReferenceLine y={0.4} stroke="#e5e7eb" strokeDasharray="4 4" />
              <Scatter
                data={points}
                shape={(props: any) => {
                  const { cx, cy, payload } = props;
                  const isWin = payload.pipeline_id === winner?.pipeline_id;
                  const colorIdx = points.findIndex((p) => p.pipeline_id === payload.pipeline_id);
                  return (
                    <g>
                      <circle
                        cx={cx}
                        cy={cy}
                        r={14}
                        fill={isWin ? "#6366f1" : COLORS[colorIdx % COLORS.length]}
                        opacity={0.85}
                      />
                      <text
                        x={cx}
                        y={cy - 18}
                        textAnchor="middle"
                        fontSize={10}
                        fill="#374151"
                      >
                        {payload.pipeline_id}
                      </text>
                    </g>
                  );
                }}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Legend */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          marginTop: 20,
          background: "#fff",
          border: "1px solid #e5e7eb",
          borderRadius: 8,
          padding: 16,
        }}
      >
        {points.map((p, i) => (
          <div key={p.pipeline_id} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}>
            <div
              style={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                background: p.pipeline_id === winner?.pipeline_id ? "#6366f1" : COLORS[i % COLORS.length],
                flexShrink: 0,
              }}
            />
            <span style={{ color: "#374151" }}>{p.pipeline_id}</span>
            <span style={{ color: "#9ca3af" }}>({(p.composite_score * 100).toFixed(1)}%)</span>
          </div>
        ))}
      </div>
    </div>
  );
}
