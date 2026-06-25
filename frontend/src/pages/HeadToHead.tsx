import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getLeaderboard, type PipelineScore } from "../api/eval";
import { useRun } from "../context/RunContext";

// ── helpers ────────────────────────────────────────────────────────────────

function scoreBadgeColor(score: number): string {
  if (score >= 0.45) return "#16a34a";
  if (score >= 0.35) return "#d97706";
  return "#dc2626";
}

function fmt(n: number | null | undefined, decimals = 2): string {
  if (n == null) return "—";
  return n.toFixed(decimals);
}

// ── AnswerCard ─────────────────────────────────────────────────────────────

function AnswerCard({
  score,
  isWinner,
  onClick,
}: {
  score: PipelineScore;
  isWinner: boolean;
  onClick: () => void;
}) {
  const badge = scoreBadgeColor(score.composite_score);
  return (
    <div
      onClick={onClick}
      style={{
        background: "#fff",
        border: `2px solid ${isWinner ? "#6366f1" : "#e5e7eb"}`,
        borderRadius: 8,
        padding: 16,
        cursor: "pointer",
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontWeight: 700, fontSize: 13, color: "#111827" }}>
          {isWinner ? "🥇 " : ""}
          {score.pipeline_id}
        </span>
        <span
          style={{
            background: badge,
            color: "#fff",
            fontSize: 11,
            fontWeight: 700,
            padding: "2px 8px",
            borderRadius: 12,
          }}
        >
          {(score.composite_score * 100).toFixed(1)}%
        </span>
      </div>

      <div
        style={{
          fontSize: 13,
          color: "#374151",
          lineHeight: 1.5,
          display: "-webkit-box",
          WebkitLineClamp: 4,
          WebkitBoxOrient: "vertical",
          overflow: "hidden",
        }}
      >
        {score.answer || <em style={{ color: "#9ca3af" }}>No answer</em>}
      </div>

      <div style={{ display: "flex", gap: 16, fontSize: 11, color: "#6b7280", marginTop: "auto" }}>
        <span>Correctness: {fmt(score.judge_correctness, 1)}</span>
      </div>
    </div>
  );
}

// ── comparison table ───────────────────────────────────────────────────────

function boldIfMax(val: number, max: number) {
  return val === max
    ? { fontWeight: 700, color: "#6366f1" }
    : { color: "#374151" };
}

function ComparisonTable({ scores }: { scores: PipelineScore[] }) {
  const sorted = [...scores].sort((a, b) => b.composite_score - a.composite_score);
  const maxComposite = Math.max(...sorted.map((s) => s.composite_score));
  const maxFaith = Math.max(...sorted.map((s) => s.faithfulness));
  const maxRel = Math.max(...sorted.map((s) => s.answer_relevancy));
  const maxCorr = Math.max(...sorted.map((s) => s.judge_correctness));
  const maxGround = Math.max(...sorted.map((s) => s.judge_groundedness));
  const maxLat = Math.max(...sorted.map((s) => s.latency_score));

  const cols = ["Pipeline", "Composite", "Faithfulness", "Relevancy", "Correctness", "Groundedness", "Latency"];

  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        overflow: "hidden",
        marginTop: 20,
      }}
    >
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "#f9fafb", borderBottom: "1px solid #e5e7eb" }}>
            {cols.map((c) => (
              <th
                key={c}
                style={{ padding: "10px 14px", textAlign: "left", fontWeight: 600, color: "#374151" }}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((s, i) => (
            <tr
              key={s.pipeline_id}
              style={{
                borderBottom: "1px solid #f3f4f6",
                background: i % 2 === 0 ? "#fff" : "#f9fafb",
              }}
            >
              <td style={{ padding: "10px 14px", fontWeight: 600, color: "#111827" }}>
                {s.pipeline_id}
              </td>
              <td style={{ padding: "10px 14px", ...boldIfMax(s.composite_score, maxComposite) }}>
                {(s.composite_score * 100).toFixed(1)}%
              </td>
              <td style={{ padding: "10px 14px", ...boldIfMax(s.faithfulness, maxFaith) }}>
                {(s.faithfulness * 100).toFixed(1)}%
              </td>
              <td style={{ padding: "10px 14px", ...boldIfMax(s.answer_relevancy, maxRel) }}>
                {(s.answer_relevancy * 100).toFixed(1)}%
              </td>
              <td style={{ padding: "10px 14px", ...boldIfMax(s.judge_correctness, maxCorr) }}>
                {fmt(s.judge_correctness, 1)}
              </td>
              <td style={{ padding: "10px 14px", ...boldIfMax(s.judge_groundedness, maxGround) }}>
                {fmt(s.judge_groundedness, 1)}
              </td>
              <td style={{ padding: "10px 14px", ...boldIfMax(s.latency_score, maxLat) }}>
                {(s.latency_score * 100).toFixed(0)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── main page ──────────────────────────────────────────────────────────────

export default function HeadToHead() {
  const { currentRunId } = useRun();
  const [selectedQueryIndex, setSelectedQueryIndex] = useState(0);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["leaderboard", currentRunId],
    queryFn: () => getLeaderboard(currentRunId!),
    enabled: !!currentRunId,
  });

  if (!currentRunId) {
    return (
      <div style={{ color: "#6b7280", textAlign: "center", marginTop: 80 }}>
        Load a run on the Leaderboard page first.
      </div>
    );
  }
  if (isLoading) return <div style={{ color: "#6b7280" }}>Loading…</div>;
  if (isError || !data) return <div style={{ color: "#ef4444" }}>Failed to load results.</div>;

  // Build ordered list of unique query IDs
  const queryIds = [...new Set(data.pipeline_scores.map((s) => s.query_id))].sort();
  const total = queryIds.length;
  const idx = Math.min(selectedQueryIndex, total - 1);
  const currentQueryId = queryIds[idx];

  const queryScores = data.pipeline_scores.filter((s) => s.query_id === currentQueryId);
  const winner = queryScores.reduce(
    (best, s) => (s.composite_score > (best?.composite_score ?? -1) ? s : best),
    null as PipelineScore | null
  );

  // Representative query text (all pipelines share same query_text per query_id)
  const queryText =
    (data.pipeline_scores.find((s) => s.query_id === currentQueryId) as any)?.query_text ??
    currentQueryId;

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto" }}>
      <h2 style={{ margin: "0 0 20px", color: "#111827" }}>Head to Head</h2>

      {/* Query selector */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          background: "#fff",
          border: "1px solid #e5e7eb",
          borderRadius: 8,
          padding: "12px 16px",
          marginBottom: 20,
        }}
      >
        <button
          disabled={idx === 0}
          onClick={() => setSelectedQueryIndex(idx - 1)}
          style={{
            padding: "4px 12px",
            border: "1px solid #d1d5db",
            borderRadius: 6,
            background: idx === 0 ? "#f3f4f6" : "#fff",
            cursor: idx === 0 ? "not-allowed" : "pointer",
            color: idx === 0 ? "#9ca3af" : "#374151",
          }}
        >
          ‹
        </button>
        <span style={{ fontSize: 13, color: "#6b7280", whiteSpace: "nowrap" }}>
          Q {idx + 1} / {total}
        </span>
        <span style={{ flex: 1, fontSize: 14, color: "#111827", fontStyle: "italic" }}>
          {queryText}
        </span>
        <button
          disabled={idx === total - 1}
          onClick={() => setSelectedQueryIndex(idx + 1)}
          style={{
            padding: "4px 12px",
            border: "1px solid #d1d5db",
            borderRadius: 6,
            background: idx === total - 1 ? "#f3f4f6" : "#fff",
            cursor: idx === total - 1 ? "not-allowed" : "pointer",
            color: idx === total - 1 ? "#9ca3af" : "#374151",
          }}
        >
          ›
        </button>
      </div>

      {/* Answer cards grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 12,
          marginBottom: 4,
        }}
      >
        {queryScores.slice(0, 4).map((s) => (
          <AnswerCard
            key={s.pipeline_id}
            score={s}
            isWinner={s.pipeline_id === winner?.pipeline_id}
            onClick={() => {}}
          />
        ))}
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 12,
        }}
      >
        {queryScores.slice(4).map((s) => (
          <AnswerCard
            key={s.pipeline_id}
            score={s}
            isWinner={s.pipeline_id === winner?.pipeline_id}
            onClick={() => {}}
          />
        ))}
      </div>

      {/* Comparison table */}
      <ComparisonTable scores={queryScores} />

      {/* Insight box */}
      {winner?.insight && (
        <div
          style={{
            background: "#f5f3ff",
            border: "1px solid #c4b5fd",
            borderLeft: "4px solid #6366f1",
            borderRadius: 8,
            padding: "16px 20px",
            marginTop: 20,
            fontSize: 14,
            color: "#1e1b4b",
            lineHeight: 1.6,
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 6, color: "#6366f1" }}>
            Query Insight
          </div>
          {winner.insight}
        </div>
      )}
    </div>
  );
}
