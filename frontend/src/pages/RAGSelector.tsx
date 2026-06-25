import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getLeaderboard, type PipelineScore } from "../api/eval";
import { useRun } from "../context/RunContext";

// ── constants ──────────────────────────────────────────────────────────────

const DIMENSIONS = ["factual_precision", "multi_hop", "summarisation", "ambiguous", "keyword_dense"] as const;
type Dim = (typeof DIMENSIONS)[number];

const DIM_LABELS: Record<Dim, string> = {
  factual_precision: "Factual",
  multi_hop: "Multi-hop",
  summarisation: "Summarise",
  ambiguous: "Ambiguous",
  keyword_dense: "Keyword",
};

const PIPELINE_DESC: Record<string, string> = {
  naive_rag: "Simple cosine similarity retrieval",
  hyde_fusion: "Hypothetical doc + multi-query fusion",
  self_rag: "Retrieve, grade, re-retrieve loop",
  graph_rag: "Entity graph with 2-hop traversal",
  agentic_rag: "ReAct agent with dynamic tool selection",
  kag_cag: "Knowledge summaries + semantic cache",
  vectorless: "BM25 + cross-encoder reranker",
};

// ── aggregation ────────────────────────────────────────────────────────────

interface PipelineAgg {
  pipeline_id: string;
  composite_score: number;
  dims: Record<Dim, number>;
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
      composite_score: avg(rows.map((r) => r.composite_score)),
      dims: {
        factual_precision: avg(rows.map((r) => r.answer_relevancy)),
        multi_hop: avg(rows.map((r) => r.judge_correctness / 5)),
        summarisation: avg(rows.map((r) => r.judge_completeness / 5)),
        ambiguous: avg(rows.map((r) => r.judge_groundedness / 5)),
        keyword_dense: avg(rows.map((r) => r.latency_score)),
      },
    }))
    .sort((a, b) => b.composite_score - a.composite_score);
}

// ── cell color ─────────────────────────────────────────────────────────────

function cellStyle(score: number): React.CSSProperties {
  if (score > 0.7) return { background: "#4f46e5", color: "#fff" };
  if (score > 0.5) return { background: "#818cf8", color: "#fff" };
  if (score > 0.3) return { background: "#c7d2fe", color: "#1e1b4b" };
  return { background: "#f3f4f6", color: "#6b7280" };
}

// ── Section 1: Heatmap ────────────────────────────────────────────────────

function Heatmap({ pipelines }: { pipelines: PipelineAgg[] }) {
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        overflow: "hidden",
        marginBottom: 28,
      }}
    >
      <div style={{ fontWeight: 600, color: "#111827", padding: "16px 20px 12px" }}>
        Capability Heatmap
        <span style={{ fontWeight: 400, fontSize: 12, color: "#9ca3af", marginLeft: 8 }}>
          Derived from eval metrics — darker = stronger
        </span>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ background: "#f9fafb", borderBottom: "1px solid #e5e7eb" }}>
              <th style={{ padding: "10px 16px", textAlign: "left", fontWeight: 600, color: "#374151", width: 140 }}>
                Pipeline
              </th>
              {DIMENSIONS.map((d) => (
                <th key={d} style={{ padding: "10px 12px", textAlign: "center", fontWeight: 600, color: "#374151" }}>
                  {DIM_LABELS[d]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pipelines.map((p) => (
              <tr key={p.pipeline_id} style={{ borderBottom: "1px solid #f3f4f6" }}>
                <td style={{ padding: "10px 16px", fontWeight: 600, color: "#111827", whiteSpace: "nowrap" }}>
                  {p.pipeline_id}
                </td>
                {DIMENSIONS.map((d) => {
                  const score = p.dims[d];
                  return (
                    <td key={d} style={{ padding: 6, textAlign: "center" }}>
                      <div
                        style={{
                          ...cellStyle(score),
                          borderRadius: 4,
                          padding: "6px 4px",
                          fontSize: 12,
                          fontWeight: 600,
                          margin: "0 4px",
                        }}
                      >
                        {(score * 100).toFixed(0)}%
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Section 2: Pipeline cards ─────────────────────────────────────────────

function Badge({ label, color }: { label: string; color: string }) {
  return (
    <span
      style={{
        background: color === "green" ? "#dcfce7" : "#fee2e2",
        color: color === "green" ? "#166534" : "#991b1b",
        fontSize: 11,
        fontWeight: 600,
        padding: "2px 8px",
        borderRadius: 12,
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}

function PipelineCard({
  pipeline,
  highlighted,
}: {
  pipeline: PipelineAgg;
  highlighted: boolean;
}) {
  const sorted = [...DIMENSIONS].sort((a, b) => pipeline.dims[b] - pipeline.dims[a]);
  const strengths = sorted.slice(0, 2);
  const weakness = sorted[sorted.length - 1];

  return (
    <div
      style={{
        background: "#fff",
        border: `2px solid ${highlighted ? "#6366f1" : "#e5e7eb"}`,
        borderRadius: 8,
        padding: 16,
        display: "flex",
        flexDirection: "column",
        gap: 8,
        transition: "border-color 0.2s",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <span style={{ fontWeight: 700, fontSize: 14, color: "#111827" }}>
          {pipeline.pipeline_id}
        </span>
        <span style={{ fontWeight: 700, color: "#6366f1", fontSize: 13 }}>
          {(pipeline.composite_score * 100).toFixed(1)}%
        </span>
      </div>

      <div style={{ fontSize: 12, color: "#6b7280" }}>
        {PIPELINE_DESC[pipeline.pipeline_id] ?? "RAG pipeline"}
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {strengths.map((d) => (
          <Badge key={d} label={DIM_LABELS[d]} color="green" />
        ))}
        <Badge label={DIM_LABELS[weakness]} color="red" />
      </div>
    </div>
  );
}

// ── Section 3: Use-case search ────────────────────────────────────────────

const DIM_KEYWORDS: Record<Dim, string[]> = {
  factual_precision: ["fact", "accurate", "precise", "correct", "specific"],
  multi_hop: ["multi", "hop", "complex", "reasoning", "chain", "step"],
  summarisation: ["summar", "overview", "digest", "long", "document"],
  ambiguous: ["ambiguous", "unclear", "vague", "interpret", "context"],
  keyword_dense: ["keyword", "search", "fast", "quick", "exact", "term"],
};

function findBestPipelines(query: string, pipelines: PipelineAgg[]): Set<string> {
  const lower = query.toLowerCase();
  const dimScores: Record<Dim, number> = {
    factual_precision: 0,
    multi_hop: 0,
    summarisation: 0,
    ambiguous: 0,
    keyword_dense: 0,
  };

  for (const dim of DIMENSIONS) {
    for (const kw of DIM_KEYWORDS[dim]) {
      if (lower.includes(kw)) dimScores[dim] += 1;
    }
  }

  // Top matched dimension(s)
  const topDim = DIMENSIONS.slice().sort((a, b) => dimScores[b] - dimScores[a])[0];

  // Return top 2 pipelines by score on that dimension
  const ranked = pipelines.slice().sort((a, b) => b.dims[topDim] - a.dims[topDim]);
  return new Set(ranked.slice(0, 2).map((p) => p.pipeline_id));
}

// ── main page ──────────────────────────────────────────────────────────────

export default function RAGSelector() {
  const { currentRunId } = useRun();
  const [useCaseInput, setUseCaseInput] = useState("");
  const [highlighted, setHighlighted] = useState<Set<string>>(new Set());

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

  function handleSearch() {
    if (!useCaseInput.trim()) {
      setHighlighted(new Set());
      return;
    }
    setHighlighted(findBestPipelines(useCaseInput, pipelines));
  }

  return (
    <div style={{ maxWidth: 1050, margin: "0 auto" }}>
      <h2 style={{ margin: "0 0 4px", color: "#111827" }}>RAG Selector</h2>
      <p style={{ margin: "0 0 24px", color: "#6b7280", fontSize: 14 }}>
        Compare pipeline capabilities and find the best fit for your use case.
      </p>

      {/* Section 1: Heatmap */}
      <Heatmap pipelines={pipelines} />

      {/* Section 3: Use-case search (above cards so results are visible) */}
      <div
        style={{
          background: "#fff",
          border: "1px solid #e5e7eb",
          borderRadius: 8,
          padding: 16,
          marginBottom: 20,
          display: "flex",
          gap: 8,
        }}
      >
        <input
          value={useCaseInput}
          onChange={(e) => setUseCaseInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          placeholder="Describe your use case… e.g. 'accurate factual QA over long documents'"
          style={{
            flex: 1,
            padding: "8px 12px",
            border: "1px solid #d1d5db",
            borderRadius: 6,
            fontSize: 14,
          }}
        />
        <button
          onClick={handleSearch}
          style={{
            padding: "8px 20px",
            background: "#6366f1",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            fontWeight: 600,
            whiteSpace: "nowrap",
          }}
        >
          Find best pipeline
        </button>
        {highlighted.size > 0 && (
          <button
            onClick={() => setHighlighted(new Set())}
            style={{
              padding: "8px 12px",
              background: "#f3f4f6",
              color: "#6b7280",
              border: "1px solid #e5e7eb",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            Clear
          </button>
        )}
      </div>

      {highlighted.size > 0 && (
        <div style={{ marginBottom: 12, fontSize: 13, color: "#6366f1", fontWeight: 600 }}>
          Best matches: {[...highlighted].join(", ")}
        </div>
      )}

      {/* Section 2: Pipeline cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 12,
        }}
      >
        {pipelines.map((p) => (
          <PipelineCard
            key={p.pipeline_id}
            pipeline={p}
            highlighted={highlighted.has(p.pipeline_id)}
          />
        ))}
      </div>
    </div>
  );
}
