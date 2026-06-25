import client from "./client";

export interface PipelineScore {
  pipeline_id: string;
  query_id: string;
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  context_recall: number;
  judge_correctness: number;
  judge_completeness: number;
  judge_groundedness: number;
  latency_score: number;
  cost_usd: number;
  composite_score: number;
  insight: string;
}

export interface LeaderboardResponse {
  run_id: string;
  status: string;
  pipeline_scores: PipelineScore[];
}

export interface StatusResponse {
  run_id: string;
  status: string;
  progress: string;
}

export async function getLeaderboard(runId: string): Promise<LeaderboardResponse> {
  const { data } = await client.get<LeaderboardResponse>(`/benchmark/eval/${runId}`);
  return data;
}

export async function getRunStatus(runId: string): Promise<StatusResponse> {
  const { data } = await client.get<StatusResponse>(`/benchmark/${runId}/status`);
  return data;
}

export async function getQueryResults(
  runId: string,
  queryId: string
): Promise<PipelineScore[]> {
  const { data } = await client.get<LeaderboardResponse>(`/benchmark/eval/${runId}`);
  return data.pipeline_scores.filter((s) => s.query_id === queryId);
}
