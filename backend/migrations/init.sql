CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic       TEXT NOT NULL,
    niche       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    chunk_count INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pipeline_results (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id         UUID NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    pipeline_id    TEXT NOT NULL,
    query_id       TEXT NOT NULL,
    query_text     TEXT NOT NULL,
    answer         TEXT,
    context_chunks JSONB,
    retrieval_ms   FLOAT,
    generation_ms  FLOAT,
    token_input    INTEGER,
    token_output   INTEGER,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval_scores (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    pipeline_id         TEXT NOT NULL,
    query_id            TEXT NOT NULL,
    faithfulness        FLOAT,
    answer_relevancy    FLOAT,
    context_precision   FLOAT,
    context_recall      FLOAT,
    judge_correctness   FLOAT,
    judge_completeness  FLOAT,
    judge_groundedness  FLOAT,
    latency_score       FLOAT,
    cost_usd            FLOAT,
    composite_score     FLOAT,
    insight             TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS capability_profiles (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id            UUID NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    pipeline_id       TEXT NOT NULL,
    dim_factual       FLOAT,
    dim_multihop      FLOAT,
    dim_summarisation FLOAT,
    dim_ambiguous     FLOAT,
    dim_keyword       FLOAT,
    best_for          TEXT[],
    avoid_when        TEXT[],
    sweet_spot        TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS query_tags (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id     UUID NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    query_id   TEXT NOT NULL,
    query_text TEXT NOT NULL,
    tags       TEXT[]
);

CREATE INDEX IF NOT EXISTS idx_pipeline_results_run_id ON pipeline_results(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_scores_run_id ON eval_scores(run_id);
CREATE INDEX IF NOT EXISTS idx_capability_profiles_run_id ON capability_profiles(run_id);
CREATE INDEX IF NOT EXISTS idx_query_tags_run_id ON query_tags(run_id);
