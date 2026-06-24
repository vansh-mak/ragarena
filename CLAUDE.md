# RAGArena — Claude Project Memory

## What this project is
RAGArena is a production-grade AI benchmarking platform that evaluates 7 RAG 
pipeline architectures against each other on the same corpus. An LLM agent 
ingests domain-specific knowledge via web search, populates all 7 pipelines 
simultaneously, fires identical queries across them, and produces a 
comparative evaluation dashboard.

## Primary goal
Portfolio project targeting top-tier AI startups (Anthropic, etc.).
Demonstrates: multi-agent systems, advanced RAG architectures, evaluation 
thinking, production infrastructure, and full-stack AI engineering.

## Tech stack
- Backend: Python, FastAPI, SQLAlchemy (async), asyncpg
- Agents: LangGraph
- Task queue: Celery + Redis
- Vector store: ChromaDB (namespaced per pipeline per run)
- LLM: llama3.2 via Ollama (local, http://localhost:11434)
- Embeddings: nomic-embed-text via Ollama (local)
- Evaluation: Ragas + custom LLM-as-judge
- Graph store: NetworkX (persisted with pickle)
- BM25: rank-bm25 + cross-encoder reranker
- Frontend: React + TypeScript + Vite + Recharts
- Database: PostgreSQL (async SQLAlchemy)

## The 7 RAG pipelines
1. Naive RAG — ChromaDB cosine similarity
2. HyDE + RAG-Fusion — hypothetical doc embedding + multi-query
3. Self-RAG / CRAG — retrieve → grade → re-retrieve (LangGraph)
4. Graph-RAG — NetworkX entity graph + community retrieval
5. Agentic RAG — ReAct agent with vector + keyword tools (LangGraph)
6. KAG / CAG — knowledge/cache augmented generation
7. Vectorless RAG — BM25 + cross-encoder reranker (no embeddings)

## Evaluation suite
- Ragas metrics: faithfulness, answer_relevancy, context_precision, context_recall
- LLM-as-judge: correctness, completeness, groundedness (1–5 rubric, Claude)
- Operational: latency_ms (retrieval + generation split), token cost USD
- Composite score: 50% Ragas + 30% LLM-judge + 20% operational (configurable)
- Capability profiler: scores each pipeline on 5 query dimensions
  (factual_precision, multi_hop, summarisation, ambiguous, keyword_dense)
- Per-pipeline output: best_for[], avoid_when[], sweet_spot string

## Folder structure
ragarena/
  backend/
    app/
      api/          # FastAPI routers
      agents/       # LangGraph agents (ingestion_agent.py)
      pipelines/    # One file per RAG pipeline
      eval/         # ragas_scorer, llm_judge, aggregator, insight_generator,
                    # capability_profiler
      models/       # SQLAlchemy async models
      tasks/        # Celery tasks and orchestrator
      config.py
      database.py
      main.py
    .env
    requirements.txt
  frontend/
    src/
      pages/        # Leaderboard, HeadToHead, CostQuality, RAGSelector, Latency
      components/   # Layout, shared UI
      api/          # client.ts, eval.ts
      context/      # RunContext.tsx
  docker-compose.yml
  CLAUDE.md

## Database tables
- benchmark_runs: run metadata, status, topic, niche
- pipeline_results: answer + timing + token counts per pipeline per query
- eval_scores: all metric scores per pipeline per query
- capability_profiles: dimension scores + best_for/avoid_when per pipeline
- query_tags: query type classification per query

## Key design decisions (do not change these)
- Single embedding model (nomic-embed-text via Ollama) across all vector pipelines
  — isolates retrieval strategy as the variable, not embedding quality
- retrieval_ms and generation_ms tracked separately inside each pipeline
- Composite score stored at default weights; dashboard reweights at read time
- ChromaDB collections namespaced as {pipeline_id}_{run_id}
- BM25 + graph indexes persisted to disk at {CHROMADB_PATH}/{pipeline}_{run_id}/
- Celery chord pattern: group(7 tasks) | callback for parallel execution
- Eval weights configurable via ScoringConfig but default is 50/30/20
- All LLM calls use llama3.2 via Ollama running locally on port 11434
- Ollama must be running before starting the backend (ollama serve)

## Build sequence (MVP plan)
### MVP 1 — COMPLETE
- [X] Project scaffold and folder structure
- [X] SQLAlchemy async models
- [X] config.py setup
- [X] LangGraph ingestion agent
- [X] Naive RAG pipeline
- [X] Basic FastAPI endpoints (ingest + run, no Celery)

### MVP 2 - COMPLETE
- [X] HyDE + RAG-Fusion pipeline
- [X] Vectorless RAG pipeline
- [X] Self-RAG / CRAG pipeline
- [X] Graph-RAG pipeline
- [X] Agentic RAG pipeline
- [X] KAG / CAG pipeline
- [X] Celery setup + parallel dispatch
- [X] Redis status tracking
- [X] Switched LLM backend to Ollama(llama3.2 + nomic-embed-text)

### MVP 3 - TO DO
- [ ] Ragas scorer
- [ ] LLM-as-judge scorer
- [ ] Score aggregator + composite
- [ ] Insight generator

### MVP 4 - TO DO
- [ ] Capability profiler + query tagger
- [ ] React scaffold + routing
- [ ] Leaderboard page
- [ ] Head-to-head page
- [ ] Cost vs quality scatter
- [ ] RAG selector + heatmap
- [ ] Latency breakdown page

## How to run locally
```bash
# Start Ollama (must be first)
# ollama serve

# Start infra
docker-compose up -d

# Start backend
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload --port 8001

# Start Celery worker (MVP 2+)
celery -A app.tasks.celery_app worker --loglevel=info -P solo

# Start frontend (MVP 4)
cd frontend
npm run dev
```

## Naming conventions
- Pipeline IDs (used everywhere as strings): 
  naive_rag, hyde_fusion, self_rag, graph_rag, agentic_rag, kag_cag, vectorless
- All async functions use async/await — no sync SQLAlchemy calls
- All pipeline classes implement: ingest(chunks), retrieve(query), generate(query)
- PipelineResult dict keys: pipeline_id, query_id, answer, context_chunks,
  retrieval_ms, generation_ms, token_input, token_output
- EvalScore dict keys: all PipelineResult keys + faithfulness, answer_relevancy,
  context_precision, context_recall, judge_correctness, judge_completeness,
  judge_groundedness, latency_score, cost_usd, composite_score, insight


## Git commit instructions

After every completed prompt, generate a git commit using this exact format:

Scope options: scaffold, models, ingestion, pipeline, eval, tasks, api, frontend

Rules:
- Subject line max 60 characters
- Bullet points max 3, each max 72 characters  
- No "created file X" — describe what the code DOES, not the file name
- If a bug was fixed during the prompt, add: fix: description as a second commit
- Never commit broken or untested code

After generating the message, also output the exact commands to run:
git add .
git commit -m "..."


## Deployment

### Frontend — Vercel
- Host: vercel.com
- Root directory: /frontend
- Build command: npm run build
- Output: dist/
- Env var: VITE_API_URL=https://ragarena-backend.up.railway.app
- Auto-deploys on push to main branch

### Backend — Railway  
- Host: railway.app
- Root directory: /backend
- Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
- Services: FastAPI app, Celery worker, Postgres, Redis
- All env vars set in Railway dashboard (never commit .env)

### Deployment workflow
After merging to main:
1. Railway auto-deploys backend (watch logs for migration errors)
2. Vercel auto-deploys frontend (~1 min build)
3. Verify: hit /health endpoint on Railway URL
4. Verify: open Vercel URL, confirm API calls succeed

### Health check endpoint
backend/app/main.py must have:
GET /health → returns {status: "ok", version: "1.0.0"}
Railway uses this to verify the container is running.   
