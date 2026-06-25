# RAGArena 🏟️

A production-grade AI benchmarking platform that evaluates 7 RAG pipeline 
architectures side-by-side on the same corpus.

## What it does

RAGArena ingests domain-specific knowledge via web search, populates 7 RAG 
pipelines simultaneously, fires identical queries across all of them, and 
produces a comparative evaluation dashboard.

## The 7 RAG Pipelines

| Pipeline | Core Mechanic |
|---|---|
| Naive RAG | ChromaDB cosine similarity |
| HyDE + RAG-Fusion | Hypothetical doc embedding + multi-query |
| Self-RAG / CRAG | Retrieve → grade → re-retrieve (LangGraph) |
| Graph-RAG | NetworkX entity graph + 2-hop traversal |
| Agentic RAG | ReAct agent with dynamic tool selection |
| KAG / CAG | Knowledge summaries + semantic cache |
| Vectorless RAG | BM25 + cross-encoder reranker (no embeddings) |

## Evaluation Suite

- **Ragas metrics**: faithfulness, answer relevancy, context precision, context recall
- **LLM-as-judge**: correctness, completeness, groundedness (1–5 rubric)
- **Operational**: latency score, cost estimate, chunks fetched
- **Composite score**: 50% Ragas + 30% LLM-judge + 20% operational (configurable weights)
- **RAG Capability Profiler**: scores each pipeline across 5 query dimensions

## Dashboard Pages

- **Leaderboard** — composite score bar chart + Ragas radar per pipeline
- **Head-to-head** — same query answered by all 7 pipelines side by side
- **Cost vs Quality** — scatter plot showing quality/latency tradeoffs
- **RAG Selector** — capability heatmap + use-case search
- **Latency** — retrieval vs generation breakdown per pipeline

## Tech Stack

**Backend**: Python, FastAPI, LangGraph, Celery, Redis, PostgreSQL, ChromaDB  
**LLM**: llama3.2 via Ollama (local) / Groq API (production)  
**Embeddings**: nomic-embed-text via Ollama / sentence-transformers  
**Evaluation**: Ragas + custom LLM-as-judge  
**Frontend**: React, TypeScript, Vite, Recharts  
**Infrastructure**: Docker, Railway, Vercel  

## Local Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker Desktop
- Ollama (ollama.ai)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/ragarena
cd ragarena/backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Pull Ollama models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Start infrastructure

```bash
docker-compose up -d
```

### Run database migrations

```bash
# Connect to postgres and run migrations/init.sql
```

### Start backend

```bash
cd backend
uvicorn app.main:app --reload --port 8001
```

### Start Celery worker

```bash
cd backend
celery -A app.tasks.celery_app worker --loglevel=info -P solo -Q celery,eval
```

### Start frontend

```bash
cd frontend
npm install
npm run dev
```

### Run a benchmark

```bash
# Ingest a topic
curl -X POST http://localhost:8001/benchmark/ingest \
  -H "Content-Type: application/json" \
  -d '{"topic": "RBI monetary policy", "niche": "Indian banking"}'

# Run benchmark with the returned run_id
curl -X POST http://localhost:8001/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"run_id": "your-run-id", "queries": ["What is the RBI repo rate?"]}'

# Check status
curl http://localhost:8001/benchmark/{run_id}/status

# Get results
curl http://localhost:8001/benchmark/eval/{run_id}
```

## Environment Variables

```env
TAVILY_API_KEY=your_tavily_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ragarena
REDIS_URL=redis://localhost:6379/0
```

## Note on Evaluation Scores

Faithfulness, context precision, and context recall require strict JSON 
output from the LLM. With llama3.2 locally these fall back to 0.0. 
Answer relevancy and LLM-judge scores (correctness, completeness, 
groundedness) work correctly and differentiate pipelines meaningfully. 
In production with Groq API these scores improve significantly.

## Architecture
