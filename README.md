# Deep Research Multi-Agent System

A production-ready multi-agent system that autonomously researches queries by decomposing them, searching sources, scraping content, synthesizing findings, and iteratively validating quality.

**Built with:** Python (Asyncio) | uv (Environment) | LangGraph (Agents) | FastAPI (Backend) | Trafilatura (Web Scraping) | WandB Weave (Observability) | Pydantic (structured I/O) | BM25 (Keyword Retrieval)

## Overview

This system autonomously executes deep research workflows on any query. Given a user question, it systematically:
1. Decomposes the query into researchable sub-questions
2. Searches the web for relevant sources
3. Extracts content from top results
4. Synthesizes structured findings with confidence scores
5. Validates research quality; if gaps exist, loops back to search with refined questions (max N iterations)
6. Reports findings as a formatted markdown document with citations
7. Logs every trace including input, decision, output and system state for observability

Use it for: competitive analysis, technical due diligence, market research, literature reviews, fact-checking, or any task requiring comprehensive web-based research with iterative validation.

## Report Sample
Report sample of PSG vs Bayern Munich (from just 12 hours before time of writing!)

![WanDB Trace of Report](./img/report_sample.png)

## Architecture

```
User Query → Orchestrator (decompose) → Searcher → Scraper → Synthesizer → Critic → Reporter
                                           ↑___________________________________↓
                                           (iterate if gaps found, max N loops)
```

### Langgraph Nodes

- **Orchestrator**: Plans research by decomposing query into 3-5 sub-questions
- **Searcher**: Parallel Tavily API calls with rate limiting
- **Scraper**: Trafilatura-based content extraction from top URLs
- **Synthesizer**: Uses BM25 keyword retrieval to find relevant passages, then LLM generates structured findings (question, answer, confidence, sources)
- **Critic**: Validates sufficiency; triggers re-search if gaps exist
- **Reporter**: Generates final markdown report with citations

## Quick Start

```bash
# Install
uv sync

# Populate environment variables
cp .env.example .env

# Run server
uv run python -m uvicorn app.main:app

# In another terminal: query via HTTP
curl -N -X POST -H "Content-Type: application/json" \
  -d '{"query":"What are the latest advances in quantum error correction?"}' \
  http://localhost:8000/research
```

Returns **SSE stream** of real-time node completions.

## Key Technical Decisions

| Tech | Why |
|----------|-----|
| **`LangGraph`** over custom orchestration | `StateGraph` provides typed state management, conditional routing, automatic persistence for retries |
| **`SSE` streaming** | Real-time updates, single connection, natural fit for agentic workflows |
| **`Pydantic` schemas with Field descriptions** | All agent outputs validated against schema, LLM respects format instructions; Schema descriptions → LLM format instructions (less separate prompt engineering needed) |
| **Parallelism limiting** | `asyncio.Semaphore` prevents API throttling; max 3 Tavily, 5 LLM concurrent calls |
| **`WandB Weave` for observability** | Automatic LLM call tracing + decorator-based function instrumentation; captures all input/output/latency |
| **`BM25` for passage retrieval** | Keyword-based ranking (no embeddings) finds relevant passages per question; faster, no API calls, high lexical recall |

## Testing

```bash
# All tests
uv run pytest tests/ -v

# Unit tests (structured I/O, rate limiting)
uv run pytest tests/unit/ -v

# Integration tests (API contract)
uv run pytest tests/integration/ -v

```

## Project Structure

```
app/
├── agents/          # Orchestrator, Searcher, Scraper, Synthesizer, Critic, Reporter
├── utils/           # LLM wrapper, rate limiting, WandB Weave integration, BM25 retrieval
├── schema.py        # Pydantic models + ResearchState (TypedDict)
├── engine.py        # LangGraph state machine compilation
├── config.py        # Environment variable loading
└── main.py          # FastAPI application + SSE event streaming
tests/
├── unit/            # Structured I/O, Agent Node tests, BM25 retrieval tests
└── integration/     # API endpoint tests, system dry run
```

## Weights & Biases (WandB Weave) Observability

Project uses WandB Weave for Observability and Logging

Sample of Weave metrics on project Dashboard
![Weave Metrics Dashboard](./img/weave_metrics.png)

Sample of Weave Trace with agent state and I/O
![Weave Trace](./img/weave_trace.png)