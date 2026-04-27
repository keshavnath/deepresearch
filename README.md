# Deep Research Multi-Agent System

A production-ready multi-agent system that autonomously researches queries by decomposing them, searching sources, scraping content, synthesizing findings, and iteratively validating quality.

**Built with:** LangGraph (orchestration) | FastAPI (API) | WandB Weave (observability) | Pydantic v2 (structured I/O)

## Architecture

```
User Query → Orchestrator (decompose) → Searcher → Scraper → Synthesizer → Critic → Reporter
                                           ↑__________________________________↓
                                           (iterate if gaps found, max 2 loops)
```

- **Orchestrator**: Plans research by decomposing query into 3-5 sub-questions
- **Searcher**: Parallel Tavily API calls with rate limiting (3 concurrent)
- **Scraper**: Trafilatura-based content extraction from top URLs
- **Synthesizer**: LLM generates structured findings (question, answer, confidence, sources)
- **Critic**: Validates sufficiency; triggers re-search if gaps exist
- **Reporter**: Generates final markdown report with citations

## Quick Start

```bash
# Install
uv sync

# Run server
uv run python -m uvicorn app.main:app

# In another terminal: query via HTTP
curl -N -X POST -H "Content-Type: application/json" \
  -d '{"query":"What are the latest advances in quantum error correction?"}' \
  http://localhost:8000/research
```

Returns **SSE stream** of real-time events (node completions, reasoning, traces).

## Key Technical Decisions (Interview Talking Points)

| Decision | Why |
|----------|-----|
| **LangGraph** over custom orchestration | StateGraph provides typed state management, conditional routing, automatic persistence for retries |
| **SSE streaming** over polling | Real-time updates, single connection, natural fit for agentic workflows |
| **Pydantic v2 schemas with Field descriptions** | Schema descriptions → LLM format instructions (no separate prompt engineering needed) |
| **Rate limiting via asyncio.Semaphore** | Prevents API throttling; max 3 Tavily, 5 LLM concurrent calls |
| **WandB Weave for observability** | Automatic LLM call tracing + decorator-based function instrumentation; captures all input/output/latency |
| **Structured I/O parser** | All agent outputs validated against schema; LLM respects format instructions |

## Testing

```bash
# Unit tests (structured I/O, rate limiting)
uv run pytest tests/unit/ -v

# Integration tests (API contract)
uv run pytest tests/integration/ -v

# With traces
uv run pytest tests/ -v -s
```

Tests initialize WandB Weave automatically (see traces in WandB dashboard).

## Project Structure

```
app/
├── agents/          # Orchestrator, Searcher, Scraper, Synthesizer, Critic, Reporter
├── utils/           # LLM wrapper, rate limiting, WandB Weave integration
├── schema.py        # Pydantic models + ResearchState (TypedDict)
├── engine.py        # LangGraph state machine compilation
├── config.py        # Environment variable loading
└── main.py          # FastAPI application + SSE event streaming
tests/
├── unit/            # Structured I/O, rate limiter tests
└── integration/     # API endpoint tests
```