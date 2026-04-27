# Deep Research Multi-Agent System

A sophisticated multi-agent system built with **LangGraph** and **FastAPI** that performs autonomous deep research.

## Architecture

The system uses a **Hub-and-Spoke (Supervisor)** pattern:

- **Orchestrator (Hub)**: The "brain" that plans, routes, and synthesizes.
- **Searcher (Worker)**: Specialized in web discovery via Tavily.
- **Scraper (Worker)**: Extracts clean content using Trafilatura.
- **Reasoner (Worker)**: Synthesizes logic and identifies data gaps.
- **Critic (Worker)**: Reflects on quality and triggers feedback loops.

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Add your TAVILY_API_KEY and OPENAI_API_KEY
   ```

3. Run the server:
   ```bash
   uv run python -m uvicorn app.main:app --reload
   ```

## API Usage

### Stream Research
```bash
curl -N -X POST -H "Content-Type: application/json" \
     -d '{"query": "Future of RISC-V in high-performance computing"}' \
     http://localhost:8000/research
```

The endpoint returns an SSE stream of JSON events:
- `node_complete`: Indicates an agent finished a task.
- `reasoning`: Real-time "thoughts" from the Orchestrator.
- `error`: Diagnostics if something fails.
