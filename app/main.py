import json
import asyncio
from typing import AsyncGenerator, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.schema import ResearchRequest
from app.engine import get_graph

app = FastAPI(title="Deep Research Multi-Agent System")


def serialize_for_json(obj: Any) -> Any:
    """Convert Pydantic models and other non-JSON types to JSON-serializable format."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    return obj


async def event_generator(request: ResearchRequest) -> AsyncGenerator[str, None]:
    """
    Streams events from the LangGraph orchestrator via SSE.
    Only streams high-level node completions (not token-by-token reasoning).
    """
    graph = get_graph()
    initial_state = {
        "query": request.query,
        "sub_questions": [],
        "research_plan": "",
        "search_results": [],
        "scraped_pages": [],
        "findings": [],
        "critique": None,
        "iteration": 0,
        "sources": [],
        "events": [],
        "report": None
    }
    
    try:
        # Using astream_events v2 for granular node/tool updates
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event["event"]
            
            if kind == "on_chain_end":
                # Only emit events for top-level node completions
                node_name = event.get("name", "Unknown")
                output = event.get("data", {}).get("output", {})
                
                # Convert any Pydantic models in output to dicts
                output = serialize_for_json(output)
                
                data = {
                    "event": "node_complete",
                    "node": node_name,
                    "data": output
                }
                yield f"data: {json.dumps(data)}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/research")
async def research(request: ResearchRequest):
    """
    Endpoint to start a deep research task and stream results.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    return StreamingResponse(
        event_generator(request),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
