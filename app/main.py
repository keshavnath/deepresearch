import json
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from app.schema import ResearchRequest
from app.engine import get_graph

app = FastAPI(title="Deep Research Multi-Agent System")

async def event_generator(request: ResearchRequest) -> AsyncGenerator[str, None]:
    """
    Streams events from the LangGraph orchestrator via SSE.
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
    
    # Configure the graph with the model settings in a side-channel/config
    config = {
        "configurable": {
            "model_name": request.model_name,
            "base_url": request.base_url,
            "api_key": request.api_key
        }
    }
    
    try:
        # Using astream_events v2 for granular node/tool updates
        async for event in graph.astream_events(initial_state, config, version="v2"):
            kind = event["event"]
            if kind == "on_chain_end":
                # We can filter for specific node completions to provide clean UI updates
                node_name = event.get("name", "Unknown")
                data = {
                    "event": "node_complete",
                    "node": node_name,
                    "data": event.get("data", {}).get("output", {})
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            elif kind == "on_chat_model_stream":
                # Stream internal reasoning if needed
                content = event["data"].get("chunk", {}).content
                if content:
                    yield f"data: {json.dumps({'event': 'reasoning', 'delta': content})}\n\n"

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
