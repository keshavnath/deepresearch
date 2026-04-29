import json
import asyncio
import time
import logging
from typing import AsyncGenerator, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.schema import ResearchRequest
from app.engine import get_graph

logger = logging.getLogger(__name__)
app = FastAPI(title="Deep Research Multi-Agent System")


def get_status_data(node_name: str, output: dict) -> dict:
    """Extract minimal status data from each node's output."""
    data = {}
    
    # Extract only the most relevant field per node
    if node_name == "orchestrate":
        data["sub_questions"] = len(output.get("sub_questions", []))
    elif node_name == "search":
        data["results_found"] = len(output.get("search_results", []))
    elif node_name == "scrape":
        data["pages_extracted"] = len(output.get("scraped_pages", []))
    elif node_name == "synthesize":
        data["findings"] = len(output.get("findings", []))
    elif node_name == "critique":
        critique = output.get("critique")
        if critique:
            data["satisfied"] = critique.satisfied
            data["gaps"] = len(critique.gaps)
    elif node_name == "write_report":
        data["report_length"] = len(output.get("report", ""))
    
    return {"type": "status", "stage": node_name, "data": data}


async def event_generator(request: ResearchRequest) -> AsyncGenerator[str, None]:
    """
    Streams research progress via SSE with two-tier events:
    1. `status`: Lightweight progress updates per stage
    2. `report`: Final research output at completion
    
    Handles edge cases gracefully:
    - Empty search results -> proceeds with limited context
    - Failed scrapes -> uses available content
    - No findings -> errors at report stage
    - Agent errors -> caught and sent as error events
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
    
    start_time = time.time()
    final_state = None
    
    # Node names we care about for status updates
    task_nodes = {"orchestrate", "search", "scrape", "synthesize", "critique", "write_report"}
    
    try:
        logger.info(f"Starting research for query: {request.query[:100]}")
        
        # Using astream_events v2 for granular node/tool updates
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event["event"]
            
            if kind == "on_chain_end":
                node_name = event.get("name", "Unknown")
                output = event.get("data", {}).get("output", {})
                
                # Only process actual task nodes, skip routing decisions
                if node_name in task_nodes:
                    try:
                        # Extract status data and emit lightweight status event
                        status_event = get_status_data(node_name, output)
                        yield f"data: {json.dumps(status_event)}\n\n"
                    except Exception as e:
                        logger.warning(f"Error extracting status for {node_name}: {e}")
                
                # Keep track of final state for the report event
                if node_name == "write_report":
                    final_state = output

        # After graph completes, emit final report event
        if final_state:
            duration = time.time() - start_time
            iteration = final_state.get("iteration", 0)
            
            report_content = final_state.get("report", "")
            if not report_content:
                logger.warning("Final state has no report content")
            
            report_event = {
                "type": "report",
                "content": report_content,
                "sources": final_state.get("sources", []),
                "metadata": {
                    "query": request.query,
                    "iterations": iteration,
                    "duration_seconds": round(duration, 2)
                }
            }
            yield f"data: {json.dumps(report_event)}\n\n"
            logger.info(f"Research completed in {duration:.2f}s after {iteration} iteration(s)")
        else:
            # Graph executed but no report state - shouldn't happen
            logger.error("Graph completed but no final_state reached")
            error_event = {"type": "error", "message": "Graph completed but no report was generated"}
            yield f"data: {json.dumps(error_event)}\n\n"

    except RuntimeError as e:
        # Agents raise RuntimeError for edge cases (no findings, etc.)
        logger.error(f"Research failed (agent error): {e}")
        error_event = {"type": "error", "message": f"Research failed: {str(e)}"}
        yield f"data: {json.dumps(error_event)}\n\n"
    except Exception as e:
        # Catch all other exceptions
        logger.exception(f"Unexpected error during research: {e}")
        error_event = {"type": "error", "message": f"Unexpected error: {str(e)}"}
        yield f"data: {json.dumps(error_event)}\n\n"

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
