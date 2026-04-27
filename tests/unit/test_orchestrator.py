import pytest
from app.agents.orchestrator import orchestrator_node
from app.schema import ResearchState

@pytest.mark.asyncio
async def test_orchestrator_node_decomp():
    """Test that the orchestrator correctly decomposes a query."""
    state: ResearchState = {
        "query": "Is quantum computing ready for commercial encryption?",
        "iteration": 0,
        "events": [],
        "sub_questions": [],
        "research_plan": "",
        "search_results": [],
        "scraped_pages": [],
        "findings": [],
        "critique": None,
        "sources": [],
        "report": None
    }
    
    # We expect the structured LLM to return valid sub-questions
    result = await orchestrator_node(state)
    
    assert "sub_questions" in result
    assert "research_plan" in result
    assert len(result["sub_questions"]) >= 3
    assert result["iteration"] == 0
    assert "events" in result
