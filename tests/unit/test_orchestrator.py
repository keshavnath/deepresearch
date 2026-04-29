import pytest
from app.agents.orchestrator import orchestrator_node
from app.schema import ResearchState

@pytest.mark.asyncio
async def test_orchestrator_node_decomp():
    """Test that the orchestrator correctly decomposes a query.
    
    Verifies:
    - sub_questions is a list with at least 1 question
    - research_plan is non-empty
    - iteration is passed through unchanged (critic increments it later in the loop)
    """
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
    
    # Orchestrator should decompose into researchable sub-questions
    result = await orchestrator_node(state)
    
    assert "sub_questions" in result
    assert isinstance(result["sub_questions"], list), "sub_questions should be a list"
    assert len(result["sub_questions"]) >= 1, "Should have at least 1 sub-question"
    
    assert "research_plan" in result
    assert isinstance(result["research_plan"], str)
    assert len(result["research_plan"]) > 0, "research_plan should not be empty"
    
    assert result["iteration"] == 0, "iteration passed through unchanged (critic increments it)"
    assert "events" in result
