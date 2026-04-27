import pytest
from app.agents.critic import critic_node
from app.schema import ResearchState, Finding, CritiqueResult
from app.config import MAX_ITERATIONS

@pytest.mark.asyncio
async def test_critic_node_satisfied():
    """Test that the critic recognizes when research is sufficient."""
    state: ResearchState = {
        "query": "What is the capital of France?",
        "findings": [
            Finding(
                question="What is the capital of France?",
                answer="The capital of France is Paris.",
                confidence=1.0,
                source_urls=["https://en.wikipedia.org/wiki/Paris"]
            )
        ],
        "iteration": 0,
        "events": [],
        "sub_questions": ["What is the capital of France?"],
        "research_plan": "Simple lookup",
        "search_results": [],
        "scraped_pages": [],
        "critique": None,
        "sources": [],
        "report": None
    }
    
    result = await critic_node(state)
    
    assert "critique" in result
    assert isinstance(result["critique"], CritiqueResult)
    # Even if LLM is unpredictable, for such a simple query it should be satisfied
    assert result["critique"].satisfied is True
    assert result["iteration"] == 1

@pytest.mark.asyncio
async def test_critic_node_max_iterations():
    """Test that sub_questions are NOT updated if iteration >= MAX_ITERATIONS."""
    # We'll rely on the default MAX_ITERATIONS=2 from config
    state: ResearchState = {
        "query": "Complex query",
        "findings": [Finding(question="test", answer="test", confidence=0.5, source_urls=[])],
        "iteration": MAX_ITERATIONS, # Already at max
        "events": [],
        "sub_questions": ["test"],
        "research_plan": "test",
        "search_results": [],
        "scraped_pages": [],
        "critique": None,
        "sources": [],
        "report": None
    }
    
    result = await critic_node(state)
    
    # Even if LLM says not satisfied, it shouldn't update sub_questions because iteration is at max
    assert "sub_questions" not in result
    assert result["iteration"] == MAX_ITERATIONS+1
