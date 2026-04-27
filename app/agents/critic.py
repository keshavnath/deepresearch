from app.schema import ResearchState, CritiqueResult
from app.utils.llm import get_llm
from app.utils.tracer import weave_op
from app.config import MAX_ITERATIONS

CRITIC_PROMPT = """
Analyze the original research query and the findings gathered so far. 
Determine if the research is sufficient or if there are critical information gaps.

Original Query: {query}

Findings so far:
{findings}
"""

@weave_op("critic")
async def critic_node(state: ResearchState) -> dict:
    llm = get_llm()
    
    findings_str = "\n".join([f"Q: {f.question}\nA: {f.answer}" for f in state["findings"]])
    
    critique = await llm.ainvoke(CRITIC_PROMPT.format(
        query=state["query"],
        findings=findings_str
    ), schema=CritiqueResult)
    
    iteration = state.get("iteration", 0)
    updates = {
        "critique": critique,
        "iteration": iteration + 1,
        "events": [f"Critic: Satisfied={critique.satisfied}. Gaps identified: {len(critique.gaps)}"]
    }
    
    # If not satisfied and we haven't looped too many times, update sub_questions for next search
    if not critique.satisfied and iteration < MAX_ITERATIONS:
        updates["sub_questions"] = critique.new_sub_questions
        
    return updates
