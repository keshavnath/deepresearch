from app.schema import ResearchState, CritiqueResult
from app.utils.llm import get_llm
from app.utils.tracer import get_tracer
from app.config import MAX_ITERATIONS

CRITIC_PROMPT = """
Analyze the original research query and the findings gathered so far. 
Determine if the research is sufficient or if there are critical information gaps.

Original Query: {query}

Findings so far:
{findings}
"""

async def critic_node(state: ResearchState) -> dict:
    tracer = get_tracer()
    tracer.log_agent_start("critic", input_state={"num_findings": len(state["findings"])})
    
    try:
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
        
        tracer.log_agent_end("critic", output=updates)
        return updates
    except Exception as e:
        tracer.log_agent_end("critic", error=str(e))
        raise
