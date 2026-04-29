import logging
from app.schema import ResearchState, CritiqueResult
from app.utils.llm import get_llm
from app.utils.tracer import weave_op
from app.config import MAX_ITERATIONS

logger = logging.getLogger(__name__)

CRITIC_PROMPT = """
Analyze the original research query and the findings gathered so far. 
Determine if the research is sufficient or if there are critical information gaps.

Original Query: {query}

Findings so far:
{findings}
"""

@weave_op("critic")
async def critic_node(state: ResearchState) -> dict:
    """Evaluate research quality and decide whether to continue or finalize.
    
    Assesses whether current findings sufficiently answer the original query.
    If gaps exist and iteration limit not reached, triggers re-search.
    
    Args:
        state: ResearchState with query, findings, and iteration count
        
    Returns:
        dict with 'critique' (CritiqueResult), 'iteration' (incremented),
        optionally 'sub_questions' (if re-searching), and 'events'
        
    Raises:
        RuntimeError: If no findings available to critique
    """
    findings = state.get("findings", [])
    
    # Edge case: No findings to critique
    if not findings:
        logger.error("Critic: No findings to evaluate")
        raise RuntimeError("Cannot critique: no findings generated")
    
    try:
        llm = get_llm()
        
        findings_str = "\n".join([f"Q: {f.question}\nA: {f.answer}\nConfidence: {f.confidence}" 
                                  for f in findings])
        
        critique = await llm.ainvoke(CRITIC_PROMPT.format(
            query=state["query"],
            findings=findings_str
        ), schema=CritiqueResult)
        
        # Validate critique result
        if not isinstance(critique.satisfied, bool):
            logger.warning(f"Invalid satisfied value: {critique.satisfied}, defaulting to False")
            critique.satisfied = False
        
        iteration = state.get("iteration", 0)
        updates = {
            "critique": critique,
            "iteration": iteration + 1,
            "events": [f"Critic: Satisfied={critique.satisfied}. Gaps identified: {len(critique.gaps)}"]
        }
        
        # If not satisfied and we haven't looped too many times, update sub_questions for next search
        if not critique.satisfied and iteration < MAX_ITERATIONS:
            if critique.new_sub_questions:
                updates["sub_questions"] = critique.new_sub_questions
                logger.info(f"Critic: Planning re-search with {len(critique.new_sub_questions)} new questions")
            else:
                logger.warning("Critic: Not satisfied but no new questions provided")
        elif iteration >= MAX_ITERATIONS:
            logger.info(f"Critic: Max iterations ({MAX_ITERATIONS}) reached, finalizing report")
        
        return updates
        
    except Exception as e:
        logger.error(f"Critic node failed: {e}")
        raise
