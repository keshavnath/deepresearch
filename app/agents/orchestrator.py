from typing import List
from pydantic import BaseModel, Field
from app.schema import ResearchState
from app.utils.llm import get_llm
from app.utils.tracer import weave_op

class OrchestratorOutput(BaseModel):
    sub_questions: List[str] = Field(description="3-5 sub-questions to research")
    plan: str = Field(description="The research plan/strategy")

ORCHESTRATE_PROMPT = """
You are the Orchestrator of a Deep Research system.
Original User Query: {query}

Your goal is to decompose this query into 3-5 specific sub-questions that will provide a comprehensive answer.
Also, outline a brief research plan.

Respond with structured output.
"""

@weave_op("orchestrator")
async def orchestrator_node(state: ResearchState) -> dict:
    llm = get_llm()
    
    result = await llm.ainvoke(ORCHESTRATE_PROMPT.format(query=state["query"]), schema=OrchestratorOutput)
    
    return {
        "sub_questions": result.sub_questions,
        "research_plan": result.plan,
        "iteration": state.get("iteration", 0),
        "events": ["Orchestrator: Decomposed query into sub-questions."]
    }
