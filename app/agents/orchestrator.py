from typing import List, Literal
from pydantic import BaseModel, Field
from app.schema import ResearchState, ResearchTask
from app.utils.llm import get_llm

class OrchestratorDecision(BaseModel):
    """The structured decision made by the orchestrator."""
    next_node: Literal["searcher", "scraper", "reasoner", "critic", "finalize"] = Field(
        description="The next specialized worker to activate."
    )
    thought: str = Field(description="Internal reasoning for this decision.")
    task_query: str = Field(default="", description="The specific query or instruction for the next worker.")

async def orchestrator_node(state: ResearchState):
    """
    The main Hub node. It analyzes the state and dispatches workers.
    """
    # Use global config if request didn't provide specific overriden keys (logic inside get_llm)
    llm = get_llm().with_structured_output(OrchestratorDecision)
    
    # Construct a prompt that includes the history and context
    history_str = "\n".join(state.get("history", []))
    
    prompt = f"""
    You are the Orchestrator of a Deep Research system.
    Original User Query: {state['query']}
    
    Current History:
    {history_str}
    
    Current Plan:
    {[t.model_dump() for t in state.get('plan', [])]}
    
    Your goal is to decide the next step. 
    - If you need to find more information, go to 'searcher'.
    - If you have URLs but need content, go to 'scraper'.
    - If you have raw content and need to synthesize logic/insights, go to 'reasoner'.
    - If you have a draft and want it verified, go to 'critic'.
    - If the research is complete and ready for the final report, go to 'finalize'.
    """
    
    decision = await llm.ainvoke(prompt)
    
    # Update status mapping for the router
    is_complete = decision.next_node == "finalize"
    next_node_val = "end" if is_complete else decision.next_node
    
    return {
        "next_node": next_node_val,
        "instructions": decision.task_query,
        "history": [f"Orchestrator: {decision.thought}"],
        "is_complete": is_complete
    }
