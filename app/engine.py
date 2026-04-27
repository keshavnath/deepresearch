from langgraph.graph import StateGraph, START, END
from app.schema import ResearchState

from app.agents.orchestrator import orchestrator_node
from app.agents.searcher import searcher_node
from app.agents.scraper import scraper_node

async def reasoner_node(state: ResearchState):
    return {"next_node": "orchestrator", "history": ["Reasoner synthesized logic"]}

async def critic_node(state: ResearchState):
    return {"next_node": "orchestrator", "history": ["Critic verified output"]}

def router(state: ResearchState):
    """
    Skeleton router that merely follows the Orchestrator's decision.
    """
    if state.get("is_complete") or state.get("next_node") == "end":
        return END
    return state.get("next_node", "orchestrator")

def get_graph():
    workflow = StateGraph(ResearchState)
    
    # Define Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("searcher", searcher_node)
    workflow.add_node("scraper", scraper_node)
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("critic", critic_node)
    
    # Define Edges
    workflow.set_entry_point("orchestrator")
    
    workflow.add_conditional_edges(
        "orchestrator",
        router,
        {
            "searcher": "searcher",
            "scraper": "scraper",
            "reasoner": "reasoner",
            "critic": "critic",
            END: END
        }
    )
    
    # All workers return to orchestrator
    workflow.add_edge("searcher", "orchestrator")
    workflow.add_edge("scraper", "orchestrator")
    workflow.add_edge("reasoner", "orchestrator")
    workflow.add_edge("critic", "orchestrator")
    
    return workflow.compile()

