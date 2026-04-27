from langgraph.graph import StateGraph, START, END
from app.schema import ResearchState
from app.agents.orchestrator import orchestrator_node
from app.agents.searcher import search_node
from app.agents.scraper import scrape_node
from app.agents.synthesizer import synthesizer_node
from app.agents.critic import critic_node
from app.agents.reporter import report_node
from app.config import MAX_ITERATIONS

def route_after_critique(state: ResearchState) -> str:
    critique = state.get("critique")
    iteration = state.get("iteration", 0)
    
    if (critique and critique.satisfied) or iteration >= MAX_ITERATIONS:
        return "write_report"
    return "search"

def get_graph():
    builder = StateGraph(ResearchState)

    # Add Nodes
    builder.add_node("orchestrate",   orchestrator_node)
    builder.add_node("search",        search_node)
    builder.add_node("scrape",        scrape_node)
    builder.add_node("synthesize",    synthesizer_node)
    builder.add_node("critique",      critic_node)
    builder.add_node("write_report",  report_node)

    # Set Entry Point
    builder.add_edge(START, "orchestrate")

    # Define Linear Flow
    builder.add_edge("orchestrate",  "search")
    builder.add_edge("search",       "scrape")
    builder.add_edge("scrape",       "synthesize")
    builder.add_edge("synthesize",   "critique")

    # Loop or Finalize
    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "search":       "search",
            "write_report": "write_report",
        }
    )

    builder.add_edge("write_report", END)

    return builder.compile()
