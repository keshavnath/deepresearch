from tavily import TavilyClient
from app.schema import ResearchState
from app.config import TAVILY_API_KEY

def get_tavily_client():
    return TavilyClient(api_key=TAVILY_API_KEY)

async def searcher_node(state: ResearchState):
    """
    Worker node that executes searches using Tavily.
    """
    query = state.get("instructions")
    if not query:
        return {"next_node": "orchestrator", "history": ["Searcher skipped: No query provided"]}
    
    client = get_tavily_client()
    # Using advanced search for deep research
    search_result = client.search(query, search_depth="advanced", max_results=5)
    
    # Extract URLs for the scraper
    urls = [r["url"] for r in search_result.get("results", [])]
    summary = "\n".join([f"- {r['title']}: {r['url']}" for r in search_result.get("results", [])])
    
    return {
        "next_node": "orchestrator",
        "history": [f"Searcher found {len(urls)} results for: {query}\n{summary}"],
        "context": [summary] # We'll append titles/urls to context for now
    }
