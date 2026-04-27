import asyncio
from app.schema import ResearchState, SearchResult
from app.config import TAVILY_API_KEY
from app.utils.tracer import weave_op
from app.utils.rate_limiter import rate_limit_tavily
from tavily import TavilyClient

def get_tavily_client():
    return TavilyClient(api_key=TAVILY_API_KEY)

async def search_one(query: str, client: TavilyClient) -> list[SearchResult]:
    """Execute a single search with rate limiting."""
    # Run synchronous search in executor with rate limit
    async def _search():
        return await asyncio.to_thread(
            client.search, 
            query, 
            search_depth="advanced", 
            max_results=5
        )
    
    search_result = await rate_limit_tavily(_search())
    
    results = []
    for r in search_result.get("results", []):
        results.append(SearchResult(
            url=r["url"],
            title=r["title"],
            content=r.get("content", ""),
            score=r.get("score", 0.0)
        ))
    return results

@weave_op("searcher")
async def search_node(state: ResearchState) -> dict:
    client = get_tavily_client()
    
    # Execute searches in parallel
    tasks = [search_one(q, client) for q in state["sub_questions"]]
    results_batches = await asyncio.gather(*tasks)
    
    # Flatten and deduplicate by URL
    seen_urls = set()
    new_results = []
    for batch in results_batches:
        for r in batch:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                new_results.append(r)
    
    return {
        "search_results": new_results,
        "events": [f"Search: Found {len(new_results)} relevant sources."]
    }
