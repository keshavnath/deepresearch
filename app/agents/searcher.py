import asyncio
import logging
from app.schema import ResearchState, SearchResult
from app.config import TAVILY_API_KEY
from app.utils.tracer import weave_op
from app.utils.rate_limiter import rate_limit_tavily
from tavily import TavilyClient

logger = logging.getLogger(__name__)

def get_tavily_client():
    return TavilyClient(api_key=TAVILY_API_KEY)

async def search_one(query: str, client: TavilyClient) -> list[SearchResult]:
    """Execute a single search query with rate limiting.
    
    Args:
        query: Search query string
        client: TavilyClient instance
        
    Returns:
        List of SearchResult objects (may be empty if no results found)
        
    Raises:
        Exception: If Tavily API fails (handled by caller)
    """
    try:
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
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        raise

@weave_op("searcher")
async def search_node(state: ResearchState) -> dict:
    """Search for relevant sources using Tavily API.
    
    Decomposes the query into sub-questions and searches for each in parallel.
    Deduplicates results by URL and returns combined list.
    
    Args:
        state: ResearchState with sub_questions and existing search_results
        
    Returns:
        dict with 'search_results' (list of new SearchResult) and 'events'
        
    Raises:
        Propagates Tavily API errors; caught by event_generator
    """
    if not state.get("sub_questions"):
        logger.warning("No sub-questions to search")
        return {
            "search_results": [],
            "events": ["Search: No sub-questions provided."]
        }
    
    client = get_tavily_client()
    
    try:
        # Execute searches in parallel
        tasks = [search_one(q, client) for q in state["sub_questions"]]
        results_batches = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle potential exceptions from gather
        new_results = []
        errors = []
        for i, batch in enumerate(results_batches):
            if isinstance(batch, Exception):
                errors.append(f"Query {i}: {batch}")
            elif batch:
                new_results.extend(batch)
        
        if errors:
            logger.warning(f"Some searches failed: {errors}")
        
        # Flatten and deduplicate by URL
        seen_urls = set()
        deduplicated = []
        for r in new_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                deduplicated.append(r)
        
        # Edge case: No results found at all
        if not deduplicated:
            logger.warning("No search results found for any sub-question")
            return {
                "search_results": [],
                "events": ["Search: No results found. Proceeding with limited information."]
            }
        
        return {
            "search_results": deduplicated,
            "events": [f"Search: Found {len(deduplicated)} relevant sources."]
        }
        
    except Exception as e:
        logger.error(f"Search node failed: {e}")
        raise
