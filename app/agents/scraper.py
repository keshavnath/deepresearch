import trafilatura
from app.schema import ResearchState

async def scraper_node(state: ResearchState):
    """
    Worker node that scrapes content from URLs found in context.
    """
    # In a real scenario, the orchestrator might pass specific URLs in 'instructions'
    # or we look at the last history entry for URLs.
    # For now, let's assume 'instructions' contains a URL or the searcher just ran.
    
    url = state.get("instructions")
    if not url or not url.startswith("http"):
        return {"next_node": "orchestrator", "history": ["Scraper found no valid URL to scrape"]}
    
    downloaded = trafilatura.fetch_url(url)
    content = trafilatura.extract(downloaded)
    
    if not content:
        return {"next_node": "orchestrator", "history": [f"Scraper failed to extract content from {url}"]}
    
    return {
        "next_node": "orchestrator",
        "context": [f"Source: {url}\nContent: {content[:2000]}..."], # Slice to avoid context overflow for now
        "history": [f"Scraped content from {url}"]
    }
