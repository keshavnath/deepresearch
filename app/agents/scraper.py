import asyncio
import trafilatura
from app.schema import ResearchState, ScrapedPage
from app.utils.tracer import weave_op

async def fetch_and_clean(url: str) -> ScrapedPage | None:
    """Fetch and extract content from a URL. Returns None if fetch/extract fails."""
    try:
        # Run in executor if trafilatura is synchronous
        downloaded = await asyncio.to_thread(trafilatura.fetch_url, url)
        if not downloaded:
            return None
        
        content = await asyncio.to_thread(trafilatura.extract, downloaded)
        if not content:
            return None
            
        return ScrapedPage(url=url, content=content)
    except Exception as e:
        # Log error silently, return None so pipeline can skip this URL
        return None

@weave_op("scraper")
async def scrape_node(state: ResearchState) -> dict:
    # Only scrape URLs not already fetched
    existing_urls = {p.url for p in state.get("scraped_pages", [])}
    
    # Priority search results (sorted by score if available)
    sorted_results = sorted(state["search_results"], key=lambda x: x.score, reverse=True)
    
    new_urls = [r.url for r in sorted_results 
                if r.url not in existing_urls][:10]  # cap at 10
    
    if not new_urls:
        return {"events": ["Scrape: No new URLs to fetch."]}

    tasks = [fetch_and_clean(url) for url in new_urls]
    results = await asyncio.gather(*tasks)
    
    # Filter out None values (failed fetches)
    valid_pages = [p for p in results if p is not None]
    
    return {
        "scraped_pages": valid_pages,
        "events": [f"Scrape: Extracted content from {len(valid_pages)} new pages."]
    }
