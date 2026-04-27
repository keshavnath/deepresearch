import asyncio
import trafilatura
from app.schema import ResearchState, ScrapedPage

async def fetch_and_clean(url: str) -> ScrapedPage:
    try:
        # Run in executor if trafilatura is synchronous
        downloaded = await asyncio.to_thread(trafilatura.fetch_url, url)
        if not downloaded:
            raise ValueError("Empty download")
        
        content = await asyncio.to_thread(trafilatura.extract, downloaded)
        if not content:
            raise ValueError("No text extracted")
            
        return ScrapedPage(url=url, content=content)
    except Exception as e:
        return e

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
    
    valid_pages = [p for p in results if isinstance(p, ScrapedPage)]
    
    return {
        "scraped_pages": valid_pages,
        "events": [f"Scrape: Extracted content from {len(valid_pages)} new pages."]
    }
