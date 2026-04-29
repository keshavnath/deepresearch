import asyncio
import logging
import trafilatura
from app.schema import ResearchState, ScrapedPage
from app.utils.tracer import weave_op

logger = logging.getLogger(__name__)

async def fetch_and_clean(url: str) -> ScrapedPage | None:
    """Fetch and extract content from a URL.
    
    Args:
        url: URL to fetch and scrape
        
    Returns:
        ScrapedPage if successful, None if fetch/extract fails
    """
    try:
        # Run in executor if trafilatura is synchronous
        downloaded = await asyncio.to_thread(trafilatura.fetch_url, url)
        if not downloaded:
            logger.debug(f"Failed to download {url}: empty response")
            return None
        
        content = await asyncio.to_thread(trafilatura.extract, downloaded)
        if not content:
            logger.debug(f"Failed to extract content from {url}")
            return None
            
        return ScrapedPage(url=url, content=content)
    except Exception as e:
        logger.debug(f"Scrape error for {url}: {e}")
        return None

@weave_op("scraper")
async def scrape_node(state: ResearchState) -> dict:
    """Scrape content from search result URLs.
    
    Fetches top-ranked URLs and extracts clean text content.
    Handles failures gracefully and returns only successfully scraped pages.
    
    Args:
        state: ResearchState with search_results and existing scraped_pages
        
    Returns:
        dict with 'scraped_pages' (list of new ScrapedPage) and 'events'
    """
    # Only scrape URLs not already fetched
    existing_urls = {p.url for p in state.get("scraped_pages", [])}
    
    # Handle case: no search results available
    if not state.get("search_results"):
        logger.warning("No search results to scrape")
        return {
            "scraped_pages": [],
            "events": ["Scrape: No URLs to scrape (no search results)."]
        }
    
    # Priority search results (sorted by score if available)
    sorted_results = sorted(state["search_results"], key=lambda x: x.score, reverse=True)
    
    new_urls = [r.url for r in sorted_results 
                if r.url not in existing_urls][:10]  # cap at 10
    
    if not new_urls:
        logger.info("All search result URLs already scraped")
        return {
            "scraped_pages": [],
            "events": ["Scrape: No new URLs to fetch."]
        }

    try:
        tasks = [fetch_and_clean(url) for url in new_urls]
        results = await asyncio.gather(*tasks)
        
        # Filter out None values (failed fetches)
        valid_pages = [p for p in results if p is not None]
        
        # Edge case: All scrapes failed
        if not valid_pages:
            logger.warning(f"Failed to scrape any of {len(new_urls)} URLs")
            return {
                "scraped_pages": [],
                "events": [f"Scrape: Failed to extract content from {len(new_urls)} URLs."]
            }
        
        success_rate = len(valid_pages) / len(new_urls)
        if success_rate < 0.3:
            logger.warning(f"Low scrape success rate: {success_rate:.1%}")
        
        return {
            "scraped_pages": valid_pages,
            "events": [f"Scrape: Extracted content from {len(valid_pages)}/{len(new_urls)} pages."]
        }
    except Exception as e:
        logger.error(f"Scraper node failed: {e}")
        raise
