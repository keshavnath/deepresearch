import asyncio
import logging
from app.schema import ResearchState, Finding
from app.utils.llm import get_llm
from app.utils.tracer import weave_op

logger = logging.getLogger(__name__)

SYNTHESIZE_PROMPT = """
Answer the following question based ONLY on the provided research context.
If the context doesn't contain the answer, state that clearly.
Question: {question}

Context:
{context}

Respond with a clear answer, a confidence score (0-1), and the source URLs used.
"""

@weave_op("synthesize_one")
async def synthesize_one(question: str, context: str, fallback_url: str = "") -> Finding:
    """Generate a finding from a question and research context.
    
    Args:
        question: Research question to answer
        context: Research context (may be empty)
        fallback_url: URL to use if LLM finds no sources (optional)
        
    Returns:
        Finding object with answer, confidence, and sources
        
    Raises:
        Exception: If LLM call fails
    """
    # Edge case: Empty context
    if not context or not context.strip():
        logger.warning(f"Synthesizing with empty context for: {question}")
        context = "(No research context available)"
    
    try:
        llm = get_llm()
        result = await llm.ainvoke(
            SYNTHESIZE_PROMPT.format(question=question, context=context), 
            schema=Finding
        )
        
        # Validate result confidence is in valid range
        if not (0.0 <= result.confidence <= 1.0):
            logger.warning(f"Invalid confidence {result.confidence}, clamping to 0.5")
            result.confidence = 0.5
        
        # Ensure at least one source if we have fallback
        if not result.source_urls and fallback_url:
            result.source_urls = [fallback_url]
        
        return result
    except Exception as e:
        logger.error(f"Synthesize failed for '{question}': {e}")
        raise

@weave_op("synthesizer")
async def synthesizer_node(state: ResearchState) -> dict:
    """Synthesize findings from scraped content.
    
    For each sub-question, generates a structured finding with answer,
    confidence score, and supporting sources.
    
    Args:
        state: ResearchState with scraped_pages and sub_questions
        
    Returns:
        dict with 'findings' (list of Finding) and 'events'
        
    Raises:
        Propagates LLM errors; caught by event_generator
    """
    if not state.get("sub_questions"):
        logger.warning("No sub-questions to synthesize")
        return {
            "findings": [],
            "events": ["Synthesis: No sub-questions to answer."]
        }
    
    scraped_pages = state.get("scraped_pages", [])
    
    # Edge case: No scraped content
    if not scraped_pages:
        logger.warning("No scraped content available for synthesis")
        context = "(No research context - no pages were successfully scraped)"
    else:
        # Truncate and format context to fit window
        context_parts = []
        for page in scraped_pages:
            context_parts.append(f"Source: {page.url}\nContent: {page.content[:3000]}")
        
        context = "\n---\n".join(context_parts)[:20000]  # Hard limit
        
        if not context.strip():
            logger.warning("Context truncated to empty string")
            context = "(Research context too large to process)"
    
    # Get first URL as fallback if synthesis has no sources
    fallback_url = scraped_pages[0].url if scraped_pages else ""
    
    try:
        tasks = [
            synthesize_one(question=q, context=context, fallback_url=fallback_url)
            for q in state["sub_questions"]
        ]
        findings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions from gather
        valid_findings = []
        for finding in findings:
            if isinstance(finding, Exception):
                logger.error(f"Synthesis task failed: {finding}")
            else:
                valid_findings.append(finding)
        
        if not valid_findings:
            logger.error("All synthesis tasks failed")
            raise RuntimeError("Failed to generate any findings")
        
        return {
            "findings": valid_findings,
            "events": [f"Synthesis: Generated {len(valid_findings)} findings."]
        }
    except Exception as e:
        logger.error(f"Synthesizer node failed: {e}")
        raise
