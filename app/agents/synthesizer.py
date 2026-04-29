import asyncio
import logging
from app.schema import ResearchState, Finding
from app.utils.llm import get_llm
from app.utils.tracer import weave_op
from app.utils.rag import BM25PassageIndex

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
async def synthesize_one(question: str, passage_index: BM25PassageIndex = None, fallback_url: str = "") -> Finding:
    """Generate a finding from a question using BM25-retrieved context.
    
    Args:
        question: Research question to answer
        passage_index: BM25PassageIndex for retrieving relevant passages (optional)
        fallback_url: URL to use if LLM finds no sources (optional)
        
    Returns:
        Finding object with answer, confidence, and sources
        
    Raises:
        Exception: If LLM call fails
    """
    # Retrieve relevant passages for this specific question
    context = ""
    if passage_index:
        context = passage_index.retrieve_for_query(question, top_k=5)
    
    # Edge case: No relevant context found
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
    """Synthesize findings from scraped content using BM25-based retrieval.
    
    For each sub-question, retrieves relevant passages via BM25 and generates
    a structured finding with answer, confidence score, and supporting sources.
    
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
    passage_index = None
    
    # Try to build BM25 index; if it fails, we'll fall back to None
    if scraped_pages:
        try:
            passage_index = BM25PassageIndex(scraped_pages)
            logger.info(f"BM25 index created for {len(scraped_pages)} pages")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index, falling back to direct context: {e}")
            passage_index = None
    else:
        logger.warning("No scraped content available for synthesis")
    
    # Get first URL as fallback if synthesis has no sources
    fallback_url = scraped_pages[0].url if scraped_pages else ""
    
    try:
        tasks = [
            synthesize_one(question=q, passage_index=passage_index, fallback_url=fallback_url)
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
