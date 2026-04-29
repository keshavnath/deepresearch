import logging
from app.schema import ResearchState
from app.utils.llm import get_llm
from app.utils.tracer import weave_op

logger = logging.getLogger(__name__)

REPORT_PROMPT = """
You are a Lead Researcher. Write a final, comprehensive markdown report based on the findings and research plan.
Include citations (URLs) for all key facts.

Original Query: {query}
Research Plan: {plan}

Findings:
{findings}

Structure:
# [Title]
## Executive Summary
## Detailed Findings
## Conclusion
## Sources

Important: Include all source URLs in a Sources section at the end.
"""

@weave_op("reporter")
async def report_node(state: ResearchState) -> dict:
    """Generate final research report from accumulated findings.
    
    Creates a structured markdown report with executive summary, findings,
    conclusions, and citations.
    
    Args:
        state: ResearchState with query, research_plan, and findings
        
    Returns:
        dict with 'report' (markdown string), 'sources' (list of URLs), and 'events'
        
    Raises:
        RuntimeError: If no findings or no report generated
    """
    findings = state.get("findings", [])
    
    # Edge case: No findings to report
    if not findings:
        logger.error("Reporter: No findings to include in report")
        raise RuntimeError("Cannot generate report: no findings available")
    
    try:
        llm = get_llm()
        
        findings_str = "\n---\n".join([
            f"**Q: {f.question}**\n\nA: {f.answer}\n\nConfidence: {f.confidence:.1%}\n\nSources: {', '.join(f.source_urls) if f.source_urls else 'N/A'}"
            for f in findings
        ])
        
        report = await llm.ainvoke(REPORT_PROMPT.format(
            query=state["query"],
            plan=state.get("research_plan", "Comprehensive research"),
            findings=findings_str
        ))
        
        # Validate report was generated
        if not report or not report.strip():
            logger.error("Reporter: LLM returned empty report")
            raise RuntimeError("Failed to generate report: LLM returned empty response")
        
        # Extract unique sources
        sources = list(set([url for f in findings for url in f.source_urls]))
        
        if not sources:
            logger.warning("Reporter: No sources found in findings")
        
        logger.info(f"Reporter: Generated report ({len(report)} chars, {len(sources)} sources)")
        
        return {
            "report": report,
            "sources": sources,
            "events": ["Reporter: Final report generated."]
        }
        
    except Exception as e:
        logger.error(f"Reporter node failed: {e}")
        raise
