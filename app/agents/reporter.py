from app.schema import ResearchState
from app.utils.llm import get_llm
from app.utils.tracer import weave_op

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
"""

@weave_op("reporter")
async def report_node(state: ResearchState) -> dict:
    llm = get_llm()
    
    findings_str = "\n---\n".join([f"Q: {f.question}\nA: {f.answer}\nSources: {f.source_urls}" for f in state["findings"]])
    
    report = await llm.ainvoke(REPORT_PROMPT.format(
        query=state["query"],
        plan=state["research_plan"],
        findings=findings_str
    ))
    
    # Extract unique sources
    sources = list(set([url for f in state["findings"] for url in f.source_urls]))
    
    return {
        "report": report.content,
        "sources": sources,
        "events": ["Report: Final report generated."]
    }
