from app.schema import ResearchState
from app.utils.llm import get_llm
from app.utils.tracer import get_tracer

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

async def report_node(state: ResearchState) -> dict:
    tracer = get_tracer()
    tracer.log_agent_start("reporter", input_state={"query": state["query"]})
    
    try:
        llm = get_llm()
        
        findings_str = "\n---\n".join([f"Q: {f.question}\nA: {f.answer}\nSources: {f.source_urls}" for f in state["findings"]])
        
        report = await llm.ainvoke(REPORT_PROMPT.format(
            query=state["query"],
            plan=state["research_plan"],
            findings=findings_str
        ))
        
        # Extract unique sources
        sources = list(set([url for f in state["findings"] for url in f.source_urls]))
        
        output = {
            "report": report.content,
            "sources": sources,
            "events": ["Report: Final report generated."]
        }
        
        tracer.log_agent_end("reporter", output=output)
        return output
    except Exception as e:
        tracer.log_agent_end("reporter", error=str(e))
        raise
