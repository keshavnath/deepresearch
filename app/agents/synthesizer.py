import asyncio
from app.schema import ResearchState, Finding
from app.utils.llm import get_llm
from app.utils.tracer import get_tracer

SYNTHESIZE_PROMPT = """
Answer the following question based ONLY on the provided research context.
If the context doesn't contain the answer, state that.
Question: {question}

Context:
{context}

Respond with a clear answer, a confidence score (0-1), and the source URLs used.
"""

async def synthesize_one(question: str, context: str) -> Finding:
    tracer = get_tracer()
    tracer.log_agent_start("synthesizer", input_state={"question": question})
    
    try:
        llm = get_llm()
        result = await llm.ainvoke(SYNTHESIZE_PROMPT.format(question=question, context=context), schema=Finding)
        tracer.log_agent_end("synthesizer", output={"answer": result.answer})
        return result
    except Exception as e:
        tracer.log_agent_end("synthesizer", error=str(e))
        raise

async def synthesizer_node(state: ResearchState) -> dict:
    tracer = get_tracer()
    tracer.log_agent_start("synthesizer_node", input_state={"num_questions": len(state["sub_questions"])})
    
    try:
        # Truncate and format context to fit window
        context_parts = []
        for page in state["scraped_pages"]:
            context_parts.append(f"Source: {page.url}\nContent: {page.content[:3000]}")
        
        full_context = "\n---\n".join(context_parts)[:20000] # Hard limit
        
        tasks = [
            synthesize_one(question=q, context=full_context)
            for q in state["sub_questions"]
        ]
        findings = await asyncio.gather(*tasks)
        
        output = {
            "findings": list(findings),
            "events": ["Synthesis: Generated findings for all sub-questions."]
        }
        
        tracer.log_agent_end("synthesizer_node", output=output)
        return output
    except Exception as e:
        tracer.log_agent_end("synthesizer_node", error=str(e))
        raise
