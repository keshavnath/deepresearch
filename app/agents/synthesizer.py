import asyncio
from app.schema import ResearchState, Finding
from app.utils.llm import get_llm

SYNTHESIZE_PROMPT = """
Answer the following question based ONLY on the provided research context.
If the context doesn't contain the answer, state that.
Question: {question}

Context:
{context}

Respond with a clear answer, a confidence score (0-1), and the source URLs used.
"""

async def synthesize_one(question: str, context: str) -> Finding:
    llm = get_llm()
    result = await llm.ainvoke(SYNTHESIZE_PROMPT.format(question=question, context=context), schema=Finding)
    return result

async def synthesizer_node(state: ResearchState) -> dict:
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
    
    return {
        "findings": list(findings),
        "events": ["Synthesis: Generated findings for all sub-questions."]
    }
