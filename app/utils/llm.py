import os
from typing import Optional, Type, Any
from langchain_openai import ChatOpenAI
from app.config import MODEL_NAME, MODEL_URL, MODEL_API_KEY
from langchain_core.output_parsers.pydantic import PydanticOutputParser


class LLMWrapper:
    """Unified LLM wrapper that supports both plain text and structured (schema-based) outputs.
    
    All LLM calls are automatically traced by WandB Weave.
    
    Usage:
        llm = get_llm()
        
        # Plain text response
        text = await llm.ainvoke("What is the capital of France?")
        
        # Structured output with schema
        person = await llm.ainvoke("Create a person", schema=Person)
    """
    def __init__(self, llm: Any):
        self._llm = llm

    async def ainvoke(self, prompt: str, schema: Optional[Type[Any]] = None):
        """Invoke the LLM with an optional schema for structured output.
        
        Args:
            prompt: The input prompt for the LLM
            schema: Optional Pydantic model class for structured output.
                   If provided, format instructions are prepended to prompt and output is parsed.
        
        Returns:
            Parsed Pydantic object if schema provided, otherwise raw text string.
        """
        if schema is None:
            # Plain text mode: invoke and return content
            resp = await self._llm.ainvoke(prompt)
            text = getattr(resp, "content", None)
            if text is None and hasattr(resp, "generations"):
                gens = resp.generations
                if gens and len(gens) and len(gens[0]):
                    text = getattr(gens[0][0], "text", None)
            if text is None:
                text = str(resp)
            return text
        
        # Structured output mode: inject schema format instructions and parse
        parser = PydanticOutputParser(pydantic_object=schema)
        try:
            fmt = parser.get_format_instructions()
        except Exception:
            fmt = None

        full_prompt = prompt
        if fmt:
            full_prompt = f"{prompt}\n\n{fmt}"

        resp = await self._llm.ainvoke(full_prompt)
        
        # Extract text content from common response shapes
        text = getattr(resp, "content", None)
        if text is None and hasattr(resp, "generations"):
            gens = resp.generations
            if gens and len(gens) and len(gens[0]):
                text = getattr(gens[0][0], "text", None)
        if text is None:
            text = str(resp)
        
        return parser.parse(text)


def get_llm(model_name: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None) -> LLMWrapper:
    """Get a unified LLM wrapper instance.
    
    Pass schemas at invocation time via ainvoke(prompt, schema=YourSchema).
    
    Args:
        model_name: Model to use (defaults to MODEL_NAME from config)
        base_url: API base URL (defaults to MODEL_URL from config)
        api_key: API key (defaults to MODEL_API_KEY from config)
    
    Returns:
        LLMWrapper instance that supports both plain text and structured output.
    """
    model = model_name or MODEL_NAME
    url = base_url or MODEL_URL
    key = api_key or MODEL_API_KEY

    llm = ChatOpenAI(
        model=model,
        openai_api_base=url,
        openai_api_key=key,
        temperature=0
    )

    return LLMWrapper(llm)