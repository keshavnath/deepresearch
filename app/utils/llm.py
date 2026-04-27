import os
from typing import Optional, Type, Any
from langchain_openai import ChatOpenAI
from app.config import MODEL_NAME, MODEL_URL, MODEL_API_KEY

def get_llm(schema: Optional[Type[Any]] = None, model_name: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None):
    """
    Returns a ChatOpenAI instance configured for the specific model provider,
    optionally bound to a structured output schema.
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

    if schema:
        return llm.with_structured_output(schema, method="json_mode")

    return llm

