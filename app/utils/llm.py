import os
from typing import Optional
from langchain_openai import ChatOpenAI
from app.config import MODEL_NAME, MODEL_URL, MODEL_API_KEY

def get_llm(model_name: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None):
    """
    Returns a ChatModel instance based on the provided name or env defaults.
    Supports OpenAI-compatible APIs.
    """
    # Use provided values or fall back to config
    model = model_name or MODEL_NAME
    url = base_url or MODEL_URL
    key = api_key or MODEL_API_KEY

    return ChatOpenAI(
        model=model,
        openai_api_base=url,
        openai_api_key=key,
        temperature=0
    )

