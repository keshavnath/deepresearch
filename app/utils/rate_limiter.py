"""Minimal rate limiting for API calls using asyncio.Semaphore."""

import asyncio
from typing import Callable, Any, TypeVar
from app.config import TAVILY_MAX_PARALLEL, LLM_MAX_PARALLEL

F = TypeVar('F', bound=Callable[..., Any])

# Semaphores for different API services
TAVILY_SEMAPHORE = asyncio.Semaphore(TAVILY_MAX_PARALLEL)  # Max concurrent Tavily searches
LLM_SEMAPHORE = asyncio.Semaphore(LLM_MAX_PARALLEL)     # Max concurrent LLM calls


async def rate_limit_tavily(coro):
    """Apply rate limit to Tavily API calls."""
    async with TAVILY_SEMAPHORE:
        return await coro


async def rate_limit_llm(coro):
    """Apply rate limit to LLM API calls."""
    async with LLM_SEMAPHORE:
        return await coro
