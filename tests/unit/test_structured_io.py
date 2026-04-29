import pytest
from pydantic import BaseModel
from typing import Literal
from app.utils.llm import LLMWrapper, get_llm
from app.config import MODEL_API_KEY, TAVILY_MAX_PARALLEL, LLM_MAX_PARALLEL
from app.utils.rate_limiter import TAVILY_SEMAPHORE, LLM_SEMAPHORE


class Person(BaseModel):
    name: str
    age: int
    sex: Literal["male", "female", "other"]


class FakeResp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Fake LLM that tracks the prompt it receives."""
    def __init__(self):
        self.last_prompt = None
    
    async def ainvoke(self, prompt: str):
        self.last_prompt = prompt
        return FakeResp('{"name":"Alice","age":30,"sex":"female"}')  # minimal JSON


@pytest.mark.asyncio
async def test_llm_person_parse_fake():
    """Test structured output with a fake LLM that returns minimal JSON."""
    wrapped = LLMWrapper(FakeLLM())
    result = await wrapped.ainvoke("irrelevant prompt", schema=Person)

    # Parser returns a pydantic model instance
    assert isinstance(result, Person)
    assert result.name == "Alice"
    assert result.age == 30
    assert result.sex == "female"


@pytest.mark.asyncio
async def test_llm_format_instructions_injected():
    """Test that format instructions from schema are actually injected into prompt."""
    fake_llm = FakeLLM()
    wrapped = LLMWrapper(fake_llm)
    
    _ = await wrapped.ainvoke("Generate a person", schema=Person)
    
    # Verify format instructions were prepended to prompt
    assert fake_llm.last_prompt is not None
    assert "name" in fake_llm.last_prompt, "Prompt should contain schema field 'name'"
    assert "age" in fake_llm.last_prompt, "Prompt should contain schema field 'age'"
    assert "sex" in fake_llm.last_prompt, "Prompt should contain schema field 'sex'"
    # Original prompt should still be there
    assert "Generate a person" in fake_llm.last_prompt


@pytest.mark.asyncio
@pytest.mark.skipif(not MODEL_API_KEY, reason="No MODEL_API_KEY configured for real LLM test")
async def test_llm_person_parse_real():
    """Integration test: verify the real LLM respects the Person schema.
    
    This test passes only a simple prompt; the schema provides all format instructions.
    """
    llm = get_llm()

    prompt = "Provide a short person profile."
    result = await llm.ainvoke(prompt, schema=Person)

    # Validate that returned object conforms to the Person schema (types & allowed values)
    assert isinstance(result, Person)
    assert isinstance(result.name, str)
    assert isinstance(result.age, int)
    assert result.sex in ("male", "female", "other")


@pytest.mark.asyncio
async def test_rate_limiting_semaphores_initialized():
    """Verify rate limiting semaphores are properly initialized."""
    assert TAVILY_SEMAPHORE._value == TAVILY_MAX_PARALLEL, "Tavily semaphore should allow X concurrent calls"
    assert LLM_SEMAPHORE._value == LLM_MAX_PARALLEL, "LLM semaphore should allow X concurrent calls"