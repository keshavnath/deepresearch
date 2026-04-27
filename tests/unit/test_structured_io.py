import pytest
from pydantic import BaseModel
from typing import Literal
from app.utils.llm import LLMWrapper, get_llm
from app.config import MODEL_API_KEY


class Person(BaseModel):
    name: str
    age: int
    sex: Literal["male", "female", "other"]


class FakeResp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    async def ainvoke(self, prompt: str):
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
