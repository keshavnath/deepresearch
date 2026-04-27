"""LangChain callbacks for tracing LLM and tool calls to WandB."""

import time
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from app.utils.tracer import get_tracer


class WandBTracingCallback(BaseCallbackHandler):
    """LangChain callback handler that logs LLM calls to WandB via tracer."""

    def __init__(self, trace_llm_calls: bool = True, trace_tool_calls: bool = True):
        self.trace_llm_calls = trace_llm_calls
        self.trace_tool_calls = trace_tool_calls
        self.tracer = get_tracer()
        self._start_times: Dict[str, float] = {}

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts processing."""
        if not self.trace_llm_calls:
            return
        
        model = serialized.get("id", ["unknown"])[-1]
        self._start_times["llm_start"] = time.time()
        # Log is done in on_llm_end with full response

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM finishes and returns a response."""
        if not self.trace_llm_calls:
            return
        
        latency_ms = (time.time() - self._start_times.get("llm_start", time.time())) * 1000
        
        # Extract model name and response
        model = response.llm_output.get("model", "unknown") if response.llm_output else "unknown"
        
        # Get prompt from kwargs (passed by LangChain)
        prompts = kwargs.get("prompts", [])
        prompt = prompts[0] if prompts else ""
        
        # Extract response text
        response_text = ""
        if response.generations and len(response.generations) > 0:
            if len(response.generations[0]) > 0:
                response_text = response.generations[0][0].text
        
        self.tracer.log_llm_call(
            model=model,
            prompt=prompt,
            response=response_text,
            latency_ms=latency_ms,
        )

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM call fails."""
        self.tracer.log_error(
            message=str(error),
            error_type="llm_error",
            context={"kwargs": str(kwargs)[:200]},
        )

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Called when a tool starts execution."""
        if not self.trace_tool_calls:
            return
        
        tool_name = serialized.get("name", "unknown_tool")
        self._start_times[f"tool_{tool_name}"] = time.time()

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool finishes execution."""
        if not self.trace_tool_calls:
            return
        
        tool_name = kwargs.get("tool_name", "unknown_tool")
        latency_ms = (time.time() - self._start_times.get(f"tool_{tool_name}", time.time())) * 1000
        
        self.tracer.log_tool_call(
            tool_name=tool_name,
            input_args={},  # Would need more context from LangChain to extract
            output=output,
            latency_ms=latency_ms,
        )

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when a tool fails."""
        tool_name = kwargs.get("tool_name", "unknown_tool")
        self.tracer.log_error(
            message=str(error),
            error_type="tool_error",
            context={"tool_name": tool_name},
        )
