"""WandB tracing and observability for the deep research system.

Provides structured logging of:
- Agent state transitions
- LLM calls (prompt, model, response)
- Tool executions
- Errors and warnings

Usage:
    tracer = get_tracer()
    tracer.log_agent_start(node_name="orchestrator", input_state=state)
    tracer.log_llm_call(model="gpt-4", prompt=prompt, response=response)
    tracer.log_agent_end(node_name="orchestrator", output=result)
"""

import time
import json
from typing import Any, Dict, Optional
from datetime import datetime
from app.config import WANDB_ENABLED, WANDB_PROJECT


class Tracer:
    """Unified tracer for logging traces and observability to WandB."""

    def __init__(self, enabled: bool = True, project: Optional[str] = None):
        self.enabled = enabled
        self.project = project
        self._trace_stack = []  # Stack of active traces for hierarchical logging
        self._trace_id_counter = 0
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                if not self.wandb.run:
                    self.wandb.init(project=project or "deepresearch", tags=["tracing"])
            except ImportError:
                print("Warning: wandb not installed. Tracing disabled.")
                self.enabled = False
                self.wandb = None

    def _log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Internal method to log events to WandB."""
        if not self.enabled or not self.wandb:
            return
        
        # Add timestamp and trace stack info
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "trace_depth": len(self._trace_stack),
            **data,
        }
        
        # Add current trace context if available
        if self._trace_stack:
            payload["trace_id"] = self._trace_stack[-1]["trace_id"]
            payload["trace_path"] = " → ".join(t["name"] for t in self._trace_stack)
        
        # Log to WandB as a nested dict under the event type
        self.wandb.log({f"trace/{event_type}": payload})

    def log_agent_start(self, node_name: str, input_state: Optional[Dict] = None) -> str:
        """Log the start of an agent node execution.
        
        Args:
            node_name: Name of the agent node (e.g., "orchestrator", "critic")
            input_state: State passed to the node
            
        Returns:
            trace_id for this agent execution
        """
        self._trace_id_counter += 1
        trace_id = f"{node_name}_{self._trace_id_counter}_{int(time.time() * 1000)}"
        
        trace_context = {
            "trace_id": trace_id,
            "name": node_name,
            "start_time": time.time(),
        }
        self._trace_stack.append(trace_context)
        
        self._log("agent_start", {
            "node_name": node_name,
            "input_keys": list(input_state.keys()) if input_state else [],
            "input_summary": str(input_state)[:500] if input_state else None,
        })
        
        return trace_id

    def log_agent_end(self, node_name: str, output: Optional[Dict] = None, error: Optional[str] = None) -> None:
        """Log the end of an agent node execution.
        
        Args:
            node_name: Name of the agent node
            output: Output/result from the node
            error: Error message if execution failed
        """
        if self._trace_stack and self._trace_stack[-1]["name"] == node_name:
            trace_context = self._trace_stack.pop()
            duration = time.time() - trace_context["start_time"]
            
            self._log("agent_end", {
                "node_name": node_name,
                "duration_seconds": duration,
                "output_keys": list(output.keys()) if output else [],
                "error": error,
                "success": error is None,
            })

    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Log an LLM API call.
        
        Args:
            model: Model name (e.g., "gpt-4o")
            prompt: Input prompt
            response: Model response
            tokens_used: Estimated tokens used
            latency_ms: Latency in milliseconds
        """
        self._log("llm_call", {
            "model": model,
            "prompt_length": len(prompt),
            "prompt_preview": prompt[:200],
            "response_length": len(response),
            "response_preview": response[:200],
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
        })

    def log_tool_call(
        self,
        tool_name: str,
        input_args: Dict[str, Any],
        output: Any,
        error: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Log a tool/function call.
        
        Args:
            tool_name: Name of the tool
            input_args: Arguments passed to the tool
            output: Result from the tool
            error: Error message if tool failed
            latency_ms: Execution time in milliseconds
        """
        self._log("tool_call", {
            "tool_name": tool_name,
            "input_args": str(input_args)[:500],
            "output": str(output)[:500],
            "error": error,
            "success": error is None,
            "latency_ms": latency_ms,
        })

    def log_state_transition(self, state_name: str, before: Dict, after: Dict) -> None:
        """Log a state transition.
        
        Args:
            state_name: Name/description of the state
            before: State before change
            after: State after change
        """
        # Summarize what changed
        changed_keys = {k for k in before if before.get(k) != after.get(k)} | set(after.keys()) - set(before.keys())
        
        self._log("state_transition", {
            "state_name": state_name,
            "changed_keys": list(changed_keys),
            "num_changes": len(changed_keys),
        })

    def log_error(self, message: str, error_type: Optional[str] = None, context: Optional[Dict] = None) -> None:
        """Log an error or warning.
        
        Args:
            message: Error message
            error_type: Type of error (e.g., "validation_error", "api_error")
            context: Additional context
        """
        self._log("error", {
            "message": message,
            "error_type": error_type or "unknown",
            "context": str(context)[:500] if context else None,
        })

    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Optional step/iteration number
        """
        if self.enabled and self.wandb:
            log_dict = {f"metrics/{metric_name}": value}
            if step is not None:
                log_dict["step"] = step
            self.wandb.log(log_dict)


# Global tracer instance
_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get or create the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(enabled=WANDB_ENABLED, project=WANDB_PROJECT)
    return _tracer


def init_tracer(enabled: bool = True, project: Optional[str] = None) -> Tracer:
    """Initialize the tracer with custom settings."""
    global _tracer
    _tracer = Tracer(enabled=enabled, project=project)
    return _tracer
