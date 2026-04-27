"""WandB Weave integration for automatic LLM and function tracing.

Weave automatically traces:
- LLM calls (input/output/latency)
- Function execution (@weave_op decorated functions)
- Nested call hierarchies
- Errors and exceptions

Usage:
    from app.utils.tracer import weave_op
    
    @weave_op("my_function")
    async def my_function(query: str) -> str:
        result = await llm.ainvoke(query)
        return result
"""

import os
from app.config import WANDB_ENABLED, WANDB_PROJECT, WANDB_API_KEY

# Import weave and initialize once
_weave_module = None
_weave_initialized = False

try:
    import wandb
    import weave
    _weave_module = weave
    
    if WANDB_ENABLED:
        # Set API key before init
        if WANDB_API_KEY:
            wandb.login(WANDB_API_KEY)
        
        # Initialize Weave
        project = WANDB_PROJECT or "deepresearch"
        try:
            weave.init(project_name=project)
            _weave_initialized = True
            print(f"Weave initialized for project '{project}'")
        except Exception as e:
            print(f"Weave init failed: {e}")

except ImportError as e:
    print(f"Weave import failed: {e}. Tracing disabled.")


def weave_op(name=None):
    """Decorator for automatic Weave tracing.
    
    Args:
        name: Optional name for the operation. If not provided, function name is used.
    
    Returns:
        Decorator function that applies weave.op if available, otherwise returns function unchanged.
    """
    def decorator(func):
        if _weave_module is None:
            # Weave not available, return function unchanged
            return func
        
        try:
            # Apply weave decorator - use keyword argument for safety
            return _weave_module.op(name=name)(func)
        except Exception as e:
            # If decorator application fails, return function unchanged but log
            print(f"Warning: Failed to apply weave.op({name}) to {func.__name__}: {e}")
            return func
    
    return decorator
