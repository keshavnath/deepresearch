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
try:
    import wandb, weave
    
    if WANDB_ENABLED:
        # Set API key before init
        if WANDB_API_KEY:
            wandb.login(WANDB_API_KEY)
        
        # Initialize Weave
        project = WANDB_PROJECT or "deepresearch"
        try:
            weave.init(project_name=project)
            print(f"Weave initialized for project '{project}'")
        except Exception as e:
            print(f"Warning: Weave init failed: {e}")
    
    # Export weave.op as weave_op decorator
    weave_op = weave.op

except ImportError:
    # Fallback: no-op decorator if weave not installed
    def weave_op(name=None):
        def noop_decorator(func):
            return func
        return noop_decorator
    
    print("Warning: weave not installed. Tracing disabled.")
