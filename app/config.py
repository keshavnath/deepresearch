import os
from dotenv import dotenv_values

# Load .env into a dict and apply to os.environ manually or just use the dict
# Using os.path.join to find .env relative to project root (2 levels up from current file)
config = {
    **dotenv_values(),  # load local .env
    **os.environ,       # override with system environment variables
}

# LLM Configuration
MODEL_NAME = config.get("MODEL_NAME", "gpt-4o")
MODEL_URL = config.get("MODEL_URL", None)  # None defaults to OpenAI's standard API
MODEL_API_KEY = config.get("MODEL_API_KEY", config.get("OPENAI_API_KEY"))

# Search & Tools Configuration
TAVILY_API_KEY = config.get("TAVILY_API_KEY")

# WandB Tracing & Observability
WANDB_ENABLED = config.get("WANDB_ENABLED", "True").lower() == "true"
WANDB_PROJECT = config.get("WANDB_PROJECT", "deepresearch")
WANDB_API_KEY = config.get("WANDB_API_KEY")

# App Configuration
APP_NAME = "Deep Research Multi-Agent System"
DEBUG = config.get("DEBUG", "False").lower() == "true"
MAX_ITERATIONS = int(config.get("MAX_ITERATIONS", 2))
