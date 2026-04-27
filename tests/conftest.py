"""Pytest configuration: Initialize Weave tracing before any tests run."""

# Import tracer FIRST to ensure weave.init() runs before test collection
import app.utils.tracer  # noqa: F401

def pytest_configure(config):
    """Called before test collection - ensures weave is initialized first."""
    # Tracer already imported above, but this hook runs super early
    pass
