"""Pytest configuration: Initialize Weave tracing before any tests run."""

# Import tracer FIRST to ensure weave.init() runs before test collection
import app.utils.tracer # noqa: F401
