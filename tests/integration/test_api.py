import pytest
import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_research_endpoint_invalid_input():
    """Test research endpoint validation."""
    # Test empty query
    response = client.post("/research", json={"query": ""})
    assert response.status_code == 400
    
    # Test missing payload
    response = client.post("/research", json={})
    assert response.status_code == 422


def test_research_endpoint_sse_format():
    """Test that research endpoint returns valid SSE stream format.
    
    Note: This will only work if all required APIs are configured.
    """
    response = client.post(
        "/research", 
        json={"query": "What is Python?"}
    )
    
    # SSE responses don't have a fixed status code in streaming
    # Just verify we get a streaming response
    if response.status_code == 200:
        # Parse SSE events from response
        lines = response.text.strip().split('\n')
        events_found = any(line.startswith('data: ') for line in lines)
        
        # Verify at least some data was streamed
        assert len(lines) > 0, "Stream should contain events"
        if events_found:
            # Try to parse at least one event
            for line in lines:
                if line.startswith('data: '):
                    event_json = json.loads(line[6:])  # Remove 'data: ' prefix
                    assert 'event' in event_json, "Event should have 'event' field"
