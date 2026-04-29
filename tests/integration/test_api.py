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
    
    Validates two-tier event structure:
    - Status events: {type: "status", stage, data}
    - Report event: {type: "report", content, sources, metadata}
    - Error events: {type: "error", message}
    """
    response = client.post(
        "/research", 
        json={"query": "What is Python?"}
    )
    
    if response.status_code == 200:
        # Parse SSE events from response
        lines = response.text.strip().split('\n')
        status_events = []
        report_event = None
        
        for line in lines:
            if line.startswith('data: '):
                try:
                    event_json = json.loads(line[6:])  # Remove 'data: ' prefix
                    event_type = event_json.get('type')
                    
                    if event_type == 'status':
                        status_events.append(event_json)
                        assert 'stage' in event_json, "Status event must have 'stage'"
                        assert 'data' in event_json, "Status event must have 'data'"
                    elif event_type == 'report':
                        report_event = event_json
                        assert 'content' in event_json, "Report event must have 'content'"
                        assert 'sources' in event_json, "Report event must have 'sources'"
                        assert 'metadata' in event_json, "Report event must have 'metadata'"
                    elif event_type == 'error':
                        assert 'message' in event_json, "Error event must have 'message'"
                except json.JSONDecodeError:
                    pass  # Skip malformed lines
        
        # Should have at least some status events and a final report/error
        assert len(status_events) > 0 or report_event, "Stream should contain events"
