"""Tests for BM25-based passage retrieval."""

import pytest
from app.schema import ScrapedPage
from app.utils.rag import BM25PassageIndex


def test_bm25_index_retrieval():
    """Test that BM25 retrieves relevant passages for queries."""
    # Create mock scraped pages
    pages = [
        ScrapedPage(
            url="https://example.com/quantum",
            content="Quantum computing uses quantum bits. Qubits can be in superposition. Quantum entanglement is key.",
            title="Quantum Computing"
        ),
        ScrapedPage(
            url="https://example.com/fashion",
            content="The latest fashion trends include bold colors. Spring fashions are coming. Designers love patterns.",
            title="Fashion Trends"
        ),
    ]
    
    # Build index
    index = BM25PassageIndex(pages)
    assert index.bm25 is not None, "BM25 index should be initialized"
    assert len(index.passages) > 0, "Should have indexed passages"
    
    # Query about quantum (should retrieve quantum passages)
    result = index.retrieve_for_query("quantum computing", top_k=3)
    assert result, "Should return non-empty result for quantum query"
    assert "quantum" in result.lower(), "Result should contain quantum-related content"
    
    # Query about fashion (should retrieve fashion passages)
    result = index.retrieve_for_query("fashion trends", top_k=3)
    assert result, "Should return non-empty result for fashion query"
    assert "fashion" in result.lower(), "Result should contain fashion-related content"


def test_bm25_empty_pages():
    """Test BM25 index with empty pages list."""
    pages = []
    index = BM25PassageIndex(pages)
    
    # Should gracefully handle empty index
    assert index.bm25 is None, "BM25 should be None for empty pages"
    result = index.retrieve_for_query("test", top_k=5)
    assert result == "", "Should return empty string for empty index"


def test_bm25_source_attribution():
    """Test that retrieved passages include source URLs."""
    pages = [
        ScrapedPage(
            url="https://source1.com",
            content="Information about artificial intelligence and machine learning.",
            title="AI Guide"
        ),
    ]
    
    index = BM25PassageIndex(pages)
    result = index.retrieve_for_query("artificial intelligence", top_k=5)
    
    # Result should include source attribution
    assert "[Source:" in result, "Result should include source attribution"
    assert "https://source1.com" in result, "Result should include URL"
