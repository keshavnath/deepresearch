"""BM25-based keyword retrieval for synthesizer.

Provides fast, lightweight passage retrieval without embeddings.
Uses rank-bm25 library (TF-IDF variant) to rank passages by relevance.
"""

import logging
import re
from typing import List
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25PassageIndex:
    """Build and query a BM25 index over passages from scraped pages."""
    
    def __init__(self, pages: List) -> None:
        """Initialize BM25 index from scraped pages.
        
        Args:
            pages: List of ScrapedPage objects with .url and .content
        """
        self.pages = pages
        self.passages = []  # List of (passage_text, source_url)
        self.bm25 = None
        
        self._build_index()
    
    def _tokenize_simple(self, text: str) -> List[str]:
        """Simple tokenization: split on whitespace and remove punctuation.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        # Convert to lowercase and split on whitespace
        tokens = text.lower().split()
        
        # Remove leading/trailing punctuation from each token
        cleaned = []
        for token in tokens:
            # Strip punctuation from edges
            token = re.sub(r'^[^\w]+|[^\w]+$', '', token)
            if token:  # Only keep non-empty
                cleaned.append(token)
        
        return cleaned
    
    def _split_into_passages(self, text: str, max_passage_len: int = 500) -> List[str]:
        """Split text into passages by sentence boundaries.
        
        Args:
            text: Text to split
            max_passage_len: Max characters per passage
            
        Returns:
            List of passage strings
        """
        # Simple sentence split on period/newline
        sentences = re.split(r'[.\n]+', text)
        
        passages = []
        current_passage = []
        current_len = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_len = len(sentence)
            
            # Start new passage if current would exceed limit
            if current_len + sentence_len > max_passage_len and current_passage:
                passages.append(' '.join(current_passage))
                current_passage = []
                current_len = 0
            
            current_passage.append(sentence)
            current_len += sentence_len + 1
        
        # Add final passage
        if current_passage:
            passages.append(' '.join(current_passage))
        
        return passages
    
    def _build_index(self) -> None:
        """Build BM25 index from pages."""
        try:
            tokenized_passages = []
            
            for page in self.pages:
                # Split page into passages
                passages = self._split_into_passages(page.content)
                
                for passage in passages:
                    if passage.strip():
                        self.passages.append({
                            'text': passage,
                            'url': page.url,
                            'tokens': self._tokenize_simple(passage)
                        })
                        tokenized_passages.append(self.passages[-1]['tokens'])
            
            if not tokenized_passages:
                logger.warning("No passages indexed - empty pages")
                self.bm25 = None
                return
            
            # Initialize BM25Okapi
            self.bm25 = BM25Okapi(tokenized_passages)
            logger.info(f"BM25 index built: {len(self.passages)} passages from {len(self.pages)} pages")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25 = None
    
    def retrieve_for_query(self, query: str, top_k: int = 5) -> str:
        """Retrieve top-k relevant passages for a query.
        
        Args:
            query: Query string
            top_k: Number of top passages to retrieve
            
        Returns:
            Formatted string of top passages with source citations,
            or empty string if retrieval fails
        """
        if not self.bm25 or not self.passages:
            logger.debug("BM25 index empty, returning empty context")
            return ""
        
        try:
            query_tokens = self._tokenize_simple(query)
            
            if not query_tokens:
                logger.debug("Query has no tokens after tokenization")
                return ""
            
            # Get BM25 scores for all passages
            scores = self.bm25.get_scores(query_tokens)
            
            # Get indices of top-k passages
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]
            
            # Build result string with sources
            result_parts = []
            for idx in top_indices:
                passage = self.passages[idx]
                result_parts.append(
                    f"[Source: {passage['url']}]\n{passage['text']}"
                )
            
            result = "\n---\n".join(result_parts)
            
            if result.strip():
                logger.debug(f"Retrieved {len(result_parts)} passages for query: {query[:50]}")
                return result
            
            return ""
            
        except Exception as e:
            logger.error(f"BM25 retrieval failed for query '{query}': {e}")
            return ""
