from typing import List, Optional, Annotated, Literal
import operator
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class SearchResult(BaseModel):
    url: str = Field(description="The URL of the search result")
    title: str = Field(description="The title of the search result")
    content: str = Field(description="A brief snippet or excerpt from the page content")
    score: float = Field(description="Relevance score of the search result (0.0-1.0)")

class ScrapedPage(BaseModel):
    url: str = Field(description="The URL of the scraped page")
    content: str = Field(description="The full text content of the page")
    title: Optional[str] = Field(None, description="The title of the page, if available")
    metadata: Optional[dict] = Field(None, description="Any additional metadata about the page")

class Finding(BaseModel):
    question: str = Field(description="The sub-question being answered")
    answer: str = Field(description="A concise, factual answer to the question based on the provided sources")
    confidence: float = Field(ge=0, le=1, description="Confidence level (0.0-1.0) in the accuracy of this answer")
    source_urls: List[str] = Field(description="List of URLs that support and substantiate this answer")

class CritiqueResult(BaseModel):
    satisfied: bool = Field(description="Whether the research sufficiently answers the original query")
    gaps: List[str] = Field(description="List of identified gaps or missing information in the research")
    new_sub_questions: List[str] = Field(description="List of new sub-questions that should be researched to fill gaps")

class ResearchState(TypedDict):
    # Input
    query: str
    
    # Orchestrator output
    sub_questions: List[str]
    research_plan: str
    
    # Search/Scrape outputs
    search_results: Annotated[List[SearchResult], operator.add]   # raw results per sub-question
    scraped_pages: Annotated[List[ScrapedPage], operator.add]     # full text of fetched URLs
    
    # Synthesis outputs
    findings: List[Finding]              # one per sub-question
    
    # Critic outputs
    critique: Optional[CritiqueResult]    # gaps, satisfied bool
    
    # Final
    report: Optional[str]
    sources: List[str]
    
    # Control
    iteration: int
    events: Annotated[List[str], operator.add]                    # SSE event log

class ResearchRequest(BaseModel):
    query: str
