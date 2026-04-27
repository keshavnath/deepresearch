from typing import List, Optional, Annotated, Literal
import operator
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class SearchResult(BaseModel):
    url: str
    title: str
    content: str  # snippet from search engine
    score: float

class ScrapedPage(BaseModel):
    url: str
    content: str
    title: Optional[str] = None
    metadata: Optional[dict] = None

class Finding(BaseModel):
    question: str
    answer: str
    confidence: float = Field(ge=0, le=1)
    source_urls: List[str]

class CritiqueResult(BaseModel):
    satisfied: bool
    gaps: List[str]
    new_sub_questions: List[str]

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
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
