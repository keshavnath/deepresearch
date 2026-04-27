from typing import List, Optional, Annotated, Literal
import operator
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class ResearchTask(BaseModel):
    query: str = Field(..., description="The search query to execute")
    reasoning: str = Field(..., description="Why this query is necessary")
    status: Literal["pending", "completed", "failed"] = "pending"
    result: Optional[str] = None

class Insight(BaseModel):
    title: str
    content: str
    sources: List[str]

class ResearchState(TypedDict):
    # Inputs
    query: str
    
    # Internal State
    plan: List[ResearchTask]
    context: Annotated[List[str], operator.add]  # Raw scraped content
    insights: Annotated[List[Insight], operator.add] # Processed reasoning
    
    # Orchestration
    next_node: str
    instructions: Optional[str]
    history: Annotated[List[str], operator.add]
    
    # Quality Control
    feedback: Optional[str]
    is_complete: bool

    # Final Output
    report: Optional[str]

class ResearchRequest(BaseModel):
    query: str
