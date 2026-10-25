from typing import List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description="Use kb for questions related to IT/support; direct for greetings/simple chat."
    )

class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description="good means evidence can answer the question; weak means not enough evidence."
    )

class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int
    citations: List[dict]
    trace: List[str]