from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

# --- Entity Models ---

class UserContext(BaseModel):
    """Structured details retrieved from your database or auth context."""
    user_id: str
    username: str
    role: str
    organization_id: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)


class LLMResponseContext(BaseModel):
    """Structured output tracking the LLM metadata and actual content."""
    content: str
    model_name: str  # e.g., 'gpt-4o', 'claude-3-5-sonnet'
    token_usage: Dict[str, int] = Field(default_factory=dict) # e.g., {"prompt": 120, "completion": 45}
    latency_ms: Optional[float] = None


# --- Combined Interaction Model ---

class LLMInteraction(BaseModel):
    """The complete context payload mapping a user query to an LLM response."""
    interaction_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user: UserContext
    raw_query: str
    llm_response: Optional[LLMResponseContext] = None
    metadata: Dict[str, Any] = Field(default_factory=dict) # Catch-all for RAG chunk IDs, filters, etc.

# Create a reusable "blank" base model for generic events
class TextEvent(BaseModel): pass