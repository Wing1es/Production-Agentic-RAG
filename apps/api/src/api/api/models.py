from pydantic import BaseModel, Field
from typing import List, Optional, Union

class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be answered") 
    thread_id: str = Field(..., description="The ID of the thread")

class RAGUsedContext(BaseModel):
    image_url: str = Field(..., description="The URL of the image of the item")
    price: Optional[float] = Field(..., description="Price of the item")
    description: str = Field(..., description="The description of the item")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    answer: str = Field(..., description="The final response to the user's query")
    used_context: List[RAGUsedContext] = Field(..., description="Information used to answer the question")
    trace_id: str = Field(..., description="The ID of the trace")

class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., description="The ID of the trace")
    thread_id: Optional[str] = Field(default=None, description="The ID of the thread")
    feedback_score: Optional[int] = Field(default=None, description="The score of the feedback")
    feedback_text: Optional[str] = Field(default=None, description="The text of the feedback")
    feedback_source_type: Optional[str] = Field(default=None, description="The type of the feedback")

class FeedbackResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    status: str = Field(..., description="The status of the request")