from pydantic import BaseModel, Field
from typing import List, Optional

class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be answered")

class RAGUsedContext(BaseModel):
    image_url: str = Field(..., description="The URL of the image of the item")
    price: Optional[float] = Field(..., description="Price of the item")
    description: str = Field(..., description="The description of the item")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    answer: str = Field(..., description="The final response to the user's query")
    used_context: List[RAGUsedContext] = Field(..., description="Information used to answer the question")