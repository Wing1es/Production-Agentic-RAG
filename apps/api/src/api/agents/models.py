from pydantic import BaseModel, Field
from typing import Annotated, List, Any, Dict
from operator import add

class RAGUsedContext(BaseModel):
    id: str = Field(description="The id of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class RAGResponse(BaseModel):
    answer: str = Field(description='Answer to the question')
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")

class IntentResponse(BaseModel):
    question_relevant: bool
    answer: str

class ToolCall(BaseModel):
    name: str = ""
    arguments: dict

class AgentResponse(BaseModel):
    answer: str
    references: List[RAGUsedContext]
    final_answer: bool = False
    tool_calls: List[ToolCall] = []

class State(BaseModel):
    messages: Annotated[List[Any], add]
    question_relevant: bool = False
    iteration: int = 0
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []
    answer: str = ""
    final_answer: bool = False
    references: Annotated[List[RAGUsedContext], add] = []
    trace_id: str = ""