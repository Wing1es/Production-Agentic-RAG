from pydantic import BaseModel, Field
from typing import Annotated, List, Any, Dict
from operator import add

class RAGUsedContext(BaseModel):
    id: str = Field(description="The id of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class RAGResponse(BaseModel):
    answer: str = Field(description='Answer to the question')
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")

class ToolCall(BaseModel):
    name: str = ""
    arguments: dict

class AgentProperties(BaseModel):
    iteration: int = 0
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []
    final_answer: bool = False

class Delegation(BaseModel):
    agent: str
    task: str

class CoordinatorAgentProperties(BaseModel):
    iteration: int = 0
    plan: List[Delegation] = []
    next_agent: str = ""
    final_answer: bool = False

class State(BaseModel):
    messages: Annotated[List[Any], add]
    user_intent: str = ""
    product_qa_agent: AgentProperties = Field(default_factory=AgentProperties)
    shopping_cart_agent: AgentProperties = Field(default_factory=AgentProperties)
    coordinator_agent: CoordinatorAgentProperties = Field(default_factory=CoordinatorAgentProperties)
    answer: str = ""
    references: Annotated[List[RAGUsedContext], add] = []
    user_id: str = ""
    cart_id: str = ""

class ProductQAAgentResponse(BaseModel):
    answer: str
    references: List[RAGUsedContext]
    final_answer: bool = False
    tool_calls: List[ToolCall] = []

class CoordinatorAgentResponse(BaseModel):
    next_agent: str
    plan: List[Delegation]
    final_answer: bool = False
    answer: str

class ShoppingCartAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []