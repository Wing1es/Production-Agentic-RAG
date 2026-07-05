from fastapi import FastAPI, Request, APIRouter
from .models import RAGRequest, RAGResponse, RAGUsedContext
from api.agents.graph import agent_wrapper

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("maven_api.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

rag_router = APIRouter()

@rag_router.post("/")
def chat(request: Request, payload: RAGRequest) -> RAGResponse:
    logger.info(f"Received chat request: {payload}")
    
    response = agent_wrapper(payload.query)

    logger.info(f"LLM Response: {response['answer']}")
    return RAGResponse(
        request_id=request.state.request_id, 
        answer=response["answer"],
        used_context=[RAGUsedContext(**item) for item in response["used_context"]]
    )

api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"]) 