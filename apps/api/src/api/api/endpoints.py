from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import StreamingResponse
from .models import RAGRequest, RAGResponse, RAGUsedContext
from api.agents.graph import agent_wrapper, rag_agent_stream_wrapper
from api.api.processors.submit_feedback import submit_feedback
from api.api.models import FeedbackRequest, FeedbackResponse

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
feedback_router = APIRouter()

@rag_router.post("/")
def chat(request: Request, payload: RAGRequest) -> StreamingResponse:
    logger.info(f"Received chat request: {payload}")
    
    # response = agent_wrapper(payload.query, payload.thread_id)

    # logger.info(f"LLM Response: {response['answer']}")

    return StreamingResponse(
        rag_agent_stream_wrapper(payload.query, payload.thread_id),
        media_type="text/event-stream"
    )
    # return RAGResponse(
    #     request_id=request.state.request_id, 
    #     answer=response["answer"],
    #     used_context=[RAGUsedContext(**item) for item in response["used_context"]],
    #     trace_id=response["trace_id"]
    # )

@feedback_router.post("/")
def send_feedback(request: Request, payload: FeedbackRequest):
    submit_feedback(
        trace_id=payload.trace_id,
        feedback_score=payload.feedback_score,
        feedback_text=payload.feedback_text if payload.feedback_text is not None else "",
        feedback_source_type=payload.feedback_source_type
    )
    return FeedbackResponse(
        request_id=request.state.request_id,
        status="success"
    )

api_router = APIRouter()
api_router.include_router(rag_router, prefix="/agent", tags=["agent"])
api_router.include_router(feedback_router, prefix="/submit_feedback", tags=["feedback"])