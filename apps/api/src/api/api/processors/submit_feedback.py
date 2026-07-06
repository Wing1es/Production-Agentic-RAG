from langsmith import Client
from api.core.config import config

client = Client(api_key=config.LANGSMITH_API_KEY)

def submit_feedback(trace_id: str, feedback_score: int | None = None, feedback_text: str = "", feedback_source_type="api")->None:
    if feedback_score:
        client.create_feedback(
            trace_id=trace_id,
            key="thumbs",
            score=feedback_score,
            feedback_source_type=feedback_source_type
        )
    
    if len(feedback_text) > 0:
        client.create_feedback(
            trace_id=trace_id,
            key="comment",
            value=feedback_text,
            feedback_source_type=feedback_source_type
        )