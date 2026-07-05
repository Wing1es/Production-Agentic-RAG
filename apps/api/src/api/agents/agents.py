from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_core.messages import convert_to_openai_messages
from groq import Groq
import instructor
from jinja2 import Template
from langsmith import traceable

from api.core.config import config
from api.agents.models import AgentResponse, IntentResponse, State
from api.agents.utils.prompt_management import prompt_template_config
from api.agents.utils.utils import format_ai_message

PROMPTS_DIR = Path(__file__).parent / "prompts"


def agent_node(state: State) -> dict:

    template = prompt_template_config(PROMPTS_DIR / "qa_agent.yaml", "qa_agent")

    prompt = template.render(
        available_tools=state.available_tools
    )

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_groq(Groq(api_key=config.GROQ_API_KEY))

    response, raw_response = client.chat.completions.create_with_completion(
        model="llama-3.3-70b-versatile",
        response_model=AgentResponse,
        messages=[
            {"role": "system", "content": prompt}, *conversation
        ],
        temperature=0.5,
    )
 
    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "references": response.references
    }



@traceable(
    name="agent_router_node",
    run_type="llm",
    metadata={"ls_provider": "groq", "ls_model_name": "llama-3.3-70b-versatile"}
)
def intent_router_node(state: State) -> dict:

    template = prompt_template_config(PROMPTS_DIR / "intent_router_agent.yaml", "intent_router_agent")

    prompt = template.render(0)

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_groq(Groq(api_key=config.GROQ_API_KEY))

    response, raw_response = client.chat.completions.create_with_completion(
        response_model=IntentResponse,
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": prompt}, *conversation]
    )

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer
    } 