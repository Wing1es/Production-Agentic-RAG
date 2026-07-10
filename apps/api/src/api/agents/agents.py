from pathlib import Path
from langchain_core.messages import convert_to_openai_messages, AIMessage
from groq import Groq
import instructor
from langsmith import traceable, get_current_run_tree

from api.core.config import config
from api.agents.models import ProductQAAgentResponse, State, CoordinatorAgentResponse, ShoppingCartAgentResponse
from api.agents.utils.prompt_management import prompt_template_config
from api.agents.utils.utils import format_ai_message

PROMPTS_DIR = Path(__file__).parent / "prompts"

@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={"ls_provider": "groq", "ls_model_name": "llama-3.3-70b-versatile"}
)
def product_qa_agent(state: State) -> dict:

    template = prompt_template_config(PROMPTS_DIR / "qa_agent.yaml", "qa_agent")

    prompt = template.render(
        available_tools=state.product_qa_agent.available_tools
    )

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_groq(Groq(api_key=config.GROQ_API_KEY), mode=instructor.Mode.JSON)

    response, raw_response = client.chat.completions.create_with_completion(
        model="llama-3.3-70b-versatile",
        response_model=ProductQAAgentResponse,
        messages=[
            {"role": "system", "content": prompt}, *conversation
        ],
        temperature=0.5,
    )

    run_tree = get_current_run_tree()

    if run_tree:
        run_tree.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }
 
    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "answer": response.answer,
        "product_qa_agent": {
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.product_qa_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.product_qa_agent.available_tools
        },
        "references": response.references
    }

@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={"ls_provider": "groq", "ls_model_name": "llama-3.3-70b-versatile"}
)
def shopping_cart_agent(state: State) -> dict:

    template = prompt_template_config(PROMPTS_DIR / "shopping_cart_agent.yaml", "shopping_cart_agent")

    prompt = template.render(
        available_tools=state.shopping_cart_agent.available_tools,
        user_id=state.user_id,
        cart_id=state.cart_id
    )

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_groq(Groq(api_key=config.GROQ_API_KEY), mode=instructor.Mode.JSON)

    response, raw_response = client.chat.completions.create_with_completion(
        model="llama-3.3-70b-versatile",
        response_model=ShoppingCartAgentResponse,
        messages=[
            {"role": "system", "content": prompt}, *conversation
        ],
        temperature=0.5,
    )

    run_tree = get_current_run_tree()

    if run_tree:
        run_tree.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }
 
    ai_message = format_ai_message(response)
    print(ai_message)

    return {
        "messages": [ai_message],
        "answer": response.answer,
        "shopping_cart_agent": {
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.shopping_cart_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.shopping_cart_agent.available_tools
        }
    }


@traceable(
    name="coordinator_agent",
    run_type="llm",
    metadata={"ls_provider": "groq", "ls_model_name": "llama-3.3-70b-versatile"}
)
def coordinator_agent(state: State) -> dict:

    template = prompt_template_config(PROMPTS_DIR / "coordinator_agent.yaml", "coordinator_agent")

    prompt = template.render()

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_groq(Groq(api_key=config.GROQ_API_KEY), mode=instructor.Mode.JSON)

    response, raw_response = client.chat.completions.create_with_completion(
        response_model=CoordinatorAgentResponse,
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": prompt}, *conversation]
    )

    run_tree = get_current_run_tree()

    if run_tree:
        run_tree.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }
    
        trace_id = str(getattr(run_tree, "trace_id", run_tree.id))
    
    else:
        trace_id = None

    if response.final_answer:
        ai_message = [AIMessage(content=response.answer)]
    else:
        ai_message = []

    return {
        "messages": ai_message,
        "answer": response.answer,
        "coordinator_agent": {
            "iteration": state.coordinator_agent.iteration + 1,
            "final_answer": response.final_answer,
            "next_agent": response.next_agent,
            "plan": response.plan
        },
        "trace_id": trace_id
    }