import json
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, MatchValue, Filter
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamMode

from api.core.config import config
from api.agents.models import State
from api.agents.utils.utils import get_tool_description
from api.agents.tools import get_formatted_context, get_formatted_review_context, add_to_shopping_cart, get_shopping_cart, remove_from_cart
from api.agents.agents import product_qa_agent, shopping_cart_agent, coordinator_agent
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

def product_qa_agent_tool_router(state: State) -> str:

    if state.product_qa_agent.final_answer:
        return "end"
    elif state.product_qa_agent.iteration > 4:
        return "end"
    elif len(state.product_qa_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"

def shopping_cart_agent_tool_router(state: State) -> str:

    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > 2:
        return "end"
    elif len(state.shopping_cart_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"
    
def coordinator_agent_edge(state: State):

    if state.coordinator_agent.iteration > 3:
        return "END"
    elif state.coordinator_agent.final_answer and len(state.coordinator_agent.plan) == 0:
        return "END"
    if state.coordinator_agent.next_agent in ("product_qa", "product_qa_agent"):
        return "product_qa_agent"
    elif state.coordinator_agent.next_agent in ("shopping_cart", "shopping_cart_agent"):
        return "shopping_cart_agent"
    else:
        return "END"

workflow = StateGraph(State)

product_qa_agent_tools = [get_formatted_context, get_formatted_review_context]
product_qa_agent_tool_node = ToolNode(product_qa_agent_tools)
product_qa_agent_tool_description = get_tool_description(product_qa_agent_tools)

shopping_cart_agent_tools = [add_to_shopping_cart, remove_from_cart, get_shopping_cart]
shopping_cart_agent_tool_node = ToolNode(shopping_cart_agent_tools)
shopping_cart_agent_tool_description = get_tool_description(shopping_cart_agent_tools)

workflow.add_node("coordinator_agent", coordinator_agent)
workflow.add_node("product_qa_agent", product_qa_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)

workflow.add_node("product_qa_agent_tool_node", product_qa_agent_tool_node)
workflow.add_node("shopping_cart_agent_tool_node", shopping_cart_agent_tool_node)

workflow.add_edge(START, "coordinator_agent")
workflow.add_conditional_edges("coordinator_agent", coordinator_agent_edge, {"product_qa_agent": "product_qa_agent", "shopping_cart_agent": "shopping_cart_agent", "END": END})
workflow.add_conditional_edges("product_qa_agent", product_qa_agent_tool_router, {"tools": "product_qa_agent_tool_node", "end": "coordinator_agent"})
workflow.add_conditional_edges("shopping_cart_agent", shopping_cart_agent_tool_router, {"tools": "shopping_cart_agent_tool_node", "end": "coordinator_agent"})
workflow.add_edge("product_qa_agent_tool_node", "product_qa_agent")
workflow.add_edge("shopping_cart_agent_tool_node", "shopping_cart_agent")

def run_agent(question: str, thread_id: str)->dict:
    initial_state = State(
        messages=[{"role": "user", "content": question}],
        user_id=thread_id,
        cart_id=thread_id,
        product_qa_agent={
            "available_tools": product_qa_agent_tool_description,
            "iteration": 0,
            "final_answer": False,
            "tool_calls": []
        },
        shopping_cart_agent={
            "available_tools": shopping_cart_agent_tool_description,
            "iteration": 0,
            "final_answer": False,
            "tool_calls": []
        },
        coordinator_agent={
            "iteration": 0,
            "final_answer": False,
            "next_agent": "",
            "plan": []
        }
    )

    run_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    with PostgresSaver.from_conn_string(
        conn_string=config.POSTGRES_CONNECTION_STRING
    ) as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        result = graph.invoke(initial_state, run_config)

    return result

def agent_wrapper(question: str, thread_id: str):

    qdrant = QdrantClient(url=config.QDRANT_URL)

    result = run_agent(question, thread_id)

    used_context = []

    for item in result.get("references", []):
        records, _ = qdrant.scroll(
            collection_name="Amazon-collection-01-hybrid-search",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if records:
            payload = records[0].payload
            image_url = payload.get("image")
            price = payload.get("price")

            if image_url: 
                used_context.append({
                    "image_url": image_url,
                    "price": price,
                    "description": item.description
                })
    
    return {
        "answer": result.get("answer", ""),
        "used_context": used_context,
        "trace_id": result.get("trace_id", "")
    }

def rag_agent_stream_wrapper(question: str, thread_id: str):

    def _string_for_sse(message: str) -> str:
        return f"data: {message}\n\n"
     
    def _process_graph_event(chunk):

        def _is_node_start(chunk):
            return chunk[1].get("type", "") == "task"
        
        def _is_node_end(chunk):
            return chunk[0] == "updates"
        
        def _tool_to_text(tool):
            tool_name = getattr(tool, "name", "") or (tool.get("name", "") if isinstance(tool, dict) else "")
            tool_args = getattr(tool, "arguments", {}) or (tool.get("arguments", {}) if isinstance(tool, dict) else {})
            if tool_name == "get_formatted_context":
                return f"Searching product catalog for: '{tool_args.get('query', '')}'..."
            elif tool_name == "get_formatted_review_context":
                return f"Analyzing user reviews for matches: '{tool_args.get('query', '')}'..."
            elif tool_name == "add_to_shopping_cart":
                return f"Adding items to your shopping cart..."
            elif tool_name == "remove_from_cart":
                return f"Removing item from your shopping cart..."
            elif tool_name == "get_shopping_cart":
                return f"Retrieving your shopping cart..."

            return f"Calling database tool {tool_name}..."

        if _is_node_start(chunk):
            node_name = chunk[1].get("payload", {}).get("name")
            if node_name == "coordinator_agent":
                print("Planning next steps...")
                return "Planning next steps..."
            elif node_name == "product_qa_agent":
                print("Analyzing products...")
                return "Analyzing products..."
            elif node_name == "shopping_cart_agent":
                print("Updating shopping cart...")
                return "Updating shopping cart..."
            elif node_name in ("product_qa_agent_tool_node", "shopping_cart_agent_tool_node", "tool_node", "mcp_tool_node"):
                input_data = chunk[1].get("payload", {}).get("input")
                tool_calls = []
                if input_data:
                    if hasattr(input_data, "tool_calls"):
                        tool_calls = input_data.tool_calls
                    elif isinstance(input_data, dict):
                        tool_calls = input_data.get("tool_calls", [])
                
                tool_texts = [
                    _tool_to_text(tool)
                    for tool in tool_calls
                    if tool is not None
                ]
                message = " ".join(text for text in tool_texts if text is not None)
                if message:
                    return message
        return None
    
    qdrant = QdrantClient(url=config.QDRANT_URL)

    initial_state = State(
        messages=[{"role": "user", "content": question}],
        user_id=thread_id,
        cart_id=thread_id,
        product_qa_agent={
            "available_tools": product_qa_agent_tool_description,
            "iteration": 0,
            "final_answer": False,
            "tool_calls": []
        },
        shopping_cart_agent={
            "available_tools": shopping_cart_agent_tool_description,
            "iteration": 0,
            "final_answer": False,
            "tool_calls": []
        },
        coordinator_agent={
            "iteration": 0,
            "final_answer": False,
            "next_agent": "",
            "plan": []
        }
    )

    run_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    stream_modes: list[StreamMode] = ["debug", "tasks"]

    with PostgresSaver.from_conn_string(
        conn_string=config.POSTGRES_CONNECTION_STRING
    ) as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)

        for chunk in graph.stream(
            initial_state,
            config=run_config,
            stream_mode=stream_modes
        ):
            processed_chunk = _process_graph_event(chunk)

            if processed_chunk:
                yield _string_for_sse(processed_chunk)
        
        final_state = graph.get_state(run_config)
        result = final_state.values or {}
    
    used_context = []

    for item in result.get("references", []):
        records, _ = qdrant.scroll(
            collection_name="Amazon-collection-01-hybrid-search",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if records:
            payload = records[0].payload
            if payload is not None:
                image_url = payload.get("image")
                price = payload.get("price")

                if image_url: 
                    used_context.append({
                        "image_url": image_url,
                        "price": price,
                        "description": item.description
                    })

    shopping_cart = get_shopping_cart(thread_id, thread_id)

    shopping_cart_tems = [
        {
            "price": float(item.get("price")) if item.get("price", ) else None,
            "quantity": item.get("quantity"),
            "currency": item.get("currency"),
            "product_image_url": item.get("product_image_url"),
            "total_price": float(item.get("total_price")) if item.get("total_price", ) else None
        }
        for item in shopping_cart
    ]
    
    yield _string_for_sse(json.dumps({
        "type": "final_answer",
        "data": {
            "answer": result.get("answer", ""),
            "used_context": used_context,
            "shopping_cart": shopping_cart_tems,
            "trace_id": result.get("trace_id", "")
        },
    }))    