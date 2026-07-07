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
from api.agents.tools import get_formatted_context, get_formatted_review_context
from api.agents.agents import agent_node, intent_router_node
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

def tool_router(state: State) -> str:

    if state.iteration > 5:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    elif state.final_answer:
        return "end"
    else:
        return "end"
    
def intent_router_conditional_edge(state: State):

    if state.question_relevant:
        return "agent_node"
    else:
        return "END"

workflow = StateGraph(State)

tools = [get_formatted_context, get_formatted_review_context]
tool_node = ToolNode(tools)
tool_desc = get_tool_description(tools)

workflow.add_node("intent", intent_router_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tool_node", tool_node)

workflow.add_edge(START, "intent")
workflow.add_conditional_edges("intent", intent_router_conditional_edge, {"agent_node": "agent", "END": END})
workflow.add_conditional_edges("agent", tool_router, {"tools": "tool_node", "end": END})
workflow.add_edge("tool_node", "agent")

def run_agent(question: str, thread_id: str)->dict:
    initial_state = State(
        messages=[{"role": "user", "content": question}],
        available_tools=tool_desc,
        iteration=0
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
            if tool_name == "retrieve_items_context":
                return f"Looking for items: {tool_args.get('query', '')}."
            elif tool_name == "retrieve_review_context":
                return f"Fetching user reviews..."

            return f"Calling tool {tool_name}..."

        if _is_node_start(chunk):
            node_name = chunk[1].get("payload", {}).get("name")
            if node_name == "intent":
                print("Analysing the question...")
                return "Analysing the question..."
            if node_name == "agent":
                print("Planning next steps")
                return "Planning next steps"
            if node_name in ("tool_node", "mcp_tool_node"):
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
        available_tools=tool_desc,
        iteration=0
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
    
    yield _string_for_sse(json.dumps({
        "type": "final_answer",
        "data": {
            "answer": result.get("answer", ""),
            "used_context": used_context,
            "trace_id": result.get("trace_id", "")
        },
    }))    