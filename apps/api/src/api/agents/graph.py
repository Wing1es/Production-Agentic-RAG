from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, MatchValue, Filter
from langgraph.checkpoint.postgres import PostgresSaver

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
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "available_tools": tool_desc,
        "iteration": 0
    }

    run_config = {
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