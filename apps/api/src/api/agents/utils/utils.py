# pyrefly: ignore [missing-import]
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

#### FORMAT AI MESSAGE ####

def format_ai_message(response):

    if response.tool_calls:
        tool_calls = []
        for i, tc in enumerate(response.tool_calls):
            tool_calls.append({
                "id": f"call_{i}",
                "name": tc.name,
                "args": tc.arguments
            })

        ai_message = AIMessage(
            content=response.answer,
            tool_calls=tool_calls
        )

    else:
        ai_message = AIMessage(
            content=response.answer,
        )

    return ai_message

def get_tool_description(tools):

    descs = []

    for tool in tools:
        descs.append(
            convert_to_openai_tool(tool)
        )
    return descs
