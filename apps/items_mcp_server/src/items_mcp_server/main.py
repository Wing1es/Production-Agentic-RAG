from fastmcp import FastMCP
from items_mcp_server.utils import retrieve_data, process_context

mcp = FastMCP("items_mcp_server")

@mcp.tool(
    name="retrieve_items", 
    description="Retrieve the top k context for a given query"
)
def get_formatted_items_context(query: str, k: int = 2, query_type: str = "fusion"):
    """
    Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for
        k: The number of context chunks to retrieve, works best with 2. Use 1 for now cause of model limits.
        query_type: The type of query to use, either "formula" or "fusion"

    Returns:
        A formatted string containing the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """
    context = retrieve_data(query, k, query_type)
    format_text = process_context(context)

    return format_text

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8001)