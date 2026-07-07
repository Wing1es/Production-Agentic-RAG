from fastmcp import FastMCP
from reviews_mcp_server.utils import retrieve_review_data, process_review_context

mcp = FastMCP("review_mcp_server")

@mcp.tool(
    name="retrieve_review_data",
    description="Retrieve the top k context, each representing an inventory item for a given query"
)
def get_formatted_review_context(query: str, item_list: list, k: int = 2):
    """
    Retrieve and format top k customer reviews matching a specific query for a list of product IDs (parent ASINs).

    Args:
        query: The search query to find relevant reviews (e.g. "battery life", "sound quality")
        item_list: List of product IDs (parent ASINs) to search reviews for. Example: ["B09ZPV8WBV"]
        top_k: Number of review text chunks to retrieve.

    Returns:
        A formatted string of the retrieved customer reviews containing the product IDs and review texts.
    """

    context = retrieve_review_data(query, item_list, k)
    formatted = process_review_context(context)

    return formatted

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8002)