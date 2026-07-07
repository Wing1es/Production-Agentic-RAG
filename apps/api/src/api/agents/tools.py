from fastembed import TextEmbedding
from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FormulaQuery, FusionQuery, SumExpression, MultExpression, Document
from qdrant_client.models import Filter, FieldCondition, MatchAny

model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={
        "ls_provider": "qdrant",
        "ls_model_name": "sentence-transformers/all-MiniLM-L6-v2"
    }
)
def get_embedding(text, model=model):
    tokens = model.token_count(text)
    run_tree = get_current_run_tree()
    run_tree.metadata["usage_metadata"] = tokens
    return list(model.embed([text]))[0]

@traceable(
    name="retrieve_items_context",
    run_type="retriever"
)
def retrieve_data(query: str, k: int = 2, query_type: str = "fusion"):
    """
    Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for
        k: The number of context chunks to retrieve, works best with 2. Use 1 for now cause of model limits.
        query_type: The type of query to use, either "formula" or "fusion"

    Returns:
        A dictionary containing the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """

    qdrant = QdrantClient(url="http://qdrant:6333")
    embedding = get_embedding(query)

    if query_type == "formula":
        fquery = FormulaQuery(
            formula=SumExpression(
                sum=[
                    MultExpression(
                        mult=[0.7, "$score[0]"]
                    ),
                    MultExpression(
                        mult=[0.3, "$score[1]"]
                    )
                ]
            )
        )
    else:
        fquery=FusionQuery(fusion="rrf")

    result = qdrant.query_points(
        collection_name="Amazon-collection-01-hybrid-search",
        prefetch=[
            Prefetch(
                query=embedding,
                using="embedding-sentence-piece",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25"
                ),
                using="BM25",
                limit=20
            )
        ],
        query=fquery,
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    retrieved_context_ratings = []
    similarity_scores = []

    for point in result.points:
        retrieved_context_ids.append(point.payload["parent_asin"])
        retrieved_context.append(point.payload["description"])
        retrieved_context_ratings.append(point.payload["average_rating"])
        similarity_scores.append(point.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context_ratings": retrieved_context_ratings,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores
    }

@traceable(
    name="process_context",
    run_type="prompt"
)
def process_context(context):
    format_text = ""

    for id, rating, context in zip(context["retrieved_context_ids"], context["retrieved_context_ratings"], context["retrieved_context"]):
        format_text += f"- ID: {id}, rating: {rating}, description: {context}\n"
    return format_text

def get_formatted_context(query: str, top_k: int = 2) -> str:
    """
Get the top k context, each representing an inventory item for a given query.

Args:
    query: The query to get the top k context for
    top_k: The number of context chunks to retrieve, works best with 2. Use 1 for now cause of model limits.

Returns:
    A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
"""

    context = retrieve_data(query, top_k)
    formatted = process_context(context)

    return formatted


@traceable(
    name="retrieve_review_context",
    run_type="retriever"
)
def retrieve_review_data(query: str, item_list: list, k: int = 2):
    """
    Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for
        k: The number of context chunks to retrieve, works best with 2. Use 1 for now cause of model limits.
        query_type: The type of query to use, either "formula" or "fusion"

    Returns:
        A dictionary containing the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """

    qdrant = QdrantClient(url="http://qdrant:6333")
    embedding = get_embedding(query)

    result = qdrant.query_points(
        collection_name="Amazon-items-collection-01-reviews",
        prefetch=[
                Prefetch(
                    query=embedding,
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="parent_asin",
                                match=MatchAny(any=item_list)
                            )
                        ]
                    ),
                    limit=20
            ),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []

    for point in result.points:
        retrieved_context_ids.append(point.payload["parent_asin"])
        retrieved_context.append(point.payload["text"])
        similarity_scores.append(point.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores
    }

@traceable(
    name="process_context",
    run_type="prompt"
)
def process_review_context(context):
    format_text = ""

    for id, context in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        format_text += f"- ID: {id}, description: {context}\n"
    return format_text

def get_formatted_review_context(query: str, item_list: list, top_k: int = 2) -> str:
    """
    Retrieve and format top k customer reviews matching a specific query for a list of product IDs (parent ASINs).

    Args:
        query: The search query to find relevant reviews (e.g. "battery life", "sound quality")
        item_list: List of product IDs (parent ASINs) to search reviews for. Example: ["B09ZPV8WBV"]
        top_k: Number of review text chunks to retrieve.

    Returns:
        A formatted string of the retrieved customer reviews containing the product IDs and review texts.
    """

    context = retrieve_review_data(query, item_list, top_k)
    formatted = process_review_context(context)

    return formatted