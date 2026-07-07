from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery
from qdrant_client.models import Filter, FieldCondition, MatchAny

model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text, model=model):
    return list(model.embed([text]))[0]


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

def process_review_context(context):
    format_text = ""

    for id, context in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        format_text += f"- ID: {id}, description: {context}\n"
    return format_text