from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FormulaQuery, FusionQuery, SumExpression, MultExpression, Document

model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text, model=model):
    return list(model.embed([text]))[0]

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

def process_context(context):
    format_text = ""

    for id, rating, context in zip(context["retrieved_context_ids"], context["retrieved_context_ratings"], context["retrieved_context"]):
        format_text += f"- ID: {id}, rating: {rating}, description: {context}\n"
    return format_text