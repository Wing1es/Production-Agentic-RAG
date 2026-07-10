import psycopg2
import numpy as np
from psycopg2.extras import RealDictCursor
from fastembed import TextEmbedding
from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FormulaQuery, FusionQuery, SumExpression, MultExpression, Document
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

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
    if run_tree is not None:
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

# Shopping cart agent

@traceable(
    name="add_to_shopping_cart",
    run_type="tool"
)
def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> str:
    """
    Add a list of items to a user's designated shopping cart in the database.

    This function inserts items into the `shopping_carts.shopping_cart_items` table.
    If a product is already in the specified cart for the user, it should handle the 
    conflict by updating/incrementing the quantity (an "upsert") to respect the
    database's uniqueness constraint.

    Args:
        items (list[dict]): A list of dictionaries representing the items to add. 
            Each dictionary should contain:
                - 'product_id' (str): The ID of the product (required).
                - 'quantity' (int): The quantity to add (optional, defaults to 1).
                - 'price' (float/Decimal): The unit price of the item (optional).
                - 'currency' (str): The 3-character currency code (optional, e.g., 'USD').
                - 'product_image_url' (str): URL to the product image (optional).
        user_id (str): The unique identifier of the user.
        cart_id (str): The ID of the shopping cart (e.g., 'main').

    Returns:
        str: A status message or confirmation ID indicating a successful insertion.

    Raises:
        ValueError: If any required fields (like 'product_id') are missing in `items` 
                    or if check constraints (price >= 0, quantity > 0) are violated.
    """

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_db",
        user="langgraph_user",
        password="langgraph_password"
    )

    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        for item in items:
            product_id = item["product_id"]
            quantity = item["quantity"]
            
            qdrant = QdrantClient(url="http://qdrant:6333")
            records, _ = qdrant.scroll(
                collection_name="Amazon-collection-01-hybrid-search",
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchValue(value=product_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )

            if not records:
                continue

            payload = records[0].payload
            product_image_url = payload.get("image")
            price = payload.get("price")
            currency = "USD"

            check_query = """
                SELECT id, quantity, price
                FROM shopping_carts.shopping_cart_items
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """

            cursor.execute(check_query, (user_id, cart_id, product_id))
            existing_item = cursor.fetchone()

            if existing_item:
                new_quantity = existing_item["quantity"] + quantity

                update_query = """
                    UPDATE shopping_carts.shopping_cart_items
                    SET
                        quantity = %s,
                        price = %s,
                        currency = %s,
                        product_image_url = COALESCE(%s, product_image_url)
                    WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                    RETURNING id, quantity, price
                """

                cursor.execute(
                                update_query, 
                                (new_quantity, price, currency, product_image_url, user_id, cart_id, product_id)
                            )
            
            else:

                insert_query = """
                    INSERT INTO shopping_carts.shopping_cart_items (
                        user_id, shopping_cart_id, product_id,
                        price, quantity, currency, product_image_url
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, quantity, price
                """

                cursor.execute(insert_query, (user_id, cart_id, product_id, quantity, price, currency, product_image_url))

    return f"Added {items} to the shopping cart"

@traceable(
    name="get_shopping_cart",
    run_type="tool"
)
def get_shopping_cart(user_id: str, cart_id: str) -> list[dict]:

    """
    Retrieve the items in a user's shopping cart from the database.

    Args:
        user_id (str): The unique identifier of the user.
        cart_id (str): The ID of the shopping cart (e.g., 'main').

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents 
            an item in the shopping cart (including product_id, quantity, 
            price, currency, and product_image_url).
    """

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_db",
        user="langgraph_user",
        password="langgraph_password"
    )

    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        fetch_query = """
        SELECT
            product_id, price, quantity,
            currency, product_image_url,
            (price * quantity) as total_price
        FROM shopping_carts.shopping_cart_items
        WHERE user_id = %s AND shopping_cart_id = %s
        ORDER BY added_at DESC
        """

        cursor.execute(fetch_query, (user_id, cart_id))

        return [dict(row) for row in cursor.fetchall()]

@traceable(
    name="remove_from_cart",
    run_type="tool"
)
def remove_from_cart(product_id: str, user_id: str, cart_id: str) -> bool:
    """
    Remove a specific product from a user's designated shopping cart in the database.

    Args:
        product_id (str): The ID of the product to remove.
        user_id (str): The unique identifier of the user.
        cart_id (str): The ID of the shopping cart (e.g., 'main').

    Returns:
        bool: True/False indicating the product was successfully removed.
    """

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_db",
        user="langgraph_user",
        password="langgraph_password"
    )

    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        delete_query = """
            DELETE FROM shopping_carts.shopping_cart_items
            WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
        """

        cursor.execute(delete_query, (user_id, cart_id, product_id))
    
    return cursor.rowcount > 0