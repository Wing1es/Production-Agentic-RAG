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

@traceable(
    name="check_warehouse_availability",
    run_type="tool"
)
def check_warehouse_availability(items: list[dict]) -> dict:
    """Check availability of items across warehouses, including partial fulfillment options.
    
    Args:
        items: A list of items to check. Each item is a dictionary with keys: product_id, quantity.

    Returns:
        A dictionary containing:
        - can_fulfill_completely: bool indicating if all items can be fulfilled from at least one warehouse
        - warehouses_full_fulfillment: list of warehouses that can fulfill the entire order
        - warehouses_partial_fulfillment: list of warehouses with partial availability
        - unavailable_items: list of items that cannot be fulfilled from any warehouse
        - details: detailed breakdown per warehouse with availability for each item
    """
    
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_db",
        user="langgraph_user",
        password="langgraph_password"
    )
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result = {
                "can_fulfill_completely": False,
                "warehouses_full_fulfillment": [],
                "warehouses_partial_fulfillment": [],
                "unavailable_items": [],
                "details": []
            }
            
            # Check each warehouse for availability
            warehouse_query = """
            SELECT DISTINCT warehouse_id, warehouse_name, warehouse_location
            FROM warehouses.inventory
            """
            
            cursor.execute(warehouse_query)
            warehouses = cursor.fetchall()
            
            for warehouse in warehouses:
                warehouse_can_fulfill_all = True
                has_any_availability = False
                warehouse_details = {
                    "warehouse_id": warehouse['warehouse_id'],
                    "warehouse_name": warehouse['warehouse_name'],
                    "warehouse_location": warehouse['warehouse_location'],
                    "items": [],
                    "can_fulfill_all": False,
                    "has_partial": False
                }
                
                for item in items:
                    product_id = item['product_id']
                    requested_quantity = item['quantity']
                    
                    # Check availability in this warehouse
                    availability_query = """
                    SELECT product_id, total_quantity, reserved_quantity, available_quantity
                    FROM warehouses.inventory
                    WHERE warehouse_id = %s AND product_id = %s
                    """
                    
                    cursor.execute(availability_query, (warehouse['warehouse_id'], product_id))
                    inventory = cursor.fetchone()
                    
                    available_qty = inventory['available_quantity'] if inventory else 0
                    
                    item_detail = {
                        "product_id": product_id,
                        "requested": requested_quantity,
                        "available": available_qty,
                        "can_fulfill_completely": available_qty >= requested_quantity,
                        "can_fulfill_partially": available_qty > 0 and available_qty < requested_quantity
                    }
                    
                    warehouse_details["items"].append(item_detail)
                    
                    # Track if warehouse can fulfill this item completely
                    if available_qty < requested_quantity:
                        warehouse_can_fulfill_all = False
                        
                    # Track if warehouse has any availability for any item
                    if available_qty > 0:
                        has_any_availability = True
                
                # Categorize warehouse
                if warehouse_can_fulfill_all:
                    warehouse_details["can_fulfill_all"] = True
                    result["warehouses_full_fulfillment"].append({
                        "warehouse_id": warehouse['warehouse_id'],
                        "warehouse_name": warehouse['warehouse_name'],
                        "warehouse_location": warehouse['warehouse_location']
                    })
                elif has_any_availability:
                    warehouse_details["has_partial"] = True
                    result["warehouses_partial_fulfillment"].append({
                        "warehouse_id": warehouse['warehouse_id'],
                        "warehouse_name": warehouse['warehouse_name'],
                        "warehouse_location": warehouse['warehouse_location']
                    })
                    
                result["details"].append(warehouse_details)
                
            # Check if any items cannot be fulfilled from any warehouse
            for item in items:
                product_id = item['product_id']
                requested_quantity = item['quantity']
                
                # Get total available quantity across all warehouses
                total_available_query = """
                SELECT product_id, SUM(available_quantity) as total_available
                FROM warehouses.inventory
                WHERE product_id = %s
                GROUP BY product_id
                """
                
                cursor.execute(total_available_query, (product_id,))
                total_available = cursor.fetchone()
                
                total_available_qty = total_available['total_available'] if total_available else 0
                
                if total_available_qty < requested_quantity:
                    result["unavailable_items"].append({
                        "product_id": product_id,
                        "requested": requested_quantity,
                        "total_available_across_warehouses": total_available_qty,
                        "shortage": requested_quantity - total_available_qty
                    })
                    
            result["can_fulfill_completely"] = len(result["warehouses_full_fulfillment"]) > 0 and len(result["unavailable_items"]) == 0
            
            return result
            
    finally:
        conn.close()

@traceable(
    name="reserve_warehouse_items",
    run_type="tool"
)
def reserve_warehouse_items(reservations: list[dict]) -> dict:
    
    """Reserve items from multiple warehouses in a single transaction.
    
    Args:
        reservations: A list of reservations. Each reservation is a dictionary with keys:
                     - warehouse_id - The warehouse to reserve from
                     - product_id - The product to reserve
                     - quantity - The quantity to reserve

    Returns:
        A dictionary containing:
        - success: bool indicating if all reservations were successful
        - reserved_items: list of successfully reserved items
        - failed_items: list of items that could not be reserved
    """
    
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_db",
        user="langgraph_user",
        password="langgraph_password"
    )
    conn.autocommit = False  # Use transaction
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result = {
                "success": False,
                "reserved_items": [],
                "failed_items": []
            }
            
            for reservation in reservations:
                warehouse_id = reservation['warehouse_id']
                product_id = reservation['product_id']
                quantity = reservation['quantity']
                
                # Check and lock the inventory row
                check_query = """
                    SELECT warehouse_id, product_id, warehouse_name, warehouse_location, 
                           total_quantity, reserved_quantity, available_quantity
                    FROM warehouses.inventory
                    WHERE warehouse_id = %s AND product_id = %s
                    FOR UPDATE
                """
                cursor.execute(check_query, (warehouse_id, product_id))
                inventory = cursor.fetchone()
                
                if inventory and inventory['available_quantity'] >= quantity:
                    # Update inventory to reserve the items
                    update_query = """
                        UPDATE warehouses.inventory
                        SET reserved_quantity = reserved_quantity + %s
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(update_query, (quantity, warehouse_id, product_id))
                    
                    result["reserved_items"].append({
                        "product_id": product_id,
                        "quantity": quantity,
                        "warehouse_id": warehouse_id,
                        "warehouse_name": inventory['warehouse_name'],
                        "warehouse_location": inventory['warehouse_location']
                    })
                else:
                    result["failed_items"].append({
                        "product_id": product_id,
                        "warehouse_id": warehouse_id,
                        "requested": quantity,
                        "available": inventory['available_quantity'] if inventory else 0,
                        "reason": "insufficient_stock" if inventory else "not_in_warehouse"
                    })
            
            # Only commit if all items were successfully reserved
            if len(result["failed_items"]) == 0:
                conn.commit()
                result["success"] = True
            else:
                conn.rollback()
                result["success"] = False
            
            return result
            
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()