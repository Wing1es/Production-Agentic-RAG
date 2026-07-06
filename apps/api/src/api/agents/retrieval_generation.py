from pathlib import Path
import groq
import instructor
import numpy as np
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, MatchValue, Filter
from qdrant_client.models import FusionQuery, Prefetch, Document
from qdrant_client.models import SumExpression, MultExpression, FormulaQuery
from fastembed import TextEmbedding
from langsmith import traceable, get_current_run_tree
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from api.core.config import config
from api.agents.models import RAGUsedContext, RAGResponse
from api.agents.utils.prompt_management import prompt_template_registry, prompt_template_config

PROMPTS_DIR = Path(__file__).parent / "prompts"

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
    name="retrieve_context",
    run_type="retriever"
)
def retrieve_data(query, qdrant, k=5, query_type="fusion"):
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

@traceable(
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(preprocessed_text, question):

    template = prompt_template_config(PROMPTS_DIR / "retrieval_generation.yaml", "retrieval_generation")

    prompt = template.render(
        preprocessed_text=preprocessed_text,
        question=question
    )
    
    return prompt

@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={
        "ls_provider": "groq",
        "ls_model_name": "meta-llama/llama-4-scout-17b-16e-instruct"
        }
)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(min=1, max=10),
    retry=retry_if_exception_type(groq.RateLimitError),
    reraise=True
)
def generate_answer(prompt):

    client = instructor.from_groq(groq.Groq(api_key=config.GROQ_API_KEY))

    response, raw_response = client.chat.completions.create_with_completion(
        messages = [
            {"role": "system", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        response_model=RAGResponse
    )

    run_tree = get_current_run_tree()
    run_tree.metadata["usage_metadata"] = {
        "input_tokens": raw_response.usage.prompt_tokens,
        "output_tokens": raw_response.usage.completion_tokens,
        "total_tokens": raw_response.usage.total_tokens
    }

    return response

@traceable(name="rag_pipeline")
def rag_pipeline(question, qdrant, top_k=5):

    retrieved = retrieve_data(question, qdrant, top_k)
    formatted_text = process_context(retrieved)
    prompt = build_prompt(formatted_text, question)
    response = generate_answer(prompt)

    final_result = {
        "question": question,
        "answer": response.answer,
        "references": response.references,
        "retrieved_context_ids": retrieved["retrieved_context_ids"],
        "retrieved_context_ratings": retrieved["retrieved_context_ratings"],
        "retrieved_context": retrieved["retrieved_context"],
        "similarity_scores": retrieved["similarity_scores"]
    }

    return final_result

def rag_pipeline_wrapper(question, top_k=5):
    qdrant = QdrantClient(url=config.QDRANT_URL)

    result = rag_pipeline(question, qdrant, top_k)

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
        "answer": result["answer"],
        "used_context": used_context
    }