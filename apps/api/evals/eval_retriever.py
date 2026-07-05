from api.agents.retrieval_generation import rag_pipeline
from api.core.config import config
import sys
from types import ModuleType

sys.modules['langchain_community.chat_models.vertexai'] = ModuleType('vertexai')
sys.modules['langchain_community.chat_models.vertexai'].ChatVertexAI = None

from langsmith import Client
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI

from fastembed import TextEmbedding

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, ResponseRelevancy

from dotenv import load_dotenv
import os

load_dotenv()


client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="gemini-2.5-pro",
        api_key=config.GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
)

model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

class FastEmbedLangchainWrapper:
    def __init__(self, fastembed_model):
        self.fastembed_model = fastembed_model
        # Ragas usage tracking expects a string model name here
        self.model = "sentence-transformers/all-MiniLM-L6-v2"
    def embed_query(self, text: str) -> list[float]:
        return list(self.fastembed_model.embed([text]))[0].tolist()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self.fastembed_model.embed(texts)]

ragas_embeddings = LangchainEmbeddingsWrapper(FastEmbedLangchainWrapper(model))


def get_run_input(run, key, default=None):
    if hasattr(run, "inputs") and run.inputs:
        return run.inputs.get(key, default)
    try:
        return run[key]
    except (TypeError, KeyError):
        return default

def get_run_output(run, key, default=None):
    if hasattr(run, "outputs") and run.outputs:
        return run.outputs.get(key, default)
    try:
        return run[key]
    except (TypeError, KeyError):
        return default

def get_example_output(example, key, default=None):
    if hasattr(example, "outputs") and example.outputs:
        return example.outputs.get(key, default)
    try:
        return example[key]
    except (TypeError, KeyError):
        return default

def get_example_input(example, key, default=None):
    if hasattr(example, "inputs") and example.inputs:
        return example.inputs.get(key, default)
    try:
        return example[key]
    except (TypeError, KeyError):
        return default


async def ragas_faithfulness(run, example):
    question = get_example_input(example, "question") or get_run_input(run, "question")
    answer = get_run_output(run, "answer")
    retrieved_context = get_run_output(run, "retrieved_context")

    if answer is None or retrieved_context is None:
        return None

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=retrieved_context
    )

    scorer = Faithfulness(llm=ragas_llm)

    return await scorer.single_turn_ascore(sample)

async def ragas_response_relevancy(run, example):
    question = get_example_input(example, "question") or get_run_input(run, "question")
    answer = get_run_output(run, "answer")
    retrieved_context = get_run_output(run, "retrieved_context")

    if answer is None or retrieved_context is None:
        return None

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=retrieved_context
    )

    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    return await scorer.single_turn_ascore(sample)

async def ragas_precision_id_based(run, example):
    retrieved_context_ids = get_run_output(run, "retrieved_context_ids")
    reference_context_ids = get_example_output(example, "reference_context_ids")

    if retrieved_context_ids is None or reference_context_ids is None:
        return None

    sample = SingleTurnSample(
        retrieved_context_ids=retrieved_context_ids,
        reference_context_ids=reference_context_ids
    )

    scorer = IDBasedContextPrecision()

    return await scorer.single_turn_ascore(sample)

async def ragas_recall_id_based(run, example):
    retrieved_context_ids = get_run_output(run, "retrieved_context_ids")
    reference_context_ids = get_example_output(example, "reference_context_ids")

    if retrieved_context_ids is None or reference_context_ids is None:
        return None

    sample = SingleTurnSample(
        retrieved_context_ids=retrieved_context_ids,
        reference_context_ids=reference_context_ids
    )

    scorer = IDBasedContextRecall()

    return await scorer.single_turn_ascore(sample)


results = client.evaluate(
    lambda x: rag_pipeline(x["question"]),
    data="rag_data_eval",
    evaluators=[
        ragas_precision_id_based,
        ragas_recall_id_based,
        ragas_faithfulness,
        ragas_response_relevancy
    ],
    experiment_prefix="retriever_eval",
    max_concurrency=1,
    metadata={"dataset_name": "rag_data_eval"}
)