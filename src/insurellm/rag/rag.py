# ___________________________Embedding model configuration.___________________________
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


def get_embedding_model_name() -> str:
    load_dotenv()
    model_name = os.getenv("EMBEDDING_MODEL_NAME")

    if not model_name:
        raise ValueError("Set EMBEDDING_MODEL_NAME in .env")

    return model_name


def get_embeddings(model_name: str | None = None) -> SentenceTransformer:
    return SentenceTransformer(
        model_name or get_embedding_model_name(),
        trust_remote_code=True,
    )


def embed_documents(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    return model.encode(
        texts,
        task="retrieval",
        prompt_name="document",
        batch_size=16,
        normalize_embeddings=True,
    ).tolist()


def embed_query(model: SentenceTransformer, text: str) -> list[float]:
    return model.encode(
        text,
        task="retrieval",
        prompt_name="query",
        normalize_embeddings=True,
    ).tolist()
