"""SentenceTransformer helpers with a LangChain Embeddings adapter."""
import os
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


def get_embedding_model_name() -> str:
    load_dotenv()
    name = os.getenv("EMBEDDING_MODEL_NAME")
    if not name:
        raise ValueError("Set EMBEDDING_MODEL_NAME in .env")
    return name


def get_embeddings_model(model_name: str | None = None) -> SentenceTransformer:
    return SentenceTransformer(
        model_name or get_embedding_model_name(),
        trust_remote_code=True,
    )


def embed_documents(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    return model.encode(
        texts, task="retrieval", prompt_name="document",
        batch_size=16, normalize_embeddings=True,
    ).tolist()


def embed_query(model: SentenceTransformer, text: str) -> list[float]:
    return model.encode(
        text, task="retrieval", prompt_name="query",
        normalize_embeddings=True,
    ).tolist()


class SentenceTransformerEmbeddings(Embeddings):
    """Adapt the project's existing embedding functions for LangChain/Qdrant."""
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_embedding_model_name()
        self.model = get_embeddings_model(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_documents(self.model, texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_query(self.model, text)