"""RAG orchestration entry point; generation can be added after retrieval is tested."""
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from insurellm.rag.embeddings import SentenceTransformerEmbeddings
from insurellm.rag.retrieval import retrieve_documents


def retrieve_context(
    question: str,
    top_k: int = 5,
    vector_store: QdrantVectorStore | None = None,
    embeddings: SentenceTransformerEmbeddings | None = None,
    model_name: str | None = None,
) -> list[tuple[Document, float]]:
    """Return retrieved context and scores; this function does not call an LLM yet."""
    return retrieve_documents(question, top_k, vector_store, embeddings, model_name)
