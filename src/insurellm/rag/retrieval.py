
"""Retrieve relevant chunks from the configured Qdrant collection."""

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from insurellm.rag.embeddings import SentenceTransformerEmbeddings
from insurellm.rag.vector_store import get_vector_store


def retrieve_documents(
    question: str,
    top_k: int = 5,
    vector_store: QdrantVectorStore | None = None,
    embeddings: SentenceTransformerEmbeddings | None = None,
    model_name: str | None = None,
) -> list[tuple[Document, float]]:
    """Return the top-k documents with their similarity scores."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    store = vector_store or get_vector_store(
        embeddings=embeddings,
        model_name=model_name,
    )

    # Performs a search and returns each document along with its score.
    results = store.similarity_search_with_score(
        question,
        k=top_k,
    )

    retrieved_documents: list[tuple[Document, float]] = []

    # To pass the original text to the LLM in addition to it's citaion in the source document.
    for document, score in results:
        original_text = document.metadata.pop(
            "original_page_content",
            None,
        )

        if original_text is not None:
            document.page_content = original_text

        retrieved_documents.append((document, float(score)))

    return retrieved_documents
