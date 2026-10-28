"""Retrieve relevant chunks from the configured Qdrant collection."""

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore


def retrieve_documents(
    question: str,
    vector_store: QdrantVectorStore,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """Return the top-k documents with their similarity scores."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Performs a search and returns each document along with its score.
    results = vector_store.similarity_search_with_score(
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
