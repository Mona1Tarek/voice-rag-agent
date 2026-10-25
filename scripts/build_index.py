
"""Build and persist the Insurellm Qdrant vector index."""

from insurellm.ingestion.chunker import chunk_documents
from insurellm.ingestion.loader import load_documents
from insurellm.rag.embeddings import SentenceTransformerEmbeddings, get_embedding_model_name
from insurellm.rag.vector_store import get_vector_store, index_documents


def main() -> None:
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded documents: {len(documents)}")

    print("\nChunking documents...")
    chunks = chunk_documents(documents)
    print(f"Generated chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks generated. Check the ingestion pipeline.")

    model_name = get_embedding_model_name()
    print(f"\nEmbedding model: {model_name}")

    embeddings = SentenceTransformerEmbeddings(model_name)
    vector_store = get_vector_store(embeddings, model_name)

    print("\nIndexing chunks in Qdrant...")
    index_documents(
        documents=chunks,
        vector_store=vector_store,
        embeddings=embeddings,
        model_name=model_name,
    )

    collection_name = vector_store.collection_name
    client = vector_store.client

    count = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    print("\nIndexing complete.")
    print(f"Collection: {collection_name}")
    print(f"Stored points: {count}")

    if count == 0:
        raise RuntimeError("Indexing finished, but Qdrant contains no points.")

    if count < len(chunks):
        print(
            f"Warning: stored points ({count}) are fewer than "
            f"generated chunks ({len(chunks)}). Check document IDs "
            "for duplicates."
        )


if __name__ == "__main__":
    main()
