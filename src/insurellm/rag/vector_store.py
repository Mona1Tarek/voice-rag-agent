"""Create and manage the Qdrant collection, insert vectors, and store metadata."""
import hashlib
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from insurellm.rag.embedding_text import build_embedding_text
from insurellm.rag.embeddings import SentenceTransformerEmbeddings, get_embedding_model_name


# To switch between different embedding models. Therefore we need different separated collections.
def _collection_name(model_name: str) -> str:
    base = os.getenv("QDRANT_COLLECTION_NAME", "insurellm")
    safe = "".join(c.lower() if c.isalnum() else "_" for c in base)
    suffix = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{suffix}"[:120]


# Qdrant requires an ID for every point you store. This point includes the vector and its associated payload.
# Instead of generating a random ID every time, we convert the existing ID into a consistent UUID using `uuid5`.
_NAMESPACE = uuid.UUID("c3452c87-b04c-4af0-a0a2-c67d7053e97a")

def _stable_id(document: Document) -> str:
    key = document.metadata.get("chunk_id") or document.metadata.get("record_id")
    if not key:
        raise ValueError("Each document needs chunk_id or record_id metadata.")
    return str(uuid.uuid5(_NAMESPACE, str(key)))


# Creating the vectore store collection.
def get_vector_store(
    embeddings: SentenceTransformerEmbeddings | None = None,
    model_name: str | None = None,
) -> QdrantVectorStore:
    
    load_dotenv()
    model_name = model_name or get_embedding_model_name()
    embeddings = embeddings or SentenceTransformerEmbeddings(model_name)
    
    path = Path(os.getenv("QDRANT_PATH", "./vectordb/qdrant"))
    path.mkdir(parents=True, exist_ok=True)
    
    client = QdrantClient(path=str(path))
    collection = _collection_name(model_name)

    if not client.collection_exists(collection):
        dimension = len(embeddings.embed_query("dimension check"))
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            ),
        )

    # This allows us to interact with Qdrant via LangChain, rather than having to write all the storage and search operations directly using the Qdrant client.
    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )


# Indexing documents into the vector store.
# Takes the documents after chunking and loads them into the vector store.
def index_documents(
    documents: list[Document],
    vector_store: QdrantVectorStore | None = None,
    embeddings: SentenceTransformerEmbeddings | None = None,
    model_name: str | None = None,
) -> QdrantVectorStore:

    if not documents:
        raise ValueError("No documents supplied for indexing.")
    
    # This gives flexibility to use the function on its own, or to reuse objects we have already prepared.
    model_name = model_name or get_embedding_model_name()
    embeddings = embeddings or SentenceTransformerEmbeddings(model_name)
    vector_store = vector_store or get_vector_store(embeddings, model_name)

    prepared, ids = [], []
    for doc in documents:
        metadata = dict(doc.metadata)       # A copy of the metadata to avoid mutating the original document's metadata.
        metadata["original_page_content"] = doc.page_content    # We store the original text within the metadata, while the `page_content` in the new document contains the text prepared for embedding.
        
        prepared.append(
            Document(
                page_content=build_embedding_text(doc),
                metadata=metadata,
            )
        )
        ids.append(_stable_id(doc))

    # Actual storage via LangChain
    # We use the `add_documents` method from LangChain's QdrantVectorStore, which handles the embedding and storage of the documents in one step. This is more efficient than manually embedding and adding each document.
    vector_store.add_documents(prepared, ids=ids)
    
    return vector_store
