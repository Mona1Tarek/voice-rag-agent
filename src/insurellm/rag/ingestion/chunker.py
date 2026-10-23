
from collections.abc import Collection, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Only these document types are split into smaller chunks.
SPLITTABLE_DOC_TYPES = {"company", "contract"}


def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    split_doc_types: Collection[str] = SPLITTABLE_DOC_TYPES,
) -> list[Document]:
    """Split long text documents while preserving metadata and structured records.

    Company and contract documents are split into overlapping chunks.
    Product and employee records remain intact as individual documents.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=SEPARATORS,
        keep_separator=True,
        add_start_index=True,
    )

    chunked_documents: list[Document] = []

    for document in documents:
        doc_type = document.metadata.get("doc_type")

        if doc_type not in split_doc_types:
            # Keep structured records intact.
            chunked_documents.append(document)
            continue

        if not document.page_content.strip():
            # Ignore empty text documents.
            continue

        chunks = splitter.split_documents([document])

        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index

        chunked_documents.extend(chunks)

    return chunked_documents
