
from collections.abc import Collection, Sequence
from pathlib import PurePosixPath

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Only these document types are split into smaller chunks.
SPLITTABLE_DOC_TYPES = {"company", "contracts"}


def _build_chunk_id(document: Document) -> str:
    """Build a readable, stable ID for company and contract chunks."""
    metadata = document.metadata

    # Normalize Windows and Unix paths.
    source = metadata["source"].replace("\\", "/")
    source_path = PurePosixPath(source)

    # Remove the leading "data/" directory if present.
    if source_path.parts and source_path.parts[0] == "data":
        source_path = PurePosixPath(*source_path.parts[1:])

    # Remove the file extension.
    source_path = source_path.with_suffix("")

    parts = [source_path.as_posix()]

    # Include page number when available to avoid collisions across pages.
    if "page" in metadata:
        parts.append(f"page_{metadata['page']}")

    # Include chunk index when available.
    if "chunk_index" in metadata:
        parts.append(f"chunk_{metadata['chunk_index']}")

    return "/".join(parts)


def get_document_id(document: Document) -> str:
    """Return the readable ID used for evaluation and citations."""
    metadata = document.metadata

    document_id = metadata.get("chunk_id") or metadata.get("record_id")

    if not document_id:
        raise ValueError(
            f"Document is missing both 'chunk_id' and 'record_id': "
            f"{metadata.get('source', '<unknown source>')}"
        )

    return document_id


def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    split_doc_types: Collection[str] = SPLITTABLE_DOC_TYPES,
) -> list[Document]:
    """Split long text documents and preserve structured records.

    Company and contract documents are split into overlapping chunks.
    Product and employee records remain intact and retain their record_id.
    Readable chunk_id values are assigned only to company and contract chunks.
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

        if not document.page_content.strip():
            continue

        if doc_type not in split_doc_types:
            # Keep structured records intact; do not duplicate record_id.
            chunked_documents.append(document)
            continue

        chunks = splitter.split_documents([document])

        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_id"] = _build_chunk_id(chunk)

        chunked_documents.extend(chunks)

    # Ensure every document has an ID and IDs are unique across the corpus.
    seen_ids: set[str] = set()

    for document in chunked_documents:
        document_id = get_document_id(document)

        if document_id in seen_ids:
            raise ValueError(f"Duplicate document ID detected: {document_id}")

        seen_ids.add(document_id)

    return chunked_documents
