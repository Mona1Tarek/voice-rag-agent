"""Split loaded documents into retrieval-friendly chunks."""

from langchain_core.documents import Document

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

SEPARATORS = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ")


def _validate_options(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _find_split(text: str, start: int, end: int, minimum: int) -> int:
    """Find the last natural boundary in the second half of a window."""
    window = text[start:end]

    for separator in SEPARATORS:
        index = window.rfind(separator)

        if index >= minimum:
            return start + index + len(separator)

    return end


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text on natural boundaries while respecting the size limit."""
    _validate_options(chunk_size, chunk_overlap)

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            end = _find_split(text, start, end, chunk_size // 2)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split a document and annotate each chunk with its position."""
    parts = split_text(document.page_content, chunk_size, chunk_overlap)
    chunk_count = len(parts)
    chunks = []

    for index, part in enumerate(parts):
        metadata = dict(document.metadata)
        metadata["chunk_index"] = index
        metadata["chunk_count"] = chunk_count

        base_id = metadata.get("record_id") or metadata.get("source", "document")
        page = metadata.get("page")

        if page is not None and not metadata.get("record_id"):
            base_id = f"{base_id}:page_{page}"

        metadata["chunk_id"] = f"{base_id}:chunk_{index}"

        chunks.append(Document(page_content=part, metadata=metadata))

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split every document in a collection into chunks."""
    chunks = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks
