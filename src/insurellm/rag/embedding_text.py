"""Prepare the text that gets embedded."""
from pathlib import Path
from langchain_core.documents import Document


def build_embedding_text(document: Document) -> str:
    if document.metadata.get("doc_type") == "contracts":
        title = Path(document.metadata["source"]).stem
        return f"Contract: {title}\n\n{document.page_content}"

    return document.page_content
