from __future__ import annotations
import re, time
from typing import Any

from langchain_qdrant import QdrantVectorStore
from insurellm.llm.client import LLMClient
from insurellm.llm.prompt import get_prompt
from insurellm.observability.tracing import observation
from insurellm.rag.embeddings import get_embedding_model_name
from insurellm.rag.retrieval import retrieve_documents

def build_messages(question: str, retrieved: list[tuple[Any, float]]) -> list[dict[str, str]]:
    parts = []
    for i, (doc, score) in enumerate(retrieved, 1):
        md = doc.metadata or {}
        cid = md.get("chunk_id") or md.get("record_id") or md.get("source") or f"chunk-{i}"
        parts.append(f"[CHUNK_ID: {cid}]\n[SOURCE: {md.get('source', 'unknown')}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(parts) if parts else "(No documents retrieved.)"
    return [
        {"role": "system", "content": get_prompt()},
        {"role": "user", "content": f"Question:\n{question}\n\nRetrieved context:\n{context}"},
    ]

def generate_answer(
    question: str,
    *,
    llm_client: LLMClient,
    top_k: int = 5,
    stream: bool = True,
    vector_store: QdrantVectorStore,
) -> dict[str, Any]:
    start = time.perf_counter()

    with observation(
        as_type="retriever",
        name="qdrant-retrieval",
        input={"question": question, "top_k": top_k},
    ) as span:
        retrieved = retrieve_documents(
            question,
            top_k=top_k,
            vector_store=vector_store,
        )

        span.update(
            output={
                "retrieved_count": len(retrieved),
                "results": [
                    {
                        "chunk_id": (
                            doc.metadata.get("chunk_id")
                            or doc.metadata.get("record_id")
                        ),
                        "source": doc.metadata.get("source"),
                        "score": float(score),
                        "text_preview": doc.page_content[:200],
                    }
                    for doc, score in retrieved
                ],
            },
            metadata={"embedding_model": get_embedding_model_name()},
        )

    retrieval_ms = (time.perf_counter() - start) * 1000
    messages = build_messages(question, retrieved)
    result = llm_client.generate(
        messages,
        stream=stream,
    )

    docs = []
    for doc, score in retrieved:
        md = doc.metadata or {}
        docs.append({
            "chunk_id": md.get("chunk_id") or md.get("record_id"),
            "source": md.get("source"),
            "doc_type": md.get("doc_type"),
            "language": md.get("language"),
            "score": float(score),
            "text": doc.page_content,
        })

    citations = list(dict.fromkeys(
        re.findall(r"\[([^\[\]]+)\]", result.text)
    ))

    return {
        "answer": result.text,
        "generated_citations": citations,
        "retrieved_documents": docs,
        "retrieval_latency_ms": retrieval_ms,
        "llm": result,
        "end_to_end_latency_ms": retrieval_ms + result.latency_ms,
    }
