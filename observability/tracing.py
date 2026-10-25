
from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client
from insurellm.rag.retrieval import retrieve_documents


def traced_retrieval(question: str, top_k: int = 5):
    langfuse = get_client()

    with langfuse.start_as_current_observation(
        as_type="retriever",
        name="qdrant-retrieval",
        input={"question": question, "top_k": top_k},
    ) as span:
        results = retrieve_documents(question, top_k=top_k)

        span.update(
            output={
                "retrieved_count": len(results),
                "results": [
                    {
                        "document_id": (
                            doc.metadata.get("chunk_id")
                            or doc.metadata.get("record_id")
                        ),
                        "source": doc.metadata.get("source"),
                        "score": float(score),
                        "text_preview": doc.page_content[:200],
                    }
                    for doc, score in results
                ],
            }
        )

        return results
