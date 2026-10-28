
"""Evaluate retrieval from the persisted Qdrant vector store, including latency."""

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from ranx import Qrels, Run, evaluate

from insurellm.rag.embeddings import SentenceTransformerEmbeddings, get_embedding_model_name
from insurellm.rag.retrieval import retrieve_documents
from insurellm.rag.vector_store import get_vector_store

EVALUATION_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVALUATION_DIR / "questions.jsonl"
RESULTS_DIR = EVALUATION_DIR / "results"

METRICS = ["hit_rate@1", "hit_rate@3", "hit_rate@5", "mrr@5"]


def load_questions():
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def calculate_metrics(qrels, run):
    return evaluate(qrels, run, METRICS)


def format_metrics(metrics):
    return [
        f"Hit Rate@1: {metrics['hit_rate@1']:.2%}",
        f"Hit Rate@3: {metrics['hit_rate@3']:.2%}",
        f"Hit Rate@5: {metrics['hit_rate@5']:.2%}",
        f"MRR@5: {metrics['mrr@5']:.4f}",
    ]


def get_document_category(question):
    relevant_ids = question["relevant_chunk_ids"]

    if any(doc_id.startswith("company/") for doc_id in relevant_ids):
        return "Company (Arabic document)"
    if any(doc_id.startswith("employees/") for doc_id in relevant_ids):
        return "Employees"
    if any(doc_id.startswith("products/") for doc_id in relevant_ids):
        return "Products"
    if any(doc_id.startswith("contracts/") for doc_id in relevant_ids):
        return "Contracts"

    return "Other"


def percentile(values, percent):
    """Calculate a percentile using the nearest-rank method."""
    ordered = sorted(values)
    index = max(0, math.ceil(percent / 100 * len(ordered)) - 1)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.top_k < 5:
        parser.error("--top-k must be at least 5.")

    questions = load_questions()
    if not questions:
        raise ValueError(f"No questions found in {QUESTIONS_PATH}")

    model_name = args.model or get_embedding_model_name()

    # Initialize the embedding model and persisted Qdrant store once.
    embeddings = SentenceTransformerEmbeddings(model_name)
    vector_store = get_vector_store(embeddings, model_name)

    qrels_data = {}
    run_data = {}
    details = []
    latencies_ms = []

    for question in questions:
        query_id = question["query_id"]

        # Measure retrieval latency only, excluding model/store initialization.
        start = time.perf_counter()
        results = retrieve_documents(
            question=question["question"],
            top_k=args.top_k,
            vector_store=vector_store,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency_ms)

        retrieved_ids = []
        scores = []

        for document, score in results:
            document_id = (
                document.metadata.get("chunk_id")
                or document.metadata.get("record_id")
            )
            if document_id is None:
                raise ValueError(
                    "Retrieved document is missing chunk_id and record_id."
                )

            retrieved_ids.append(str(document_id))
            scores.append(float(score))

        qrels_data[query_id] = {
            doc_id: 1 for doc_id in question["relevant_chunk_ids"]
        }
        run_data[query_id] = {
            doc_id: score
            for doc_id, score in zip(retrieved_ids, scores, strict=True)
        }

        details.append({
            **question,
            "retrieved_chunk_ids": retrieved_ids,
            "similarity_scores": scores,
            "latency_ms": round(latency_ms, 3),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULTS_DIR / "qdrant_retrieval_results.jsonl"
    with results_path.open("w", encoding="utf-8") as file:
        for result in details:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    def metrics_for(query_ids):
        return calculate_metrics(
            Qrels({qid: qrels_data[qid] for qid in query_ids}),
            Run({qid: run_data[qid] for qid in query_ids}),
        )

    all_query_ids = [q["query_id"] for q in questions]
    lines = [
        "Qdrant Retrieval Evaluation",
        "===========================",
        f"Total Questions: {len(questions)}",
        f"Embedding Model: {model_name}",
        f"Top K: {args.top_k}",
        "",
        "Overall Retrieval Metrics",
        *format_metrics(metrics_for(all_query_ids)),
        "",
        "Retrieval Latency (ms)",
        f"Mean: {statistics.mean(latencies_ms):.2f}",
        f"Median: {statistics.median(latencies_ms):.2f}",
        f"P95: {percentile(latencies_ms, 95):.2f}",
        f"Min: {min(latencies_ms):.2f}",
        f"Max: {max(latencies_ms):.2f}",
        "",
    ]

    # Metrics grouped by query language.
    for language in ("ar", "en"):
        query_ids = [
            q["query_id"]
            for q in questions
            if q["query_language"] == language
        ]
        if query_ids:
            lines.extend([
                f"Language: {language}",
                *format_metrics(metrics_for(query_ids)),
                "",
            ])

    # Metrics grouped by language and target document category.
    for language, title in (
        ("ar", "Arabic Query Evaluation"),
        ("en", "English Query Evaluation"),
    ):
        lines.extend([title, "=" * len(title), ""])
        categories = defaultdict(list)

        for question in questions:
            if question["query_language"] == language:
                categories[get_document_category(question)].append(
                    question["query_id"]
                )

        for category in (
            "Company (Arabic document)",
            "Employees",
            "Products",
            "Contracts",
            "Other",
        ):
            query_ids = categories.get(category, [])
            if query_ids:
                lines.extend([
                    category,
                    *format_metrics(metrics_for(query_ids)),
                    "",
                ])

    report = "\n".join(lines)
    report_path = RESULTS_DIR / "qdrant_retrieval_evaluation.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nDetailed results: {results_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
