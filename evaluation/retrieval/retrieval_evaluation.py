
"""Evaluate embedding-based retrieval."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ranx import Qrels, Run, evaluate
from sentence_transformers.util import cos_sim

from insurellm.ingestion.chunker import chunk_documents, get_document_id
from insurellm.ingestion.loader import load_documents
from insurellm.rag.embedding_text import build_embedding_text
from insurellm.rag.embeddings import (
    embed_documents,
    embed_query,
    get_embedding_model_name,
    get_embeddings_model,
)

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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.top_k < 5:
        parser.error("--top-k must be at least 5.")

    questions = load_questions()
    documents = chunk_documents(load_documents())
    document_ids = [get_document_id(doc) for doc in documents]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate document IDs found in the corpus.")

    model_name = args.model or get_embedding_model_name()
    model = get_embeddings_model(model_name)

    # Embed each document once.
    # document_vectors = embed_documents(model, [doc.page_content for doc in documents])
    document_vectors = embed_documents(model, [build_embedding_text(doc) for doc in documents])

    qrels_data = {}
    run_data = {}
    details = []

    for question in questions:
        query_id = question["query_id"]
        query_vector = embed_query(model, question["question"])

        # Rank documents by cosine similarity.
        scores = cos_sim(query_vector, document_vectors)[0].tolist()
        ranked = sorted(
            zip(document_ids, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[:args.top_k]

        qrels_data[query_id] = {
            doc_id: 1 for doc_id in question["relevant_chunk_ids"]
        }
        run_data[query_id] = {
            doc_id: float(score) for doc_id, score in ranked
        }

        details.append({
            **question,
            "retrieved_chunk_ids": [doc_id for doc_id, _ in ranked],
            "similarity_scores": [float(score) for _, score in ranked],
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR / "retrieval_results.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for result in details:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Group queries by language and target document category.
    language_groups = {
        language: [
            q["query_id"]
            for q in questions
            if q["query_language"] == language
        ]
        for language in ("ar", "en")
    }

    def get_document_category(question):
        relevant_ids = question["relevant_chunk_ids"]

        # Use the ground-truth IDs to identify the target category.
        if any(doc_id.startswith("company/") for doc_id in relevant_ids):
            return "Company (Arabic document)"
        if any(doc_id.startswith("employees/") for doc_id in relevant_ids):
            return "Employees"
        if any(doc_id.startswith("products/") for doc_id in relevant_ids):
            return "Products"
        if any(doc_id.startswith("contracts/") for doc_id in relevant_ids):
            return "Contracts"

        return "Other"

    def metrics_for(query_ids):
        return calculate_metrics(
            Qrels({qid: qrels_data[qid] for qid in query_ids}),
            Run({qid: run_data[qid] for qid in query_ids}),
        )

    lines = [
        "Retrieval Evaluation",
        "--------------------",
        f"Total Questions: {len(questions)}",
        f"Embedding Model: {model_name}",
        f"Top K: {args.top_k}",
        "",
        "Overall",
        *format_metrics(metrics_for(
            [q["query_id"] for q in questions]
        )),
        "",
    ]

    # Language-level results.
    for language in ("ar", "en"):
        lines.extend([
            f"Language: {language}",
            *format_metrics(metrics_for(language_groups[language])),
            "",
        ])

    # Detailed results, separated by query language.
    for language, section_title in (
        ("ar", "Arabic Query Evaluation"),
        ("en", "English Query Evaluation"),
    ):
        lines.extend([section_title, "=" * len(section_title), ""])

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
            if not query_ids:
                continue

            lines.extend([
                category,
                *format_metrics(metrics_for(query_ids)),
                "",
            ])

    report = "\n".join(lines)
    report_path = RESULTS_DIR / "retrieval_evaluation.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Detailed results: {RESULTS_DIR / 'retrieval_results.jsonl'}")
    print(f"Report: {report_path}")



if __name__ == "__main__":
    main()
