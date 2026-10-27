from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from insurellm.llm.client import LLMClient
from insurellm.rag.pipeline import generate_answer
from evaluation.generation.evaluators import citation_metrics, citation_id_validity, abstention_heuristic, lexical_overlap
from evaluation.generation.metrics import summarize

from insurellm.rag.retrieval import get_vector_store
from insurellm.rag.embeddings import (
    SentenceTransformerEmbeddings,
    get_embedding_model_name,
)

HERE = Path(__file__).resolve().parent
def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=HERE/"generation_questions.jsonl")
    parser.add_argument("--output-dir", type=Path, default=HERE/"results")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1: parser.error("--top-k must be >= 1")
    questions = read_jsonl(args.dataset)
    if args.limit: questions = questions[:args.limit]
    if not questions: raise SystemExit(f"No questions in {args.dataset}")
    client = LLMClient()
    
    embeddings = SentenceTransformerEmbeddings(get_embedding_model_name())
    vector_store = get_vector_store(embeddings=embeddings)
    
    rows = []
    for i, q in enumerate(questions, 1):
        qid = q.get("query_id", f"row-{i}")
        print(f"[{i}/{len(questions)}] {qid}: {q['question']}")
        try:
            out = generate_answer(
                q["question"],
                llm_client=client,
                top_k=args.top_k,
                stream=not args.no_stream,
                vector_store=vector_store,
            )
            
            llm = out["llm"]
            retrieved_ids = [d["chunk_id"] for d in out["retrieved_documents"] if d.get("chunk_id")]
            generated = out["generated_citations"]
            diag = {}
            diag.update(citation_metrics(generated, q.get("reference_citations", [])))
            diag["citation_id_validity"] = citation_id_validity(generated, retrieved_ids)
            diag.update(abstention_heuristic(out["answer"], bool(q["is_answerable"])))
            diag["reference_lexical_overlap_diagnostic"] = (
                lexical_overlap(q.get("reference_answer", ""), out["answer"])
                if q["is_answerable"] else None)
            row = {
                "query_id": qid, "question": q["question"], "query_language": q.get("query_language"),
                "scenario": q.get("scenario"), "is_answerable": q.get("is_answerable"),
                "reference_answer": q.get("reference_answer"),
                "reference_citations": q.get("reference_citations", []),
                "answer": out["answer"], "generated_citations": generated,
                "retrieved_documents": out["retrieved_documents"], "retrieved_chunk_ids": retrieved_ids,
                "diagnostics": diag, "provider": llm.provider, "model": llm.model,
                "retrieval_latency_ms": out["retrieval_latency_ms"], "llm_latency_ms": llm.latency_ms,
                "end_to_end_latency_ms": out["end_to_end_latency_ms"], "ttft_ms": llm.ttft_ms,
                "generation_duration_ms": llm.generation_duration_ms,
                "input_tokens": llm.input_tokens, "output_tokens": llm.output_tokens,
                "total_tokens": llm.total_tokens, "output_tokens_per_second": llm.tokens_per_second,
                "finish_reason": llm.finish_reason, "http_status_code": llm.status_code,
                "rate_limit_headers": llm.rate_limit_headers, "error": llm.error,
                "human_review": {"correctness_0_to_2": None, "faithfulness_0_to_2": None,
                    "citation_semantic_support_0_to_2": None, "language_quality_0_to_2": None,
                    "relevance_0_to_2": None, "notes": None},
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat()
            }
        except Exception as exc:
            row = {"query_id": qid, "question": q.get("question"),
                   "provider": client.provider, "model": client.model,
                   "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        if row.get("error"): print("  ERROR:", row["error"])
        else: print("  Answer:", row["answer"][:150].replace("\n", " "))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"{client.provider}_{client.model.replace('/', '-')}_{stamp}"
    raw = args.output_dir/"raw"/f"generation_{slug}.jsonl"
    report_path = args.output_dir/"reports"/f"summary_{slug}.json"
    write_jsonl(raw, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = summarize(rows)
    report.update({"provider": client.provider, "model": client.model,
        "dataset": str(args.dataset), "top_k": args.top_k, "streaming": not args.no_stream,
        "raw_results_file": str(raw), "generated_at_utc": datetime.now(timezone.utc).isoformat()})
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRaw results: {raw}\nSummary: {report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
