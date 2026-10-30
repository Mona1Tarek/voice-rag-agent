# Voice RAG Agent

A bilingual Arabic–English voice assistant that answers questions about Insurellm using a document-grounded RAG pipeline. The system combines multi-format document ingestion, semantic retrieval, grounded answer generation, real-time voice interaction, and end-to-end observability.

## 1. Project Overview

The assistant allows users to ask questions about Insurellm in Arabic or English and receive spoken answers in the same language.

It uses a shared knowledge base across both languages: users can ask in English about information documented in Arabic, or ask in Arabic about information documented in English.

The system is designed around four main capabilities:

- **Multi-format ingestion:** Extracting and processing information from PDFs, Markdown documents, and CSV files, including tabular information.
- **Grounded RAG:** Retrieving relevant document chunks and generating answers supported by the retrieved evidence.
- **Voice interaction:** Handling Arabic and English speech through LiveKit, with support for interruptions and conversational context.
- **Observability and evaluation:** Tracing the pipeline through Langfuse and evaluating retrieval, generation, citations, and latency.

## 2. Knowledge Base

The project uses the Insurellm fictional insurance-company knowledge base, based on the dataset from [ed-donner/llm_engineering](https://github.com/ed-donner/llm_engineering/tree/main/week5/knowledge-base).

The knowledge base includes:

| Category | Format | Content |
|---|---|---|
| Company | Arabic PDF | Company overview, culture, careers, and background |
| Contracts | Markdown | Insurance contracts and their terms |
| Employees | PDF | Employee information presented in tabular format |
| Products | CSV | Structured product information and features |

The ingestion pipeline processes these formats and converts their content into searchable document chunks while preserving relevant metadata.

Employee PDFs contain tabular information. The ingestion process must preserve the relationships between table fields so that questions about employee records can be answered from the extracted content.

## 3. Architecture

```text
                    User
                     |
             Arabic / English
                     |
              LiveKit Agent
                     |
          +----------+----------+
          |          |          |
         STT        VAD         Turn
          |                    Detection
          +----------+----------+
                     |
              Voice Agent LLM
                     |
                RAG Tool
                     |
              RAG Pipeline
                     |
          +----------+----------+
          |          |          |
       Embedding  Qdrant     Retrieval
          Model   Vector       |
                   Store       |
                              Top-K
                                |
                         Context Assembly
                                |
                         RAG LLM (Qwen)
                                |
                     Answer + Citation IDs
                                |
                         Voice Agent
                                |
                              TTS
                                |
                         Spoken Response

                 Langfuse Observability
                 Traces and Measurements
```

### Main components

- **Ingestion:** Loads source documents, extracts their content, creates chunks, and attaches metadata.
- **Embedding model:** Converts document chunks and user questions into vector representations.
- **Qdrant:** Stores document vectors and metadata in a persistent local collection.
- **Retrieval:** Finds the most relevant chunks for a user query.
- **RAG generation:** Uses the retrieved context to generate a grounded answer and its supporting citation IDs.
- **LiveKit:** Orchestrates the real-time voice interaction, including speech recognition, turn handling, LLM responses, and speech synthesis.
- **Langfuse:** Records traces and observations across the voice and RAG pipeline.

## 4. Technology Stack and Model Selection

Each pipeline stage has different requirements. Models and configurations were selected through experimentation, considering language support, response quality, latency, and suitability for the stage.

### 4.1 Speech-to-Text

**Selected model:** `munsit-en-ar`

Arabic transcription quality was a key consideration because the assistant must understand both Arabic and English, including mixed-language speech.

Munsit was compared with `google/gemini-3.5-transcribe-live`. In the project's tests, Munsit produced fewer Arabic transcription errors while maintaining responsive transcription, making it the preferred option.

References:
- [Munsit Speech-to-Text](https://docs.munsit.com/speech-to-text/get-started)
- [Munsit Streaming](https://docs.munsit.com/speech-to-text/streaming)

### 4.2 Voice Agent LLM

**Selected model:** `google/gemma-4-31b-it`

This model is responsible for conversational orchestration and tool interaction in the LiveKit agent.

The selection was informed by LiveKit's published inference benchmark. In its optimized serving setup, Gemma 4 31B achieved approximately **192 ms time to first token**, while models including Gemini 2.5 Flash, GPT-5.5, GPT-4.1, and Gemini 3 Flash were reported at approximately 900–1,100 ms.

These are external benchmark results, not latency measurements from this project. Actual performance depends on the serving configuration and runtime conditions.

LiveKit's serving setup also provides automatic prompt caching for repeated prompt content. Cache usage can be inspected through Langfuse traces, where cached input tokens are visible.

Benchmark reference: [LiveKit — Latency-Optimized Inference with Gemma 4](https://livekit.com/blog/latency-optimized-inference-gemma-4-on-livekit).

### 4.3 RAG LLM

**Selected model:** `qwen/qwen3.8-27b` via Groq

This model generates answers from the retrieved document context and returns citation identifiers separately from the answer text.

It was compared with `openai/gpt-oss-20b`, also tested through Groq. In the project's experiments, Qwen provided a better balance of answer quality and observed latency, so it was selected for the RAG generation stage.

This comparison reflects the project's tested setup rather than a general benchmark of the two models.

### 4.4 Text-to-Speech

**Selected model:** `xai/tts-1`, voice `eve`

TTS selection focused on naturalness, speech quality, and the Arabic user experience.

After listening to and comparing available options, the xAI voice was preferred because it sounded smoother and more natural, particularly for Arabic responses. Although other options could be faster, the overall listening experience was the deciding factor.

### 4.5 Embedding Model

**Selected model:** `jinaai/jina-embeddings-v5-text-nano`

The embedding model was selected as a trade-off between retrieval quality and latency.

In the project's comparison, `BAAI/bge-m3` achieved better retrieval accuracy but introduced substantially higher latency. Jina's `jina-embeddings-v5-text-nano` offered a more suitable latency/quality balance for this voice application.

Model reference: [Jina Embeddings v5 Text Nano](https://jina.ai/en-US/models/jina-embeddings-v5-text-nano/).

### 4.6 Vector Store and Supporting Technologies

- **Qdrant:** Persistent local vector storage and similarity search.
- **LiveKit Agents:** Real-time voice interaction and orchestration.
- **Langfuse:** Tracing and inspection of voice and RAG pipeline execution.
- **uv:** Python dependency and project environment management.

## 5. Latency Optimization

Latency is especially important in voice applications because delays between the user's question and the assistant's response directly affect conversational flow.

Several optimizations were applied during development.

### Reusing initialized components

Initially, components such as the vector store, embedding model, and LLM client were being initialized during individual calls. Repeated initialization introduced unnecessary overhead.

The implementation was adjusted to initialize these components once and reuse them across requests, avoiding repeated setup work in the request path.

### VAD and turn handling

Voice Activity Detection (VAD) and turn-handling configuration were tuned to improve the detection of user speech boundaries and reduce unnecessary waiting before processing a turn.

### Reducing minimum response delay

The configured `min_delay` was reduced from **0.7 seconds to 0.3 seconds**, allowing the agent to proceed sooner rather than waiting for the previous delay.

### Model selection through experiments

Different models were tested for individual pipeline stages. The final choices reflect observed trade-offs between response quality, language performance, and latency rather than relying only on model specifications.

These changes improved the system's responsiveness, although further latency optimization remains possible.

## 6. Retrieval and Grounded Generation

### Retrieval

For each user question, the system:

1. Generates a query embedding.
2. Searches the Qdrant collection for semantically relevant chunks.
3. Retrieves the top five results by default.
4. Passes the retrieved context to the generation stage.

Document metadata is preserved to support identification of the source chunks and citation generation.

### Grounded generation

The RAG LLM receives the user's question and retrieved context. It is instructed to:

- Answer using the available document evidence.
- Respond in the user's language.
- Avoid inventing unsupported details.
- Abstain or clearly state when the documents do not contain the requested information.
- Keep the answer text separate from its citation identifiers.

Citation IDs are returned separately from the user-facing answer, so they can be used for evidence inspection without being read aloud by the voice agent.

## 7. Evaluation

Evaluation is divided into retrieval and generation to assess different parts of the RAG pipeline.

### 7.1 Evaluation Dataset and Language Coverage

The evaluation questions cover four query/document language combinations:

- **English → Arabic (EN–AR):** English questions whose relevant information is documented in Arabic.
- **Arabic → English (AR–EN):** Arabic questions whose relevant information is documented in English.
- **English → English (EN–EN):** English questions answered using English documents.
- **Arabic → Arabic (AR–AR):** Arabic questions answered using Arabic documents.

This tests both same-language retrieval and cross-language retrieval, which is particularly important for a bilingual knowledge base.

The retrieval evaluation uses **68 answerable questions**. The generation evaluation dataset extends the original set with **10 additional unanswerable questions**, giving 78 questions in total.

These additional questions deliberately ask for information that does not exist in the knowledge base. They are used to evaluate whether the model can recognize missing evidence and abstain instead of hallucinating an answer.

### 7.2 Retrieval Evaluation

The Qdrant retrieval evaluation uses **68 answerable questions**, with `top_k = 5`.

#### Metrics

- **Hit Rate@1, @3, and @5:** Measures whether at least one relevant chunk appears within the first K retrieved results.
- **MRR@5:** Measures the reciprocal rank of the first relevant result within the top five results, averaged across questions.

#### Overall results

| Metric | Result |
|---|---:|
| Hit Rate@1 | 67.65% |
| Hit Rate@3 | 80.88% |
| Hit Rate@5 | 89.71% |
| MRR@5 | 0.7593 |

#### Retrieval latency

| Metric | Result |
|---|---:|
| Mean | 173.31 ms |
| Median | 164.12 ms |
| P95 | 224.22 ms |

#### Results by query language

| Language | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Arabic | 63.64% | 75.76% | 81.82% | 0.7106 |
| English | 71.43% | 85.71% | 97.14% | 0.8052 |

#### Results by category

| Query language | Category | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | MRR@5 |
|---|---|---:|---:|---:|---:|
| Arabic | Company | 50.00% | 64.29% | 71.43% | 0.5857 |
| Arabic | Employees | 100.00% | 100.00% | 100.00% | 1.0000 |
| Arabic | Products | 20.00% | 40.00% | 60.00% | 0.3500 |
| Arabic | Contracts | 90.00% | 100.00% | 100.00% | 0.9500 |
| English | Company | 66.67% | 73.33% | 93.33% | 0.7433 |
| English | Employees | 100.00% | 100.00% | 100.00% | 1.0000 |
| English | Products | 75.00% | 100.00% | 100.00% | 0.8333 |
| English | Contracts | 60.00% | 90.00% | 100.00% | 0.7700 |

The results show stronger retrieval performance for English queries overall. Arabic product queries are the weakest category in this evaluation, with a 60% Hit Rate@5, making them a useful target for future retrieval improvements.

### 7.3 Generation Evaluation

The generation evaluation dataset contains **78 questions**: the 68 answerable questions used in retrieval evaluation, plus 10 additional unanswerable questions designed to test the system's ability to abstain when the knowledge base does not contain the requested information.

Due to Groq rate limits and the size of the evaluation inputs and outputs, running the entire dataset in a single execution led to timeouts and crashes.

To make evaluation more reliable, the dataset was processed in batches of **four questions at a time**. The metrics below represent one successful four-question batch, not the complete 78-question dataset.

#### Evaluation metrics

The generation evaluation considers several dimensions:

1. **Citation precision, recall, and F1:** Compare generated citation IDs against the reference citation IDs to assess citation coverage and relevance.
2. **Citation ID validity:** Measures whether generated citation IDs exist among the documents retrieved for the question.
3. **Abstention heuristic match rate:** Checks whether the system abstains consistently with the answerability labels using predefined English and Arabic abstention phrases.
4. **Reference lexical overlap:** A diagnostic comparison between generated answers and reference-answer tokens. It is not treated as a correctness metric.
5. **Operational metrics:** Request success rate, end-to-end latency, retrieval latency, LLM latency, time to first token, token counts, and output token throughput.

Correctness, faithfulness, semantic citation support, and language quality require human review or a separately validated evaluation judge.

#### Request and latency results

| Metric | Result |
|---|---:|
| Questions in this batch | 4 |
| Successful requests | 4 |
| Request success rate | 100% |
| Errors | 0 |
| Mean end-to-end latency | 1,594.22 ms |
| Median end-to-end latency | 1,175.32 ms |
| P95 end-to-end latency | 3,325.71 ms |
| Mean retrieval latency | 310.89 ms |
| Mean LLM latency | 1,283.32 ms |
| Mean time to first token (TTFT) | 1,279.96 ms |

#### Citation results

| Metric | Result |
|---|---:|
| Citation precision | 1.00 |
| Citation recall | 0.875 |
| Citation F1 | 0.917 |
| Citation ID validity | 1.00 |
| Abstention heuristic match rate | 1.00 |

The citation metrics indicate that the generated citation IDs were valid in this batch, with strong precision and recall. However, the sample is small and should be interpreted as preliminary rather than conclusive evidence of overall generation quality.

## 8. Observability with Langfuse

Langfuse is used to inspect the execution of the voice and RAG pipeline.

The traces help investigate:

- Speech-to-text processing.
- Voice-agent LLM calls and tool interactions.
- RAG retrieval and retrieved context.
- RAG LLM generation.
- Text-to-speech processing.
- Latency and errors across individual stages.
- Prompt caching information exposed by the serving integration, including cached input token usage where available.

This makes it possible to inspect how a user request moves through the system and identify where delays or failures occur.

**Shared Langfuse trace:**  
[Open project trace](https://us.cloud.langfuse.com/project/cmvh4olvs01wpad0jy0g2dbq8/traces/b3ad950f7714de500cbae03252c77d49?observation=dc3c3212e9ca7894)


## 9. Known Limitations and Future Improvements

The current implementation demonstrates the complete ingestion, retrieval, generation, and voice workflow, but several areas remain open for improvement.

### Voice experience

- Occasional speech output interruptions or choppiness can affect the smoothness of the conversation.
- End-to-end latency is improved but is not yet as low as desired.
- Further tuning of streaming, turn detection, and speech generation could improve conversational responsiveness.

### Retrieval quality

- Arabic product retrieval is weaker than other evaluated categories.
- A reranking stage could be tested to reorder the initial retrieved candidates using a more detailed relevance model.
- Hybrid search combining dense vector retrieval with keyword-based retrieval could also be evaluated, particularly for exact product names and structured terminology.

### Evaluation coverage

- The retrieval evaluation covers 68 answerable questions across four language combinations.
- The generation dataset contains 78 questions, including 10 unanswerable questions, but the reported generation metrics currently represent one four-question batch.
- More batches and human-reviewed examples are needed before drawing broader conclusions about generation quality.

### Future work

1. Experiment with a reranker and compare retrieval metrics against the current baseline.
2. Evaluate hybrid search for queries involving exact names, product terminology, and structured fields.
3. Improve Arabic product retrieval.
4. Investigate speech streaming and TTS behavior to reduce choppiness.
5. Continue reducing latency across the voice and RAG stages.
6. Run the full generation evaluation dataset in controlled batches and aggregate the results.
7. Expand human evaluation of answer correctness, groundedness, and citation support.

## 10. Repository Structure

```text
voice-rag-agent/
├── data/
│   ├── company/
│   ├── contracts/
│   ├── employees/
│   └── products/
│
├── documents/
│   └── dataLoader_metadata/
│
├── evaluation/
│   ├── retrieval/
│   │   ├── retrieval_evaluation.py
│   │   ├── qdrant_retrieval_evaluation.py
│   │   └── results/
│   │
│   └── generation/
│       ├── evaluators.py
│       ├── metrics.py
│       ├── run_generation_evaluation.py
│       ├── generation_questions.jsonl
│       └── results/
│
├── scripts/
│   └── build_index.py
│
├── src/
│   └── insurellm/
│       ├── config.py
│       ├── ingestion/
│       │   ├── chunker.py
│       │   └── loader.py
│       ├── rag/
│       │   ├── embedding_text.py
│       │   ├── embeddings.py
│       │   ├── pipeline.py
│       │   ├── retrieval.py
│       │   └── vector_store.py
│       ├── llm/
│       │   ├── client.py
│       │   └── prompt.py
│       ├── voice/
│       │   └── agent.py
│       └── observability/
│           ├── tracing.py
│           └── voice_metrics.py
│
├── tests/
│   ├── test_ingestion.py
│   └── test_pdf_extraction.py
│
├── vectordb/
│   └── qdrant/
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## 11. Setup and Running

### Prerequisites

- Python version compatible with the project configuration.
- [uv](https://docs.astral.sh/uv/) package manager.
- API credentials for the configured embedding, LLM, LiveKit, and Langfuse services.

### 1. Clone the repository

```bash
git clone <repository-url>
cd voice-rag-agent
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Add the required API credentials and configuration values to `.env`. Refer to `.env.example` and `src/insurellm/config.py` for the environment variables used by the project.


### 4. Build the vector index

```bash
uv run python scripts/build_index.py
```

This runs the ingestion and indexing workflow and stores the vectors in the configured local Qdrant storage path.

### 5. Run the voice agent

Development mode:

```bash
uv run lk agent dev ./src/insurellm/voice/agent.py
```

Console mode:

```bash
uv run python src/insurellm/voice/agent.py console
```

### 6. Run evaluations

Retrieval evaluation:

```bash
uv run python -m evaluation.retrieval.retrieval_evaluation
```

Qdrant retrieval evaluation:

```bash
uv run python -m evaluation.retrieval.qdrant_retrieval_evaluation
```

Generation evaluation:

```bash
uv run python -m evaluation.generation.run_generation_evaluation
```

The generation evaluation script also supports options such as `--top-k`, `--limit`, and `--no-stream`.


## 12. PDF Extraction Test

The repository includes `tests/test_pdf_extraction.py`, which compares PDF text extraction approaches on an Arabic company document.

It compares:

- `pypdf` text extraction.
- `pdfplumber` text extraction.
- Tesseract OCR using Arabic language data.

This test helps inspect extraction quality and investigate Arabic PDF handling. It is an extraction comparison test, not a claim that OCR is automatically applied to every document in the production ingestion pipeline.

## 13. Demo

A short demo video should demonstrate the main user experience, including:

- Asking questions in Arabic and English.
- Receiving answers grounded in the knowledge base.
- Interrupting the assistant while it is speaking.
- Asking a follow-up or changing the question while preserving conversational context.
- Inspecting the corresponding Langfuse traces.

**Demo video:** [[Open demo video](https://drive.google.com/file/d/1bTNlS3rgXeXuGxZjeZpRhlXYtif3vrcq/view?usp=sharing)]

## 14. Evaluation Artifacts

Evaluation outputs are stored under the corresponding `evaluation/` directories.

They include retrieval metrics, generation metrics, and detailed per-question results where produced by the evaluation scripts.

Langfuse provides trace-level evidence for inspecting individual voice and RAG requests.

