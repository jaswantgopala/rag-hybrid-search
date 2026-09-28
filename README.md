# RAG Pipeline with Hybrid Search Over Internal Docs

A Retrieval-Augmented Generation system that ingests internal documentation, retrieves with **both dense vector search and BM25 keyword search**, and answers with **inline citations that are independently verified** after generation. The whole stack runs on free, local tools at zero ongoing cost.

**Headline result:** 88.5% answer correctness and 85% retrieval relevance on a 20-question golden evaluation set, with 77% of citations independently verified as supporting their claims.

---

## The problem

Most RAG demos embed one PDF and call an LLM. Real internal documentation is messier: multiple formats, exact identifiers (config keys, error codes) that embeddings can miss, and answers that must be traceable to a source. A wrong answer with a confident citation is worse than no answer.

This project tackles those production concerns directly: hybrid retrieval, chunking-strategy decisions backed by data, citation verification, confidence scoring, and a graceful "I don't know" path.

## Architecture

```
Documents (.md .txt .html .pdf)
        |
   Loader  ->  plaintext + metadata (source, page)
        |
   Chunker (fixed-size | structure-aware | semantic)
        |
   Embed + dedupe (cosine > 0.95)
        |------------------------------|
   ChromaDB (dense)               BM25 (sparse)     <- built from the same chunks
        |------------------------------|
   Query -> dense top-k + sparse top-k
        |
   Reciprocal Rank Fusion (0.7 dense / 0.3 sparse)
        |
   Cross-encoder reranker (top 20 -> top 5)
        |
   Local LLM generates answer with [1], [2] citations
        |
   Citation verifier (second LLM call checks each claim vs its source)
        |
   Confidence score -> answer, or graceful fallback if too low
        |
   FastAPI  +  Streamlit dashboard (with live file upload)
```

## Results

Evaluated on a 20-question golden set (direct lookups, multi-hop questions needing two documents, and deliberately unanswerable questions), scored automatically. Fully local pipeline, `structure_aware` chunking:

| Metric | Score |
|---|---|
| Answer correctness (LLM-judged vs golden answer) | 88.5% |
| Faithfulness (verified citation ratio) | 77.1% |
| Retrieval relevance (right source docs retrieved) | 85% |
| Citation accuracy | 77.1% |

### Chunking strategy comparison

Same 20 questions, same corpus, three strategies:

| Metric | fixed_size | structure_aware | semantic |
|---|---|---|---|
| Correctness | 0.885 | 0.885 | 0.730 |
| Faithfulness | 0.771 | 0.771 | 0.825 |
| Retrieval relevance | 0.850 | 0.850 | 0.750 |
| Citation accuracy | 0.771 | 0.771 | 0.825 |

`fixed_size` and `structure_aware` tied, which makes sense on short documents where header-based splitting barely moves chunk boundaries. `semantic` chunking gave up about 15 points of correctness for about 5 points of faithfulness and citation accuracy. Topic-coherent chunks map more cleanly to a single claim but retrieve less precisely. I default to `structure_aware` because it matches fixed-size accuracy and should scale better to longer, hierarchical documents; `semantic` is the better pick when citation precision matters more than raw accuracy.

### Why the reranker matters

After Reciprocal Rank Fusion, the top candidates' scores sit within about 0.01 of each other, so fusion alone barely separates good chunks from mediocre ones. The cross-encoder reads the query and each chunk together and produced a score of about +8 for the correct chunk versus about -11 for the irrelevant ones, a gap of roughly 19 points on a test query.

## Why hybrid beats dense-only for technical docs

Embeddings capture meaning but can miss exact tokens: a function name, a config key, an error code like `5003`. BM25 catches those verbatim matches. RRF fuses the two ranked lists by position rather than trying to compare incompatible score scales, so neither retriever's weakness dominates.

## The honesty layer

- **Grounded prompt:** answer only from the provided context, cite with bracketed numbers, say so when the context is insufficient.
- **Independent citation verification:** every claim is re-checked against its cited chunk by a separate LLM call, since a model can attach `[1]` to a claim `[1]` doesn't actually support.
- **Composite confidence:** 40% retrieval confidence, 35% verified-citation ratio, 25% LLM-judged completeness.
- **Graceful fallback:** below the retrieval-confidence threshold, the system declines to answer and instead reports the closest chunks found and which documents are worth checking manually. Tested against an out-of-corpus question (a parental leave policy that doesn't exist in the docs).

## Tech stack (all free)

| Component | Tool |
|---|---|
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local) |
| Vector store | ChromaDB (persistent, file-based) |
| Sparse search | `rank_bm25` (BM25Okapi) |
| Reranker | cross-encoder `ms-marco-MiniLM-L-6-v2` (local) |
| LLM (generation + judging) | Ollama, `llama3.1` (local) |
| Chunking | LangChain text splitters |
| API | FastAPI |
| Dashboard | Streamlit |

The pipeline started on Groq's free API tier but exhausted its 200K tokens/day cap during evaluation (each question triggers several LLM calls). Moving every LLM call to local Ollama removed rate limits entirely, at the cost of slower per-call latency.

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/ask` | Question in; answer, sources, and confidence breakdown out |
| `GET /v1/documents` | List indexed documents |
| `POST /v1/ingest` | Re-run ingestion and rebuild all indexes |
| `POST /v1/upload` | Upload new files, then re-ingest automatically |

Interactive docs are served at `/docs`.

## Run it locally

Prerequisites: Python 3.11+, [Ollama](https://ollama.com) with `llama3.1` pulled.

```bash
pip install -r requirements.txt
ollama pull llama3.1

# put documents in data/raw/, then build the indexes
python -m src.ingestion.loader
python -m src.ingestion.chunker
python -m src.ingestion.indexer

# two terminals
uvicorn src.api.main:app --reload
streamlit run src/api/dashboard.py
```

Run the evaluation and chunking comparison with `python -m src.eval.run_eval` and `python -m src.eval.compare_strategies`.

## Limitations

- **Small eval set.** 20 questions is enough to show the framework works and to compare strategies directionally, but not for tight statistical confidence. Growing it past 50 is the first next step.
- **Small corpus.** Evaluated on a small internal-docs-style corpus, so absolute numbers should not be read as production-scale figures.
- **Judge and generator share a model.** Correctness and citation checks use the same local model family that generates answers, which can bias scores. A separate, stronger judge would be more rigorous.
- **Dense-vs-hybrid toggle is a placeholder.** The dashboard has a "compare hybrid vs dense-only" checkbox that isn't wired up yet.
- **No containerization.** A Dockerfile and compose file were started but not finished (the container couldn't load the HuggingFace models offline), so the project runs locally.

## Project structure

```
src/
  ingestion/   loader.py, chunker.py, indexer.py
  retrieval/   retriever.py, fusion.py, reranker.py
  generation/  generator.py, citation_verifier.py, confidence.py,
               graceful_fallback.py, local_llm.py
  eval/        run_eval.py, compare_strategies.py
  api/         main.py, dashboard.py
data/
  raw/  processed/  chroma_db/  bm25_index/  eval/
```
