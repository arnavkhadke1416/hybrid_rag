# Hybrid RAG

A Python project for document-grounded question answering using Gemini,
Sentence Transformers, FAISS, BM25, and reciprocal-rank fusion. It includes
setup and configuration, PDF/DOCX ingestion, chunk persistence, dense and
keyword retrieval, rank fusion, grounded answer generation, and a command-line
RAG service.

## Architecture

The workflow is explicit and local-first: PDFs and DOCX files in
`documents/` are parsed and chunked with source metadata; chunks are persisted
under `data/processed/`; the configured Sentence Transformers model builds a
FAISS dense index while BM25 builds a keyword index over the same chunk
snapshot; reciprocal-rank fusion combines retrieved candidates; and the CLI
sends the question plus selected excerpts to Gemini to produce a grounded
answer with validated source labels. Ingestion and question answering are
separate commands. The evaluation runner uses a synthetic corpus and does
not call Gemini.

## Prerequisites and environment setup

Use Python 3.12 (tested baseline), PowerShell on Windows, and internet access
for dependency and initial embedding-model downloads. A Gemini API key is
required only for answer generation and authenticated model verification.
The configured retrieval stack is CPU-first and uses `faiss-cpu`; a host GPU
does not imply GPU-enabled embeddings or acceleration.

Create and activate a virtual environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the editable project and pinned runtime and
development dependencies from `pyproject.toml`. Check local setup and CLI
availability without calling Gemini:

```powershell
python scripts\check_setup.py
python -m hybrid_rag.cli --help
```

## Configuration

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` to a valid Google AI
Studio API key:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` locally; never paste its values into logs, issue trackers, or
source control. Keep
`GEMINI_MODEL` set to the exact model ID you have verified for that key; the
requested starting value is `gemini-3.7-flash`, and this project will not
silently substitute another model.

Check local setup and missing configuration without displaying secrets:

```powershell
python scripts\check_setup.py
```

Once the key and model ID are configured, verify model access with an
authenticated metadata request (no generation request is made):

```powershell
python scripts\check_setup.py --verify-model
```

The default document input is `documents/`. Add authorized PDFs and DOCX
files there; this is the sole default source location. Discovery is recursive.
Scanned PDFs without extractable text are reported; OCR is not performed.
Generated manifests and indexes belong under `data/`, not alongside source
documents. See [documents/README.md](./documents/README.md) for file formats
and handling details.

## Current commands

After installing the project, display the CLI help:

```powershell
python -m hybrid_rag.cli --help
```

Ingest documents under `documents/`, persist chunks, and build both indexes:

```powershell
python scripts\ingest_documents.py
```

Chunk size and overlap are measured in characters. Chunking preserves
paragraph boundaries where practical and retains source/page metadata. The
ingestion command compares SHA-256 hashes with the previous manifest, reuses
unchanged chunks, and reports added, changed, removed, failed, and unchanged
files. If nothing changed and both indexes already match the current version,
it skips indexing and does not re-embed the corpus. Individual load failures
are reported; successfully processed files are still indexed.

Ask one question using the indexes already on disk:

```powershell
python -m hybrid_rag.cli ask "How do I restore account access?"
python -m hybrid_rag.cli ask "Which encryption method is used?" --diagnostics
```

Asking does not ingest or rebuild documents. For multiple questions in one
process, use the interactive command and enter `quit` or `exit` to stop:

```powershell
python -m hybrid_rag.cli interactive
python -m hybrid_rag.cli interactive --diagnostics
```

Explicitly reprocess all source files and rebuild chunks and indexes with:

```powershell
python scripts\rebuild_indexes.py
python scripts\ingest_documents.py --force
```

For a safe rebuild after a configuration or schema change, ensure source
files are present in `documents/`, run `python scripts\rebuild_indexes.py`,
review its added/changed/removed/failed counts and index version, then retry
the intended `ask` command. Do not manually delete one index at a time.
Indexes/chunks are staged before publication and the manifest is published
last; a subsequent ingestion run recovers an interrupted publication.

`python scripts\build_indexes.py` is a compatibility command that also
performs incremental discovery and ensures both indexes are current. Index
files carry the same version marker, derived from the ordered chunk snapshot,
embedding model, tokenizer, chunk configuration, and index schemas. The
manifest is published only after both indexes have built successfully.
Indexes and chunks are staged before publication; if publication is
interrupted, the next ingestion run recovers the previous or committed set.
No background filesystem watcher runs.

Changing the embedding model, BM25 tokenizer, chunk size/overlap, or chunk
schema requires a rebuild. Such configuration changes are detected through
the index version and invalidate the previous pair; `rebuild_indexes.py`
forces a full reprocessing and rebuild.

The dense path uses the exact configured
`sentence-transformers/all-MiniLM-L6-v2` model. It normalizes document and
query vectors and uses FAISS `IndexFlatIP` for cosine-like ranking. The index
and its chunk-ID mapping are stored under `data/indexes/faiss/`; the mapping
includes an index checksum so mismatched files are rejected. Query retrieval
is available through `hybrid_rag.retrieval.dense.retrieve_dense`.

The keyword path uses `rank-bm25` with the versioned
`lowercase-tech-token-v1` tokenizer. It lowercases terms while preserving
technical forms such as `AES-256`, version strings, product codes, and
identifiers. Its corpus, tokenizer metadata, and chunk-ID mapping are stored
as JSON under `data/indexes/bm25/`; no pickle deserialization is used.
`hybrid_rag.retrieval.bm25.retrieve_bm25` returns matching `Chunk` records
and BM25 scores. Both retrieval paths return the shared `RetrievalResult`
record shape. `HybridRetriever` composes both paths and RRF:

```python
from hybrid_rag.retrieval.hybrid import HybridRetriever

retriever = HybridRetriever()
response = retriever.retrieve("How do I restore account access?", diagnostic=True)
for result in response.results:
    print(result.chunk.source_path, result.chunk.page_number, result.chunk.text)
for item in response.diagnostics:
    print(item.dense_rank, item.bm25_rank, item.fused_rank, item.source_path, item.chunk_id)
```

`HybridRetriever` validates and whitespace-normalizes each query, retrieves
dense and BM25 candidates, then fuses them with RRF. `DENSE_TOP_K`,
`BM25_TOP_K`, `FUSION_TOP_K`, and `RRF_K` configure candidate counts and
fusion. RRF uses 1-based ranks and adds `1 / (rrf_k + rank)` for each list
containing a chunk. Ties are resolved by stable chunk ID; dense and BM25
scores and citation metadata remain attached. Diagnostic mode logs and
returns dense ranks, BM25 ranks, fused ranks, source paths, and chunk IDs.

Generate an answer from the retrieved results with the official Google Gen
AI SDK:

```python
from hybrid_rag.generation.gemini import GeminiGenerator

answer = GeminiGenerator().generate(question, response.results)
print(answer.answer)
for source in answer.sources:
    print(source.citation_label, source.file_name, source.page_number, source.chunk_id)
```

Generation requires `GEMINI_API_KEY` and uses the exact `GEMINI_MODEL` from
configuration; it does not substitute another model. Retrieved excerpts are
treated as untrusted evidence, citation labels are checked against the
context sent, and unknown labels are rejected. The client uses bounded
retries for transient failures and a 30-second request timeout. It does not
log prompts or credentials by default. A generation request sends the
question and retrieved excerpts to Google; do not submit sensitive material
unless that use is approved. Verify account access after configuring the key:

```powershell
python scripts\check_setup.py --verify-model
```

## Retrieval evaluation

Run the deterministic retrieval benchmark with its small, synthetic,
manually labeled corpus. It does not read `documents/` or call Gemini:

```powershell
python scripts\evaluate_retrieval.py
python -m ruff check .
python -m pytest
```

Detailed rankings, disagreements, timings, dataset hash, and parameter
sweeps are saved to `evaluation/results.json`. On the five-query evaluation
set, the measured retrieval metrics were:

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.875 | 1.000 | 1.000 | 1.000 |
| BM25 | 0.875 | 1.000 | 1.000 | 1.000 |
| Hybrid | 0.875 | 1.000 | 1.000 | 1.000 |

The page-citation case retrieved its labeled page chunk. The candidate-count
and RRF sweep found no metric improvement over the current configuration;
its deterministic tie-break selected candidate depth 3 and `rrf_k=10`, but
the production defaults remain unchanged because all tested settings tied.
The chunking sweep rebuilt the synthetic corpus with four size/overlap
pairs. The existing 800/120 setting had the best Recall@1 (1.000, compared
with 0.875 for the other three pairs), so it was retained. Re-chunked
relevance is assigned when a result overlaps at least half of a manually
labeled source chunk.

The benchmark ran on CPU with the configured
`sentence-transformers/all-MiniLM-L6-v2` model and six synthetic chunks.
This run measured about 3.9 seconds to load the model, 0.071 seconds to
build FAISS, 0.006 seconds to build BM25, and mean per-query latency of
0.012 seconds for dense, 0.0005 seconds for BM25, and 0.013 seconds for
hybrid retrieval. These are single-run timings on the development machine,
not deployment targets; real corpus size, cold model downloads, and host
hardware will change them. Peak memory was not reliably measured, so no
numeric RAM guarantee is made. The embedding model and its PyTorch runtime
are the main memory consumers; profile memory with a representative corpus
before deployment. Although an NVIDIA GPU is present on the development
host, the benchmark explicitly used CPU execution; GPU acceleration was
not tested.

This is a small synthetic retrieval set, not a representative quality
study. Dense and BM25 disagree on rankings, but each method retrieves all
labeled relevant chunks by rank 3. Retrieval alone also returns candidates
for the unsupported question; the reported non-empty retrieval rate is not
an abstention-quality measure. Citation validity, answer faithfulness, and
generation abstention remain unmeasured because the benchmark makes no
Gemini requests. Expand and review the evaluation set before making
production-quality or generation-quality claims.

## Privacy and API usage

Document parsing, embedding, indexing, and retrieval run locally after the
embedding model has been downloaded. The first model load may download model
weights. Each successful `ask` sends the question and retrieved excerpts to
Google Gemini; it may incur API charges. Do not submit material unless you
are authorized to send it to that service. `scripts\check_setup.py
--verify-model` performs only an authenticated model-metadata request, while
the synthetic evaluation makes no Gemini requests.

The project `.gitignore` excludes `.env` files (except the safe
`.env.example` template), local `data/` indexes/manifests, and files placed
under `documents/`. Keep those protections in place. Review ignore rules
before adding private files elsewhere in the workspace.

## Common errors and fixes

- **Gemini credentials are missing:** if `.env` does not exist, copy
  `.env.example` to `.env`; set a
  valid `GEMINI_API_KEY`, then run `python scripts\check_setup.py`. The key
  value is never printed. Generation will not work until credentials are
  configured.
- **The model is unavailable or unauthorized:** run
  `python scripts\check_setup.py --verify-model` and confirm the exact
  `GEMINI_MODEL` ID and key permissions with the project owner. Do not
  silently substitute another model. The authenticated metadata check in
  this development environment passed for `gemini-3.7-flash` and reported
  `generateContent` support; it did not make a generation request.
- **Indexes or chunks are missing/incompatible:** run
  `python scripts\ingest_documents.py` for normal synchronization, or
  `python scripts\rebuild_indexes.py` after an index configuration/schema
  change. Avoid deleting index files manually.
- **A source file fails to load:** read the failure reported by ingestion,
  correct or replace the file, and rerun ingestion. Password-protected,
  corrupt, empty, unsupported, and image-only/scanned files cannot be
  indexed as ordinary text; OCR is not included.
- **The embedding model cannot load:** verify network access for its initial
  download and confirm the exact configured model remains
  `sentence-transformers/all-MiniLM-L6-v2`.

## Current scope and model verification

Phases 0 through 12 are implemented: environment/setup checks, validated
settings and typed records, PDF/DOCX extraction, normalization, chunking,
JSONL/manifest persistence, exact-model embeddings, dense FAISS retrieval,
BM25 keyword retrieval, Reciprocal Rank Fusion, hybrid retrieval, grounded
Gemini generation, service/command-line workflows, and incremental ingestion
with transactional index publication, plus synthetic retrieval evaluation,
linting, operational documentation, and final handover guidance. The exact
configured embedding model was loaded and smoke-tested in the development
environment (384 dimensions). An authenticated metadata check confirmed
that `gemini-3.7-flash` supports `generateContent` for the configured key.
This does not verify generation quality; application tests mock Gemini, and
no Gemini generation request was made during this work.
