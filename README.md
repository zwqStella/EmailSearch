# EmailSearch

Local Outlook email search with **keyword + semantic + hybrid** search.
Reads mail directly from your **local Classic Outlook** app via COM automation
— no auth, no Conditional Access, no Entra app registration. Stores everything
in SQLite + sqlite-vec and serves a React UI on `127.0.0.1`. **Your email
content never leaves the machine** — embeddings are computed locally with
sentence-transformers; OCR runs locally with rapidocr-onnxruntime.

## Features

- **No auth, no tokens**: Outlook is already authenticated to your mailbox; we
  just ask it for messages via the COM automation API. Reads from Outlook's
  local OST cache, so it works offline (for cached items).
- **Sees everything Outlook does**: primary mailbox, shared mailboxes, PSTs,
  every folder including custom ones.
- **Idempotent loader**: pick a date range, hit Load. Already-stored messages
  are skipped automatically (keyed on the message's Internet Message-Id).
- **Body + attachment search**: PDFs (`pymupdf`), DOCX, XLSX, plain text, CSV,
  and images (inline + attached) via OCR.
- **Three search modes**: keyword (FTS5 + bm25), semantic (sqlite-vec KNN with
  `paraphrase-multilingual-MiniLM-L12-v2` embeddings, multilingual — English,
  Chinese, 50+ languages), and hybrid (runs every leg and merges client-side
  by max score per email). FTS5 uses the `trigram` tokenizer so CJK substring
  queries work correctly.
- **Hard filters**: narrow any search by date range (presets), sender, or
  folder. Filters apply BEFORE ranking — a folder filter excludes everything
  outside that folder rather than just deprioritizing it.
- **100% local**: no cloud services at all.

## Requirements

- **Windows** with **Classic Outlook** installed (the desktop app you've always
  used). New Outlook for Windows (Monarch) does **not** expose COM and won't
  work — install Classic side-by-side; both share the same mailbox cache.
- Python ≥ 3.11, Node.js ≥ 20 (only for the frontend build).

## Quick start

```powershell
# 1. Install Python + Node deps (first time only)
uv sync
cd frontend; npm install; cd ..

# 2. Run — `start` rebuilds the frontend, stops any stale server on the
#    port, then starts fresh and opens the browser. Re-run after any code
#    change (Python or React) to pick it up.
uv run emailsearch start
# → http://127.0.0.1:8765

# `serve` is the lower-level alternative: it just runs uvicorn (no rebuild,
# no auto-stop). Use it for `--reload` or alongside the Vite dev server.
uv run emailsearch serve --open-browser
```

That's it. There's no sign-in step. Outlook auto-launches the first time we
ask for messages.

## Workflow

1. **Settings tab**: confirms the Outlook backend is connected and shows
   index stats.
2. **Load tab**: pick a date range (default: last 30 days), optionally select
   folders, click **Load emails**. Watch counters tick: `added / skipped /
   errors / attachments`. Re-running with the same range adds 0 new emails.
3. **Search tab**: type a query, switch between **hybrid / keyword / semantic**.
   Results that matched on attachment content show a **📎 filename** badge.

## Architecture

4-table SQLite schema:

- `emails` — Outlook metadata, body text, attachments-as-JSON,
  `searchable_text` (concatenated body + attachment text for FTS).
- `emails_fts` — FTS5 keyword index over `subject + from_address +
  searchable_text`, kept in sync via triggers.
- `vec_email_chunks` — sqlite-vec `vec0` virtual table holding 384-dim
  embeddings with auxiliary columns (`+email_id`, `+source_type`,
  `+source_name`, `+chunk_index`, `+chunk_text`) — no separate chunks table
  needed.
- `sync_jobs` — load-job history, so the Recent jobs list survives server
  restarts. Dangling `running` rows are reconciled to `cancelled` on startup.

LlamaIndex is used **only** as a chunker + embedder library
(`SentenceSplitter` + `HuggingFaceEmbedding`); no `IngestionPipeline`, no
`VectorStoreIndex`, no custom `VectorStore` adapter. Search is plain SQL.

## Development

```powershell
uv run pytest -q             # backend tests
uv run ruff check .          # lint
uv run emailsearch info      # dump resolved paths

cd frontend
npm install
npm run dev                  # http://127.0.0.1:5173 (proxies /api → :8765)
# In another terminal:
uv run emailsearch serve     # backend on :8765 (no --open-browser)
```

### Layout

```
src/emailsearch/
  outlook/       COM client (pywin32) + RawMessage / RawAttachment models
  extract/       PDF/DOCX/XLSX/text loaders + OCR + cid-image splicer
  embed/         LlamaIndex chunker + HF encoder + chunk-builder
  db/            schema.sql + connection + repositories
  search/        keyword + semantic_fts + semantic_knn legs (no fusion; merged client-side)
  sync/          in-memory job registry + idempotent loader
  web/           FastAPI app + routes (status/sync/search)
  eval/          IR-style quality harness (P@K, R@K, MRR, nDCG@K, latency)
  cli.py         `emailsearch start|serve|info`

frontend/
  src/api/       typed fetch wrappers
  src/pages/     SearchPage, LoadPage, SettingsPage
  src/components/EmailPreview.tsx (sandboxed iframe + attachment cards)
```

### Search-quality evaluation

A small IR-style harness lives under `src/emailsearch/eval/`. It runs a
curated set of queries through `search()` in every mode and reports
Precision@5/10/20, Recall@5/10/20, MRR, nDCG@10, and p50/p95 latency,
plus a per-category breakdown (verbatim / topic / semantic /
multilingual / person / thread / attachment) and a per-query
rank-of-first-relevant-hit table.

Queries and ground-truth labels live in **`eval/specs.toml`** — a local
TOML file that's gitignored so your real subject patterns don't get
published. The committed `eval/specs.example.toml` is a template.

```powershell
# 1. Bootstrap your local specs from the example (first time only)
Copy-Item eval/specs.example.toml eval/specs.toml
#    then edit eval/specs.toml to point at your own mailbox patterns.

# 2. Materialize ground-truth IDs from those patterns
uv run python scripts/materialize_queries.py
#    → writes eval/queries.toml with the resolved Message-IDs.

# 3. Sanity-check that every relevance ID resolves in the DB
uv run python -m emailsearch.eval validate

# 4. Run all queries through every mode and write a markdown report
uv run python -m emailsearch.eval run --json-out eval/report.json
#    → writes eval/report.md + eval/report.json.
```

Why config-driven? The relevance labels need to be picked by *subject
or sender pattern* — not by what the search system returns — so the
metrics stay unbiased. Patterns reference real mailbox content, so we
keep them in a local file rather than baked into the script.

## Configuration (`.env`)

All settings have working defaults — no config file is required. To override any
of them, `copy .env.example .env` and edit the values; pydantic-settings picks
them up automatically on next start.

| Variable                       | Default                                                | Purpose                                                |
|--------------------------------|--------------------------------------------------------|--------------------------------------------------------|
| `EMAILSEARCH_DATA_DIR`         | `~/.emailsearch`                                       | Where the DB and ML models live.                       |
| `EMAILSEARCH_DB_PATH`          | `<data_dir>/emails.db`                                 | Override the SQLite path.                              |
| `EMAILSEARCH_EMBED_MODEL`      | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model (multilingual, 384-dim).            |
| `EMAILSEARCH_EMBED_DIM`        | `384`                                                  | Must match the model's vector width.                   |
| `EMAILSEARCH_OCR_ENABLED`      | `true`                                                 | Disable to skip image OCR (much faster).               |
| `EMAILSEARCH_MAX_ATTACHMENT_MB`| `25`                                                   | Larger attachments are recorded as `skipped_too_large`. |
| `EMAILSEARCH_LLM_ENABLED`      | `true`                                                 | LLM summarization + query rerank (see below). Set to `false` if you don't have a local model server running. |
| `EMAILSEARCH_LLM_BASE_URL`     | `http://127.0.0.1:4141/v1`                             | OpenAI-compatible base URL. Default targets [copilot-api](https://github.com/ericc-ch/copilot-api). |
| `EMAILSEARCH_LLM_MODEL`        | `gpt-4o-mini`                                          | Sent as the `model` field. Strict backends (copilot-api, Azure OpenAI) reject unknown names with 400; LM Studio / Ollama / llama.cpp usually ignore this and serve whatever's loaded. |
| `EMAILSEARCH_LLM_TIMEOUT_S`    | `60`                                                   | Per-call timeout in seconds. Claude / GPT-class models can take 10-30s.    |
| `EMAILSEARCH_LLM_MAX_TOKENS`   | `200`                                                  | Cap on summary length (model-side).                    |
| `EMAILSEARCH_LLM_AUGMENT_MAX_TOKENS` | `80`                                             | Cap on query-augmentation length (model-side).         |
| `EMAILSEARCH_LLM_DISTILL_MAX_TOKENS` | `40`                                             | Cap on query-distillation length (model-side).         |
| `EMAILSEARCH_LLM_MAX_INPUT_CHARS` | `8000`                                              | Truncate body before prompting to bound cost / latency. |
| `EMAILSEARCH_DEBUG_ENABLED`    | `true`                                                 | Include a per-request query-transformation trace in `/api/search` (logged to the browser console). |
| `EMAILSEARCH_HOST`             | `127.0.0.1`                                            | **Don't change** unless you understand the implications. |
| `EMAILSEARCH_PORT`             | `8765`                                                 | Server port.                                           |

### LLM summarization + query helpers

When `EMAILSEARCH_LLM_ENABLED=true` (the default), the LLM is used in **two
places**, both best-effort (any failure falls back to the non-LLM path):

1. **At ingest**: every new email is sent to a local OpenAI-compatible
   `/v1/chat/completions` endpoint for a 1-3 sentence topical summary. The
   summary is stored on the email row, shown above the body in the preview
   pane and above the snippet in search results, appended to the FTS
   `searchable_text` so keyword search picks up summary-only terms, AND
   embedded as its own `source_type='summary'` vector so the semantic KNN
   leg can match topical summaries directly.

2. **At query time** (semantic + hybrid modes): the user's query is
   **distilled** to strip natural-language filler ("help me find the email
   about Q3 budget" → "Q3 budget") before the FTS leg, and **augmented**
   with related vocabulary before the embedding step. Emails whose summary
   chunk matched are promoted above body-only matches inside the KNN leg
   (see `search.service.SUMMARY_PROMOTION_BASE`).

Compatible local servers include LM Studio, llama.cpp's `llama-server`,
llamafile, Ollama (with the `/v1` shim enabled), and vLLM. Point
`EMAILSEARCH_LLM_BASE_URL` at whatever your server exposes.

Set `EMAILSEARCH_LLM_ENABLED=false` if you don't have a local model server —
all LLM features will be skipped and search falls back to embedding + bm25
ranking without distillation / augmentation. There's also a live integration
test suite at `tests/test_llm_integration.py` that probes the configured
endpoint and exercises every job kind (summarize / distill / augment); it
skips silently when no server is reachable.

Existing emails that were loaded **before** you enabled this flag won't have
summaries — run `scripts/backfill_summaries.py` to fill them in without
re-loading from Outlook, or clear the index and re-load.

### Diagnostics

Every `/api/search/stream` event carries a per-leg `trace` field with the
full query-transformation + ranking detail (raw query → distilled → FTS
MATCH expression → vec0 top chunks → final ranking). The frontend logs it
to the browser console (collapsed group) on every search so you can
diagnose surprising results without re-issuing the query. The server also
emits `log.info` lines at each step in the same trace, visible in the
uvicorn output. Set `EMAILSEARCH_DEBUG_ENABLED=false` to drop the wire-side
payload (server logs stay on).

## Troubleshooting

**"Outlook unavailable" on the Settings tab.** The COM client couldn't
connect. Most likely:
- You only have **New Outlook for Windows** installed. Install Classic Outlook
  side-by-side; both share the same OST cache.
- Outlook is launching as a different user (e.g. you ran the server as Admin
  but Outlook runs unelevated). Run both at the same elevation.

**The first Load is slow.** We walk every folder and your OST cache may not
yet have all items materialized. Subsequent Loads (and especially re-Loads of
the same range) are much faster.

**A message I expect doesn't show up.** Outlook's `Restrict("[ReceivedTime]
>= ...")` filter occasionally misses items at the boundary. Try widening the
range by one day on each side.

**I changed the embedding model / FTS tokenizer and nothing happened.** Schema
changes only apply to *new* indexes. Open the **Settings** tab and click
**Clear all indexed emails** — this drops the FTS, vec0, and emails tables
and rebuilds them from the current schema. Then re-Load.

## What's intentionally **not** in v1

- Sending or replying to mail.
- Calendar / contacts.
- Nested-message attachments (forwarded emails as attachments) — recorded as
  `unsupported`.
- Background / delta sync — load is user-initiated.
- Encryption at rest (SQLCipher).
- Storing attachment binaries (only the extracted text is kept; the UI's
  "Open in Outlook" link uses `outlook:<EntryID>` to jump to the original).

## License

MIT.
