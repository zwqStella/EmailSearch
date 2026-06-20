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
  Chinese, 50+ languages), and hybrid (Reciprocal Rank Fusion). FTS5 uses the
  `trigram` tokenizer so CJK substring queries work correctly.
- **100% local**: no cloud services at all.

## Requirements

- **Windows** with **Classic Outlook** installed (the desktop app you've always
  used). New Outlook for Windows (Monarch) does **not** expose COM and won't
  work — install Classic side-by-side; both share the same mailbox cache.
- Python ≥ 3.11, Node.js ≥ 20 (only for the frontend build).

## Quick start

```powershell
# 1. Install Python + Node deps
uv sync
cd frontend; npm install; npm run build; cd ..

# 2. Run (no .env edits required — defaults are sensible)
uv run emailsearch serve --open-browser
# → http://127.0.0.1:8765
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
  search/        keyword + semantic + hybrid (RRF)
  sync/          in-memory job registry + idempotent loader
  web/           FastAPI app + routes (status/sync/search)
  cli.py         `emailsearch serve|info`

frontend/
  src/api/       typed fetch wrappers
  src/pages/     SearchPage, LoadPage, SettingsPage
  src/components/EmailPreview.tsx (sandboxed iframe + attachment cards)
```

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
| `EMAILSEARCH_HOST`             | `127.0.0.1`                                            | **Don't change** unless you understand the implications. |
| `EMAILSEARCH_PORT`             | `8765`                                                 | Server port.                                           |

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
