-- EmailSearch v3 schema. 3 tables:
--   1. emails              — metadata + body + attachments-as-JSON + searchable_text for FTS
--   2. emails_fts          — FTS5 contentless table mirroring emails (for keyword search)
--   3. vec_email_chunks    — sqlite-vec vec0 virtual table for embeddings (with aux columns)

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- 1) Email row. attachments is JSON; searchable_text is the FTS5 input.
CREATE TABLE IF NOT EXISTS emails (
    id              TEXT PRIMARY KEY,    -- Graph message id, idempotency key
    subject         TEXT,
    from_address    TEXT,
    from_name       TEXT,
    to_addresses    TEXT,                -- JSON array of {address, name}
    cc_addresses    TEXT,                -- JSON array of {address, name}
    received_at     INTEGER NOT NULL,    -- unix epoch seconds
    sent_at         INTEGER,
    folder_id       TEXT,
    folder_name     TEXT,
    conversation_id TEXT,
    body_text       TEXT,                -- plain text; inline-image OCR spliced in
    body_html       TEXT,                -- preserved verbatim for preview iframe
    summary         TEXT,                -- optional LLM-generated 1-3 sentence summary (NULL when disabled or generation failed); included in searchable_text so FTS picks it up
    web_link        TEXT,                -- Graph webLink → "Open in Outlook"
    attachments     TEXT,                -- JSON array of {att_id,name,content_type,size,extracted_text,status}
    searchable_text TEXT,                -- body_text + " " + concat(attachments[*].extracted_text)
    has_attachments INTEGER NOT NULL DEFAULT 0,
    body_ocr_used   INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS emails_received_at_idx ON emails(received_at DESC);
CREATE INDEX IF NOT EXISTS emails_folder_idx       ON emails(folder_id);

-- 2) FTS5 keyword index. content='emails' makes it contentless (no double storage);
--    we sync via triggers below.
--    tokenize='trigram' (overlapping 3-character shingles) handles CJK correctly:
--    Chinese/Japanese/Korean has no whitespace, so substring matching is what we need.
--    Works fine for English too; bm25 ranking is approximate for ASCII as a trade-off.
CREATE VIRTUAL TABLE IF NOT EXISTS emails_fts USING fts5(
    subject,
    from_address,
    searchable_text,
    content='emails',
    content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS emails_ai AFTER INSERT ON emails BEGIN
    INSERT INTO emails_fts(rowid, subject, from_address, searchable_text)
    VALUES (new.rowid, new.subject, new.from_address, new.searchable_text);
END;

CREATE TRIGGER IF NOT EXISTS emails_ad AFTER DELETE ON emails BEGIN
    INSERT INTO emails_fts(emails_fts, rowid, subject, from_address, searchable_text)
    VALUES('delete', old.rowid, old.subject, old.from_address, old.searchable_text);
END;

CREATE TRIGGER IF NOT EXISTS emails_au AFTER UPDATE ON emails BEGIN
    INSERT INTO emails_fts(emails_fts, rowid, subject, from_address, searchable_text)
    VALUES('delete', old.rowid, old.subject, old.from_address, old.searchable_text);
    INSERT INTO emails_fts(rowid, subject, from_address, searchable_text)
    VALUES (new.rowid, new.subject, new.from_address, new.searchable_text);
END;

-- 3) Vector store. vec0 supports auxiliary columns (prefixed `+`) which carry per-row
--    metadata without forcing a join. We hold chunk text + email linkage right here.
--    NOTE: float[N] dimension MUST match config.embed_dim; changing the model later
--    requires dropping & re-embedding (out of scope for v1).
CREATE VIRTUAL TABLE IF NOT EXISTS vec_email_chunks USING vec0(
    chunk_id      TEXT PRIMARY KEY,
    embedding     float[384],
    +email_id     TEXT,
    +source_type  TEXT,                 -- 'body' | 'attachment' | 'summary'
    +source_name  TEXT,                 -- attachment filename when source_type='attachment'
    +chunk_index  INTEGER,
    +chunk_text   TEXT
);

-- 4) Load-job history. In-memory state is mirrored here on every status / counter
--    change so the UI keeps its history across server restarts. Job IDs are
--    UUID hex strings; folder_ids is a JSON array (or NULL = all folders).
CREATE TABLE IF NOT EXISTS sync_jobs (
    job_id                       TEXT PRIMARY KEY,
    status                       TEXT NOT NULL,   -- pending|running|succeeded|failed|cancelled
    start_at                     INTEGER NOT NULL,
    end_at                       INTEGER NOT NULL,
    folder_ids                   TEXT,            -- JSON array or NULL
    started_at                   INTEGER NOT NULL DEFAULT 0,
    finished_at                  INTEGER,
    count_added                  INTEGER NOT NULL DEFAULT 0,
    count_skipped                INTEGER NOT NULL DEFAULT 0,
    count_errors                 INTEGER NOT NULL DEFAULT 0,
    count_attachments_processed  INTEGER NOT NULL DEFAULT 0,
    last_message_id              TEXT,
    error                        TEXT,
    created_at                   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS sync_jobs_created_idx ON sync_jobs(created_at DESC);
