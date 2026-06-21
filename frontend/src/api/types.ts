// API types — mirror src/emailsearch/{db,search}/models.py.
// Kept minimal — only what the UI uses.

export type SearchMode = 'keyword' | 'semantic' | 'hybrid';
export type MatchedIn = 'body' | 'attachment' | 'both';
/** Source identifies which search leg produced a given hit. The streaming
 *  endpoint emits one `hits` event per leg as soon as that leg finishes. */
export type LegSource = 'keyword' | 'semantic_fts' | 'semantic_knn';
export type ExtractionStatus = 'ok' | 'skipped_too_large' | 'unsupported' | 'failed' | 'empty';

export interface SearchHit {
  email_id: string;
  subject: string;
  from_address: string;
  from_name: string | null;
  received_at: number; // unix seconds
  snippet: string;
  score: number;
  matched_in: MatchedIn;
  matched_attachment_name: string | null;
  web_link: string | null;
  /** LLM-generated topical summary; null when summarization is disabled or failed. */
  summary: string | null;
}

export interface SearchResponse {
  hits: SearchHit[];
  mode: SearchMode;
  query: string;
  /** Per-search transformation + ranking trace. Populated whenever the
   *  backend's `EMAILSEARCH_DEBUG_ENABLED` setting is true (the default);
   *  `null` when explicitly disabled server-side. Loose-typed on purpose
   *  so the backend can iterate on the shape without breaking the frontend. */
  debug?: Record<string, unknown> | null;
}

// ---------- streaming search events ----------

/** First event emitted by /api/search/stream. Tells the client what query
 *  the backend echoed back, what mode it ran in, and how many legs it's
 *  going to stream. */
export interface SearchStreamMeta {
  type: 'meta';
  query: string;
  mode: SearchMode;
  sources: LegSource[];
  filters: SearchFilters;
  debug_enabled: boolean;
}

/** One leg's complete result. Streamed as soon as that leg finishes, in
 *  whatever order the legs complete (FTS legs are typically sub-ms;
 *  semantic_knn may take a couple of seconds due to LLM query
 *  augmentation). */
export interface SearchStreamHits {
  type: 'hits';
  source: LegSource;
  hits: SearchHit[];
  /** Per-leg trace fragment when debug is enabled. */
  trace?: Record<string, unknown> | null;
}

/** Emitted when one leg crashes — other legs still complete normally. */
export interface SearchStreamError {
  type: 'error';
  source: LegSource | 'unknown';
  message: string;
}

/** Final event. Indicates the stream is complete; the client can stop its
 *  spinner. */
export interface SearchStreamDone {
  type: 'done';
  duration_ms: number;
}

export type SearchStreamEvent =
  | SearchStreamMeta
  | SearchStreamHits
  | SearchStreamError
  | SearchStreamDone;

export interface AttachmentRecord {
  att_id: string;
  name: string;
  content_type: string;
  size: number;
  is_inline: boolean;
  content_id: string | null;
  extracted_text: string;
  status: ExtractionStatus;
  ocr_used: boolean;
  error: string | null;
}

export interface EmailRow {
  id: string;
  subject: string;
  from_address: string;
  from_name: string | null;
  to_addresses: { address: string; name?: string | null }[];
  cc_addresses: { address: string; name?: string | null }[];
  received_at: number;
  sent_at: number | null;
  folder_id: string | null;
  folder_name: string | null;
  conversation_id: string | null;
  body_text: string;
  body_html: string;
  web_link: string | null;
  /** LLM-generated topical summary; null when summarization is disabled or failed. */
  summary: string | null;
  attachments: AttachmentRecord[];
  has_attachments: boolean;
  body_ocr_used: boolean;
}

export interface OutlookStatus {
  available: boolean;
  detail: string;
  backend: string;
}

export interface JobState {
  job_id: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';
  start_at: number;
  end_at: number;
  folder_ids: string[] | null;
  started_at: number;
  finished_at: number | null;
  count_added: number;
  count_skipped: number;
  count_errors: number;
  count_attachments_processed: number;
  last_message_id: string | null;
  error: string | null;
  cancel_requested?: boolean;
}

export interface MailFolder {
  id: string;
  displayName: string;
  store?: string;
  path?: string;
  totalItemCount?: number;
}

export interface Stats {
  emails: number;
  chunks: number;
}

// ---------- search filters ----------

/** Optional hard filters applied to every search mode. All fields are
 *  independently nullable — leave a field undefined to skip that dimension. */
export interface SearchFilters {
  /** Inclusive lower bound on received_at (unix seconds). */
  start_at?: number | null;
  /** Exclusive upper bound on received_at (unix seconds). */
  end_at?: number | null;
  /** Case-insensitive exact match on from_address. */
  from_address?: string | null;
  /** Exact match on folder_id. */
  folder_id?: string | null;
}

export interface SenderFacet {
  address: string;
  name: string | null;
  count: number;
}

export interface FolderFacet {
  folder_id: string;
  folder_name: string;
  count: number;
}

export interface FilterFacets {
  senders: SenderFacet[];
  folders: FolderFacet[];
}
