// API types — mirror src/emailsearch/{db,search}/models.py.
// Kept minimal — only what the UI uses.

export type SearchMode = 'keyword' | 'semantic' | 'hybrid';
export type MatchedIn = 'body' | 'attachment' | 'both';
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
}

export interface SearchResponse {
  hits: SearchHit[];
  mode: SearchMode;
  query: string;
}

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
