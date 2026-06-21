// Typed fetch wrappers. Keep this thin — TanStack Query handles caching/retries.

import type {
  EmailRow,
  FilterFacets,
  JobState,
  MailFolder,
  OutlookStatus,
  SearchFilters,
  SearchMode,
  SearchStreamEvent,
  Stats,
} from './types';

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

// ---------- backend status ----------
export const getOutlookStatus = () => api<OutlookStatus>('/api/outlook/status');
export const triggerOutlookSync = () =>
  api<{ ok: boolean; detail: string }>('/api/outlook/sync', { method: 'POST' });
/** Open an indexed email in the local Outlook app (Classic Outlook via
 *  COM). Backend resolves the stored EntryID and calls Display() — the
 *  ``outlook:<EntryID>`` URL scheme used previously is NOT registered
 *  with Windows and fails browser-side with "scheme does not have a
 *  registered handler". */
export const openEmailInOutlook = (emailId: string) =>
  api<{ ok: boolean; detail: string }>(
    `/api/outlook/open/${encodeURIComponent(emailId)}`,
    { method: 'POST' },
  );

// ---------- sync ----------
export const startLoad = (body: { start: string; end: string; folder_ids?: string[] | null }) =>
  api<{ job_id: string }>('/api/sync/load', { method: 'POST', body: JSON.stringify(body) });

export const getJob = (id: string) => api<JobState>(`/api/sync/jobs/${id}`);
export const listJobs = () => api<{ jobs: JobState[] }>('/api/sync/jobs');
export const cancelJob = (id: string) =>
  api<{ ok: boolean; job_id: string }>(`/api/sync/jobs/${id}/cancel`, { method: 'POST' });
export const clearJobHistory = () =>
  api<{ ok: boolean; deleted: number }>('/api/sync/jobs', { method: 'DELETE' });
export const listFolders = () => api<{ folders: MailFolder[] }>('/api/folders');

export const clearIndex = () =>
  api<{ ok: boolean; deleted: { emails: number; chunks: number } }>(
    '/api/index',
    { method: 'DELETE' },
  );

// ---------- search ----------

/**
 * Stream per-leg search events from /api/search/stream.
 *
 * Each leg (keyword / semantic_fts / semantic_knn) produces a self-scored
 * hit list with no cross-leg fusion; the caller is responsible for
 * merging and reranking by score as events arrive. See
 * `src/pages/SearchPage.tsx` for the merge implementation.
 *
 * Cancellation: pass an `AbortSignal` (e.g. from `AbortController`) so
 * the in-flight search can be aborted when the user changes the query
 * mid-flight. The server detects the disconnect and cancels any
 * still-running legs.
 *
 * The returned async iterator yields one parsed `SearchStreamEvent` per
 * NDJSON line. It terminates when the server closes the stream (after
 * the `done` event) OR when `signal` is aborted (the iterator will
 * raise the abort reason on the next iteration).
 */
export async function* searchStream(
  q: string,
  mode: SearchMode,
  limit = 20,
  filters?: SearchFilters,
  signal?: AbortSignal,
): AsyncGenerator<SearchStreamEvent> {
  // Only emit params that are actually set — the backend treats omitted
  // params as "no filter on this dimension".
  const params = new URLSearchParams({ q, mode, limit: String(limit) });
  if (filters?.start_at != null) params.set('start_at', String(filters.start_at));
  if (filters?.end_at != null) params.set('end_at', String(filters.end_at));
  if (filters?.from_address) params.set('from_address', filters.from_address);
  if (filters?.folder_id) params.set('folder_id', filters.folder_id);

  const res = await fetch(`/api/search/stream?${params.toString()}`, {
    headers: { Accept: 'application/x-ndjson' },
    signal,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  if (!res.body) {
    throw new Error('search stream returned no body');
  }

  // Parse NDJSON incrementally: buffer bytes, split on '\n', JSON.parse
  // each complete line. Anything left in the buffer at end-of-stream is
  // a final partial line (shouldn't happen with a well-behaved server,
  // but we still flush it).
  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Process every complete line currently in the buffer.
      let newlineIdx = buffer.indexOf('\n');
      while (newlineIdx !== -1) {
        const line = buffer.slice(0, newlineIdx).trim();
        buffer = buffer.slice(newlineIdx + 1);
        if (line) {
          try {
            yield JSON.parse(line) as SearchStreamEvent;
          } catch (err) {
            // A malformed JSON line is non-fatal — log + skip so one bad
            // line doesn't kill the whole stream.
            // eslint-disable-next-line no-console
            console.error('searchStream: failed to parse NDJSON line', line, err);
          }
        }
        newlineIdx = buffer.indexOf('\n');
      }
    }
    // Flush any trailing partial line.
    const tail = buffer.trim();
    if (tail) {
      try {
        yield JSON.parse(tail) as SearchStreamEvent;
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('searchStream: failed to parse trailing NDJSON line', tail, err);
      }
    }
  } finally {
    // Release the reader lock so the body can be garbage-collected. If
    // we got here via an AbortError, `cancel()` is a no-op on an
    // already-aborted stream.
    try {
      await reader.cancel();
    } catch {
      // Ignore — the reader may already be closed.
    }
  }
}

export const getFilterFacets = () => api<FilterFacets>('/api/filters');

export const getEmail = (id: string) => api<EmailRow>(`/api/emails/${encodeURIComponent(id)}`);
export const getStats = () => api<Stats>('/api/stats');
