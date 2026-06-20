// Typed fetch wrappers. Keep this thin — TanStack Query handles caching/retries.

import type {
  EmailRow,
  JobState,
  MailFolder,
  OutlookStatus,
  SearchMode,
  SearchResponse,
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
export const search = (q: string, mode: SearchMode, limit = 20) =>
  api<SearchResponse>(`/api/search?q=${encodeURIComponent(q)}&mode=${mode}&limit=${limit}`);

export const getEmail = (id: string) => api<EmailRow>(`/api/emails/${encodeURIComponent(id)}`);
export const getStats = () => api<Stats>('/api/stats');
