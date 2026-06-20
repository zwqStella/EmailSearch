import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  cancelJob,
  clearJobHistory,
  getJob,
  listFolders,
  listJobs,
  startLoad,
  triggerOutlookSync,
} from '../api/client';
import type { JobState } from '../api/types';

function isoDate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function statusColor(s: JobState['status']): string {
  switch (s) {
    case 'running':
      return 'bg-blue-100 text-blue-800';
    case 'succeeded':
      return 'bg-emerald-100 text-emerald-800';
    case 'failed':
      return 'bg-red-100 text-red-800';
    case 'cancelled':
      return 'bg-amber-100 text-amber-900';
    default:
      return 'bg-gray-100 text-gray-800';
  }
}

export default function LoadPage() {
  const qc = useQueryClient();
  const today = new Date();
  const monthAgo = new Date(today);
  monthAgo.setDate(today.getDate() - 30);

  const [start, setStart] = useState(isoDate(monthAgo));
  const [end, setEnd] = useState(isoDate(today));
  const [selectedFolders, setSelectedFolders] = useState<string[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const folders = useQuery({
    queryKey: ['folders'],
    queryFn: listFolders,
    // Don't probe Outlook on page mount — folder walks can take a while on
    // big mailboxes. The user clicks "Load folder list" to trigger it.
    enabled: false,
    staleTime: 5 * 60_000,
  });
  // The server's job list is the source of truth. We poll it on a short
  // interval whenever ANY job is running, so the "Active job" card survives
  // navigation away and back — local state alone was lost on unmount.
  const jobs = useQuery({
    queryKey: ['jobs'],
    queryFn: listJobs,
    refetchInterval: (q) => {
      const data = q.state.data as { jobs: JobState[] } | undefined;
      const anyRunning = (data?.jobs ?? []).some(
        (j) => j.status === 'running' || j.status === 'pending',
      );
      return anyRunning ? 1000 : false;
    },
  });

  // Discover the running job from the server-backed list, falling back to
  // the locally-stashed id if the user just started something but the next
  // jobs poll hasn't landed yet.
  const runningFromList = (jobs.data?.jobs ?? []).find(
    (j) => j.status === 'running' || j.status === 'pending',
  );
  const effectiveJobId = runningFromList?.job_id ?? activeJobId;

  const activeJob = useQuery({
    queryKey: ['job', effectiveJobId],
    queryFn: () => getJob(effectiveJobId!),
    enabled: !!effectiveJobId,
    refetchInterval: (q) => {
      const data = q.state.data as JobState | undefined;
      if (!data) return 1000;
      return data.status === 'running' || data.status === 'pending' ? 1000 : false;
    },
  });

  // When the active job reaches a terminal state, refresh the recent-jobs list.
  useEffect(() => {
    const data = activeJob.data;
    if (!data) return;
    if (data.status === 'succeeded' || data.status === 'failed' || data.status === 'cancelled') {
      qc.invalidateQueries({ queryKey: ['jobs'] });
    }
  }, [activeJob.data, qc]);

  const loadMutation = useMutation({
    mutationFn: () =>
      startLoad({
        // Send as midnight UTC of the chosen day. End is exclusive ⇒ +1 day on the date input.
        start: new Date(`${start}T00:00:00Z`).toISOString(),
        end: new Date(`${end}T00:00:00Z`).toISOString(),
        folder_ids: selectedFolders.length > 0 ? selectedFolders : null,
      }),
    onSuccess: (resp) => {
      setActiveJobId(resp.job_id);
      qc.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  const syncMutation = useMutation({ mutationFn: triggerOutlookSync });

  const cancelMutation = useMutation({
    mutationFn: (id: string) => cancelJob(id),
    onSuccess: () => {
      // Snappier feedback than waiting for the next poll.
      qc.invalidateQueries({ queryKey: ['job', effectiveJobId] });
      qc.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  const clearHistoryMutation = useMutation({
    mutationFn: clearJobHistory,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <section className="lg:col-span-2 bg-white rounded shadow-sm p-4">
        <h2 className="text-base font-semibold mb-3">Load emails from Outlook</h2>
        <div className="grid grid-cols-2 gap-3">
          <label className="text-sm">
            <span className="block text-gray-700">From (UTC)</span>
            <input
              type="date"
              value={start}
              onChange={(e) => setStart(e.target.value)}
              className="mt-1 w-full border rounded px-2 py-1 text-sm"
            />
          </label>
          <label className="text-sm">
            <span className="block text-gray-700">To (UTC, exclusive)</span>
            <input
              type="date"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
              className="mt-1 w-full border rounded px-2 py-1 text-sm"
            />
          </label>
        </div>

        <div className="mt-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-700">Folders (empty = all)</span>
            <div className="flex items-center gap-2">
              {folders.isFetching && <span className="text-xs text-gray-500">loading…</span>}
              {folders.error && (
                <span className="text-xs text-red-700">
                  {(folders.error as Error).message}
                </span>
              )}
              <button
                onClick={() => folders.refetch()}
                disabled={folders.isFetching}
                className="text-xs px-2 py-1 rounded border bg-white hover:bg-gray-100 disabled:opacity-60"
              >
                {folders.data ? 'Refresh' : 'Load folder list'}
              </button>
            </div>
          </div>
          <div className="mt-2 max-h-44 overflow-y-auto border rounded p-2 bg-gray-50">
            {(folders.data?.folders ?? []).map((f) => (
              <label key={f.id} className="flex items-center gap-2 text-sm py-0.5">
                <input
                  type="checkbox"
                  checked={selectedFolders.includes(f.id)}
                  onChange={(e) =>
                    setSelectedFolders((cur) =>
                      e.target.checked ? [...cur, f.id] : cur.filter((x) => x !== f.id),
                    )
                  }
                />
                {f.displayName}
                {typeof f.totalItemCount === 'number' && (
                  <span className="text-xs text-gray-500">({f.totalItemCount})</span>
                )}
              </label>
            ))}
            {!folders.isFetching && folders.data == null && (
              <div className="text-xs text-gray-500">
                Folder list isn't loaded yet. Click <strong>Load folder list</strong> to
                fetch from Outlook — or skip this and leave it empty to load from all
                folders.
              </div>
            )}
            {!folders.isFetching && folders.data != null && folders.data.folders.length === 0 && (
              <div className="text-xs text-gray-500">No mail folders found.</div>
            )}
          </div>
        </div>

        <div className="mt-4 flex gap-3 items-center flex-wrap">
          <button
            onClick={() => loadMutation.mutate()}
            disabled={loadMutation.isPending}
            className="px-4 py-2 rounded bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-60"
          >
            {loadMutation.isPending ? 'Starting…' : 'Load emails'}
          </button>
          <button
            onClick={() => syncMutation.mutate()}
            disabled={syncMutation.isPending}
            className="px-3 py-2 rounded border bg-white text-sm hover:bg-gray-100 disabled:opacity-60"
            title="Trigger Outlook Send/Receive — pulls fresh items into the local cache before you load."
          >
            {syncMutation.isPending ? 'Asking Outlook…' : 'Sync Outlook now'}
          </button>
          {loadMutation.error && (
            <span className="text-sm text-red-700">
              {(loadMutation.error as Error).message}
            </span>
          )}
          {syncMutation.data && (
            <span
              className={`text-xs ${
                syncMutation.data.ok ? 'text-emerald-700' : 'text-red-700'
              }`}
            >
              {syncMutation.data.detail}
            </span>
          )}
        </div>
        <p className="text-xs text-gray-500 mt-2 max-w-prose">
          <strong>Missing older emails?</strong> Outlook only caches a limited
          time window locally (default 1 year for Cached Exchange Mode). To
          index older messages, open Outlook → <em>File → Account Settings →
          double-click your account → "Mail to keep offline"</em> and drag the
          slider to <em>All</em>. After Outlook finishes syncing (it can take a
          while), come back and Load again.
        </p>

        {activeJob.data && (
          <div className="mt-5 border rounded p-3 bg-gray-50">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-medium">Active job</span>
              <div className="flex items-center gap-2">
                {(activeJob.data.status === 'pending' ||
                  activeJob.data.status === 'running') && (
                  <button
                    onClick={() => cancelMutation.mutate(activeJob.data.job_id)}
                    disabled={
                      cancelMutation.isPending || activeJob.data.cancel_requested
                    }
                    className="text-xs px-2 py-0.5 rounded border border-red-200 bg-white text-red-700 hover:bg-red-50 disabled:opacity-60"
                  >
                    {activeJob.data.cancel_requested
                      ? 'Stopping…'
                      : cancelMutation.isPending
                        ? 'Sending…'
                        : 'Stop'}
                  </button>
                )}
                <span
                  className={`text-xs px-2 py-0.5 rounded ${statusColor(activeJob.data.status)}`}
                >
                  {activeJob.data.status}
                </span>
              </div>
            </div>
            <div className="grid grid-cols-4 gap-2 text-sm">
              <Stat label="added" value={activeJob.data.count_added} />
              <Stat label="skipped" value={activeJob.data.count_skipped} />
              <Stat label="errors" value={activeJob.data.count_errors} />
              <Stat label="attachments" value={activeJob.data.count_attachments_processed} />
            </div>
            {activeJob.data.error && (
              <div className="mt-2 text-xs text-red-700">{activeJob.data.error}</div>
            )}
          </div>
        )}
      </section>

      <section className="bg-white rounded shadow-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-base font-semibold">Recent jobs</h3>
          <button
            onClick={() => {
              if (
                window.confirm(
                  'Delete all finished jobs from history? Running jobs are kept.',
                )
              )
                clearHistoryMutation.mutate();
            }}
            disabled={
              clearHistoryMutation.isPending ||
              (jobs.data?.jobs ?? []).length === 0
            }
            className="text-xs px-2 py-1 rounded border bg-white text-gray-700 hover:bg-gray-100 disabled:opacity-60"
          >
            {clearHistoryMutation.isPending ? 'Clearing…' : 'Clear history'}
          </button>
        </div>
        <ul className="divide-y text-sm">
          {(jobs.data?.jobs ?? []).map((j) => (
            <li key={j.job_id} className="py-2">
              <div className="flex items-center justify-between">
                <span
                  className={`text-xs px-2 py-0.5 rounded ${statusColor(j.status)}`}
                >
                  {j.status}
                </span>
                <span className="text-xs text-gray-500">
                  {new Date(j.start_at * 1000).toLocaleDateString()} –{' '}
                  {new Date(j.end_at * 1000).toLocaleDateString()}
                </span>
              </div>
              <div className="text-xs text-gray-700 mt-1">
                +{j.count_added} / skip {j.count_skipped} / err {j.count_errors}
              </div>
            </li>
          ))}
          {(jobs.data?.jobs ?? []).length === 0 && (
            <li className="text-xs text-gray-500 py-2">No jobs yet.</li>
          )}
        </ul>
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="text-xl font-semibold text-gray-900">{value}</div>
      <div className="text-xs text-gray-600">{label}</div>
    </div>
  );
}
