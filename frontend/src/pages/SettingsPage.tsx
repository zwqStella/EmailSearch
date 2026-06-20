import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { clearIndex, getOutlookStatus, getStats } from '../api/client';

export default function SettingsPage() {
  const qc = useQueryClient();
  const status = useQuery({
    queryKey: ['outlook-status'],
    queryFn: getOutlookStatus,
    // Probe once on mount; user can click Refresh if they want a re-check.
    // The probe is cheap (no folder walk) but still opens a COM session,
    // so we don't poll on a timer.
    staleTime: 60_000,
  });
  const stats = useQuery({ queryKey: ['stats'], queryFn: getStats });

  const clear = useMutation({
    mutationFn: clearIndex,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['stats'] });
      qc.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  const onClear = () => {
    if (clear.isPending) return;
    const n = stats.data?.emails ?? 0;
    const ok = window.confirm(
      n > 0
        ? `Delete ${n} indexed email(s) and ${stats.data?.chunks ?? 0} vector chunk(s)?\n\n` +
          "Your Outlook mailbox is NOT touched — this only clears EmailSearch's local index. " +
          'You can re-load from the Load tab.'
        : 'Reset the index? (No data to delete, but tables will be rebuilt — picks up any schema changes.)',
    );
    if (ok) clear.mutate();
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <section className="bg-white rounded shadow-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-base font-semibold">Outlook backend</h2>
          <button
            onClick={() => status.refetch()}
            disabled={status.isFetching}
            className="text-xs px-2 py-1 rounded border bg-white hover:bg-gray-100 disabled:opacity-60"
          >
            {status.isFetching ? 'Checking…' : 'Refresh'}
          </button>
        </div>
        {status.isLoading && (
          <div className="text-sm text-gray-500">Checking Outlook…</div>
        )}
        {status.data && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span
                className={`text-xs px-2 py-0.5 rounded ${
                  status.data.available
                    ? 'bg-emerald-100 text-emerald-900'
                    : 'bg-red-100 text-red-900'
                }`}
              >
                {status.data.available ? 'connected' : 'unavailable'}
              </span>
              <span className="text-sm text-gray-700">{status.data.detail}</span>
            </div>
            <div className="text-xs text-gray-600">
              Backend: <code>{status.data.backend}</code>
            </div>
          </div>
        )}
        {status.data && !status.data.available && (
          <div className="mt-4 text-sm text-amber-900 bg-amber-50 border border-amber-200 rounded p-3">
            Make sure <strong>Classic Outlook</strong> is installed and running on this
            machine. New Outlook for Windows does not expose COM and won't work here.
            Outlook auto-launches when this app first asks for messages.
          </div>
        )}
        <p className="text-xs text-gray-500 mt-4">
          EmailSearch reads mail directly from your local Outlook app via COM
          automation — no auth tokens, no Conditional Access, no Entra app
          registration. Whatever Outlook can see, this app can index.
        </p>
      </section>

      <section className="bg-white rounded shadow-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-base font-semibold">Index</h2>
          <button
            onClick={onClear}
            disabled={clear.isPending}
            className="text-xs px-2 py-1 rounded border border-red-200 bg-white text-red-700 hover:bg-red-50 disabled:opacity-60"
          >
            {clear.isPending ? 'Clearing…' : 'Clear all indexed emails'}
          </button>
        </div>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <Stat label="Emails" value={stats.data?.emails ?? 0} />
          <Stat label="Vector chunks" value={stats.data?.chunks ?? 0} />
        </div>
        {clear.error && (
          <div className="mt-2 text-xs text-red-700">
            {(clear.error as Error).message}
          </div>
        )}
        {clear.data && (
          <div className="mt-2 text-xs text-emerald-700">
            Cleared {clear.data.deleted.emails} email(s) and{' '}
            {clear.data.deleted.chunks} chunk(s). Schema rebuilt.
          </div>
        )}
        <p className="text-xs text-gray-600 mt-3">
          Embedding model:{' '}
          <code>sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2</code>{' '}
          (384-dim, multilingual). Database is stored locally under{' '}
          <code>~/.emailsearch/emails.db</code>.
        </p>
        <p className="text-xs text-gray-500 mt-2">
          Clearing only deletes the local SQLite index — your Outlook mailbox is
          untouched. Use this after changing the embedding model or FTS tokenizer
          so the next Load picks up the new schema.
        </p>
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="border rounded p-3 bg-gray-50">
      <div className="text-2xl font-semibold text-gray-900">{value}</div>
      <div className="text-xs text-gray-600">{label}</div>
    </div>
  );
}
