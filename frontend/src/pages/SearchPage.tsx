import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import DOMPurify from 'dompurify';
import { search, getEmail } from '../api/client';
import type { SearchHit, SearchMode } from '../api/types';
import EmailPreview from '../components/EmailPreview';

const MODES: SearchMode[] = ['hybrid', 'keyword', 'semantic'];

// FTS5 `snippet()` returns indexed text verbatim with `<mark>` wrappers — it
// does NOT escape the underlying content. Attachment text (PDF/DOCX/CSV/HTML)
// is part of the indexed corpus, so a snippet can legitimately contain
// `<script>` or other markup the user never authored. Strip everything except
// our highlight markers.
const sanitizeSnippet = (html: string) =>
  DOMPurify.sanitize(html, { ALLOWED_TAGS: ['mark'], ALLOWED_ATTR: [] });

function useDebounced<T>(value: T, delay = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return v;
}

function fmtDate(epoch: number): string {
  return new Date(epoch * 1000).toLocaleString();
}

function MatchBadge({ hit }: { hit: SearchHit }) {
  if (hit.matched_in === 'attachment') {
    return (
      <span className="inline-block ml-2 px-2 py-0.5 rounded bg-amber-100 text-amber-900 text-xs">
        📎 {hit.matched_attachment_name ?? 'attachment'}
      </span>
    );
  }
  if (hit.matched_in === 'both') {
    return (
      <span className="inline-block ml-2 px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 text-xs">
        body + 📎 {hit.matched_attachment_name ?? ''}
      </span>
    );
  }
  return null;
}

export default function SearchPage() {
  const [q, setQ] = useState('');
  const [mode, setMode] = useState<SearchMode>('hybrid');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const debounced = useDebounced(q, 250);

  const results = useQuery({
    queryKey: ['search', debounced, mode],
    queryFn: () => search(debounced, mode, 20),
    enabled: debounced.trim().length > 0,
  });

  const previewQuery = useQuery({
    queryKey: ['email', selectedId],
    queryFn: () => getEmail(selectedId!),
    enabled: !!selectedId,
  });

  const hits = useMemo(() => results.data?.hits ?? [], [results.data]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-[calc(100vh-8rem)]">
      <section className="flex flex-col bg-white rounded shadow-sm overflow-hidden">
        <div className="p-3 border-b flex gap-2 items-center">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search emails…"
            className="flex-1 border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            autoFocus
          />
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as SearchMode)}
            className="border rounded px-2 py-2 text-sm bg-white"
            aria-label="Search mode"
          >
            {MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
        <div className="flex-1 overflow-y-auto">
          {results.isFetching && (
            <div className="p-3 text-xs text-gray-500">Searching…</div>
          )}
          {results.error && (
            <div className="p-3 text-sm text-red-700">
              {(results.error as Error).message}
            </div>
          )}
          {!debounced.trim() && (
            <div className="p-6 text-sm text-gray-500">
              Type a query to search across body text and attachments. Try the
              <code className="mx-1 bg-gray-100 px-1 rounded">semantic</code>
              mode for paraphrases.
            </div>
          )}
          {debounced.trim() && !results.isFetching && hits.length === 0 && (
            <div className="p-3 text-sm text-gray-500">No results.</div>
          )}
          <ul className="divide-y">
            {hits.map((h) => (
              <li
                key={h.email_id}
                onClick={() => setSelectedId(h.email_id)}
                className={`p-3 cursor-pointer hover:bg-blue-50 ${
                  selectedId === h.email_id ? 'bg-blue-100' : ''
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="font-medium text-sm text-gray-900 truncate">
                    {h.subject || '(no subject)'}
                    <MatchBadge hit={h} />
                  </div>
                  <span className="text-xs text-gray-500 flex-shrink-0 ml-2">
                    {fmtDate(h.received_at)}
                  </span>
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  {h.from_name ? `${h.from_name} <${h.from_address}>` : h.from_address}
                </div>
                <div
                  className="text-xs text-gray-700 mt-1 line-clamp-3"
                  // Snippet may include attachment-derived text (PDF/CSV/etc.)
                  // — sanitize down to plain text + <mark> only.
                  dangerouslySetInnerHTML={{ __html: sanitizeSnippet(h.snippet) }}
                />
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="bg-white rounded shadow-sm overflow-hidden">
        {!selectedId && (
          <div className="p-6 text-sm text-gray-500">Select a result to preview.</div>
        )}
        {selectedId && previewQuery.isLoading && (
          <div className="p-6 text-sm text-gray-500">Loading…</div>
        )}
        {previewQuery.data && <EmailPreview email={previewQuery.data} />}
      </section>
    </div>
  );
}
