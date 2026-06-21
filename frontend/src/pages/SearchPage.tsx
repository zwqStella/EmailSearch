import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import DOMPurify from 'dompurify';
import { searchStream, getEmail, getFilterFacets } from '../api/client';
import type {
  LegSource,
  SearchFilters,
  SearchHit,
  SearchMode,
  SearchStreamEvent,
} from '../api/types';
import EmailPreview from '../components/EmailPreview';
import SearchableSelect from '../components/SearchableSelect';

const MODES: SearchMode[] = ['hybrid', 'keyword', 'semantic'];

// Date-range presets for the time filter. Value is the lookback window in
// seconds (0 = no lower bound).
const DATE_PRESETS: { label: string; seconds: number }[] = [
  { label: 'Any time', seconds: 0 },
  { label: 'Past 7 days', seconds: 7 * 86400 },
  { label: 'Past 30 days', seconds: 30 * 86400 },
  { label: 'Past 90 days', seconds: 90 * 86400 },
  { label: 'Past year', seconds: 365 * 86400 },
];

// FTS5 `snippet()` returns indexed text verbatim with `<mark>` wrappers and
// does NOT escape the underlying content. Attachment text (PDF/DOCX/CSV/HTML)
// can contain `<script>` or other markup the user never authored. Strip
// everything except our highlight markers.
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

/** Aggregated state we accumulate as per-leg events arrive from
 *  /api/search/stream. Hits live in a Map keyed by email_id so the
 *  "keep the higher score across legs" rule is a one-liner. */
interface SearchStreamState {
  /** Echoed query the server is running. Lags the UI input while a
   *  request is in flight; we use it to label the result panel with
   *  the canonical query. */
  query: string;
  mode: SearchMode;
  /** All sources expected for this run, from the meta event. */
  expectedSources: LegSource[];
  /** Sources we've received a `hits` or `error` event for. */
  arrivedSources: Set<LegSource>;
  /** email_id → best hit seen so far. Metadata rides along with the
   *  winning hit. */
  hitsById: Map<string, SearchHit>;
  /** Per-leg trace fragments for the console debug log. */
  traces: Partial<Record<LegSource, Record<string, unknown> | null | undefined>>;
  /** True while the stream is open (between meta and done). */
  inFlight: boolean;
  /** Per-leg error messages from `error` events. */
  errors: Partial<Record<LegSource | 'unknown', string>>;
  /** Top-level error (e.g. fetch failed before any event arrived). */
  fatalError: string | null;
  /** Total wall-clock for the search, set by the done event. */
  durationMs: number | null;
}

const EMPTY_STATE: SearchStreamState = {
  query: '',
  mode: 'hybrid',
  expectedSources: [],
  arrivedSources: new Set(),
  hitsById: new Map(),
  traces: {},
  inFlight: false,
  errors: {},
  fatalError: null,
  durationMs: null,
};

export default function SearchPage() {
  const [q, setQ] = useState('');
  const [mode, setMode] = useState<SearchMode>('hybrid');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // Filter state is local-only — applied on every search. Default to
  // "Any time" / no sender / no folder.
  const [datePresetIdx, setDatePresetIdx] = useState(0);
  const [senderFilter, setSenderFilter] = useState('');
  const [folderFilter, setFolderFilter] = useState('');

  const debounced = useDebounced(q, 250);

  // Facets are static-ish (change only on ingest); long stale time +
  // cheap backend query → no need for aggressive refetching.
  const facets = useQuery({
    queryKey: ['filter-facets'],
    queryFn: getFilterFacets,
    staleTime: 60_000,
  });

  // Derive the SearchFilters wire object from the dropdown state.
  // Memoized so the empty-filter case produces the same object identity
  // (no spurious refetches).
  const filters = useMemo<SearchFilters>(() => {
    const preset = DATE_PRESETS[datePresetIdx];
    const f: SearchFilters = {};
    if (preset.seconds > 0) {
      f.start_at = Math.floor(Date.now() / 1000) - preset.seconds;
    }
    if (senderFilter) f.from_address = senderFilter;
    if (folderFilter) f.folder_id = folderFilter;
    return f;
  }, [datePresetIdx, senderFilter, folderFilter]);

  const filtersActive =
    datePresetIdx !== 0 || senderFilter !== '' || folderFilter !== '';

  // -- Streaming state machine -----------------------------------------
  //
  // Plain useState rather than TanStack Query because (a) the result
  // arrives incrementally across multiple events, not as a single
  // response, and (b) we want AbortController cancellation when the
  // query changes mid-flight.
  const [streamState, setStreamState] = useState<SearchStreamState>(EMPTY_STATE);
  // The previous AbortController, so we can cancel an in-flight stream
  // when a new search starts.
  const inFlightAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    // Cancel any prior in-flight stream — the user typed something new.
    inFlightAbortRef.current?.abort();
    inFlightAbortRef.current = null;

    const trimmed = debounced.trim();
    if (!trimmed) {
      // Reset to empty state when the input clears.
      setStreamState(EMPTY_STATE);
      return;
    }

    const ac = new AbortController();
    inFlightAbortRef.current = ac;
    // Capture filters / mode at request time so the closure doesn't drift.
    const requestFilters = filters;
    const requestMode = mode;

    // Reset visible state for the new search. We DON'T clear hitsById
    // synchronously — that would cause a 1-frame flash of "No results"
    // between the prior result and the first incoming event.
    setStreamState({
      ...EMPTY_STATE,
      query: trimmed,
      mode: requestMode,
      inFlight: true,
    });

    (async () => {
      try {
        for await (const event of searchStream(
          trimmed, requestMode, 20, requestFilters, ac.signal,
        )) {
          // The user may have started a NEW search while this one was
          // still draining — bail out of applying its events.
          if (ac.signal.aborted) break;
          applyStreamEvent(event, setStreamState);
        }
      } catch (err) {
        if (ac.signal.aborted) return; // expected when superseded
        const message = err instanceof Error ? err.message : String(err);
        setStreamState((s) => ({ ...s, inFlight: false, fatalError: message }));
      }
    })();

    return () => {
      ac.abort();
    };
  }, [debounced, mode, filters]);

  // Sorted hit list. Sort is stable (Array.from preserves insertion order
  // within equal scores), so an email that came in via a fast leg stays
  // visible while a later leg's update for the same email just rewrites
  // its row in place.
  const hits = useMemo(() => {
    return Array.from(streamState.hitsById.values()).sort(
      (a, b) => b.score - a.score,
    );
  }, [streamState.hitsById]);

  // Console debug log — fires once when every expected source has
  // arrived (or a fatal error landed). Per-leg traces render as
  // collapsible groups for inspection.
  const loggedKeyRef = useRef<string | null>(null);
  useEffect(() => {
    const allArrived =
      streamState.expectedSources.length > 0 &&
      streamState.expectedSources.every((s) => streamState.arrivedSources.has(s));
    if (!allArrived && !streamState.fatalError) return;
    // De-dupe so we only log once per completed search.
    const key = `${streamState.query}::${streamState.mode}::${streamState.durationMs}::${streamState.fatalError ?? ''}`;
    if (loggedKeyRef.current === key) return;
    loggedKeyRef.current = key;

    // eslint-disable-next-line no-console
    console.info(
      `🔎 search: ${JSON.stringify(streamState.query)} (${streamState.mode}) → ` +
        `${hits.length} hit(s) across ${streamState.expectedSources.length} leg(s)` +
        (streamState.durationMs != null ? ` in ${streamState.durationMs}ms` : ''),
    );

    if (streamState.fatalError) {
      // eslint-disable-next-line no-console
      console.warn('🔎 search fatal error:', streamState.fatalError);
      return;
    }

    if (Object.keys(streamState.errors).length) {
      // eslint-disable-next-line no-console
      console.warn('🔎 per-leg errors:', streamState.errors);
    }

    if (Object.keys(streamState.traces).length === 0) {
      // No backend traces — most likely cause is the server-side opt-out.
      // eslint-disable-next-line no-console
      console.warn(
        '🔎 no debug traces in /api/search/stream events — set ' +
          'EMAILSEARCH_DEBUG_ENABLED=true (default) and restart the backend ' +
          'to see the full ranking trace.',
      );
      return;
    }

    // eslint-disable-next-line no-console
    console.groupCollapsed(
      `🔎 search debug: ${JSON.stringify(streamState.query)} (${streamState.mode})`,
    );
    for (const src of streamState.expectedSources) {
      const trace = streamState.traces[src];
      if (!trace) continue;
      // eslint-disable-next-line no-console
      console.groupCollapsed(`leg: ${src}`);
      // eslint-disable-next-line no-console
      console.log('trace:', trace);
      // Render the hit-style sub-fields as tables when present.
      const ftsHits = (trace as { fts_hits?: unknown }).fts_hits;
      if (Array.isArray(ftsHits)) {
        // eslint-disable-next-line no-console
        console.log('FTS hits:');
        // eslint-disable-next-line no-console
        console.table(ftsHits);
      }
      const vecChunks = (trace as { vec_top_chunks?: unknown }).vec_top_chunks;
      if (Array.isArray(vecChunks)) {
        // eslint-disable-next-line no-console
        console.log('vec0 top chunks:');
        // eslint-disable-next-line no-console
        console.table(vecChunks);
      }
      const finalScores = (trace as { final_scores?: unknown }).final_scores;
      if (Array.isArray(finalScores)) {
        // eslint-disable-next-line no-console
        console.log('final scores:');
        // eslint-disable-next-line no-console
        console.table(finalScores);
      }
      // eslint-disable-next-line no-console
      console.groupEnd();
    }
    // eslint-disable-next-line no-console
    console.groupEnd();
  }, [
    streamState.query,
    streamState.mode,
    streamState.expectedSources,
    streamState.arrivedSources,
    streamState.traces,
    streamState.errors,
    streamState.fatalError,
    streamState.durationMs,
    hits.length,
  ]);

  const previewQuery = useQuery({
    queryKey: ['email', selectedId],
    queryFn: () => getEmail(selectedId!),
    enabled: !!selectedId,
  });

  const clearFilters = () => {
    setDatePresetIdx(0);
    setSenderFilter('');
    setFolderFilter('');
  };

  const isSearching = streamState.inFlight;
  const showEmptyMsg =
    !!debounced.trim() &&
    !isSearching &&
    hits.length === 0 &&
    !streamState.fatalError;
  // Per-leg progress chip — "2/3 sources" while the slow embedding leg
  // is still working but FTS results are already visible.
  const legProgress = streamState.expectedSources.length > 0
    ? `${streamState.arrivedSources.size}/${streamState.expectedSources.length} sources`
    : '';

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 h-[calc(100vh-7rem)]">
      <section className="flex flex-col bg-white rounded shadow-sm overflow-hidden lg:col-span-5 xl:col-span-4 min-w-0">
        <div className="p-3 border-b flex gap-2 items-center">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search emails…"
            className="flex-1 min-w-0 border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            autoFocus
          />
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as SearchMode)}
            className="border rounded px-2 py-2 text-sm bg-white flex-shrink-0"
            aria-label="Search mode"
          >
            {MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
        {/* Filter bar — hard filters narrow the candidate set BEFORE ranking.
            All three dropdowns + the clear link share a single flex row so
            they don't wrap into ragged stacks in the narrow results column. */}
        <div className="px-3 py-2 border-b bg-gray-50 flex flex-nowrap gap-2 items-center text-xs">
          <select
            value={datePresetIdx}
            onChange={(e) => setDatePresetIdx(Number(e.target.value))}
            className="border rounded px-2 py-1 bg-white min-w-0 flex-shrink"
            aria-label="Date filter"
          >
            {DATE_PRESETS.map((p, i) => (
              <option key={p.label} value={i}>
                {p.label}
              </option>
            ))}
          </select>
          <SearchableSelect
            value={senderFilter}
            onChange={setSenderFilter}
            options={(facets.data?.senders ?? []).map((s) => ({
              value: s.address,
              label: s.name ?? s.address,
              detail: s.name ? `<${s.address}>` : undefined,
              meta: String(s.count),
            }))}
            allLabel="All senders"
            ariaLabel="Sender filter"
            searchPlaceholder="Filter senders…"
            className="min-w-0 flex-1"
          />
          <select
            value={folderFilter}
            onChange={(e) => setFolderFilter(e.target.value)}
            className="border rounded px-2 py-1 bg-white min-w-0 flex-1 truncate"
            aria-label="Folder filter"
          >
            <option value="">All folders</option>
            {facets.data?.folders.map((f) => (
              <option key={f.folder_id} value={f.folder_id}>
                {f.folder_name} · {f.count}
              </option>
            ))}
          </select>
          {filtersActive && (
            <button
              type="button"
              onClick={clearFilters}
              className="text-blue-700 underline whitespace-nowrap flex-shrink-0"
            >
              Clear
            </button>
          )}
        </div>
        <div className="flex-1 overflow-y-auto">
          {isSearching && (
            <div className="p-3 text-xs text-gray-500">
              Searching… {legProgress}
            </div>
          )}
          {streamState.fatalError && (
            <div className="p-3 text-sm text-red-700">
              {streamState.fatalError}
            </div>
          )}
          {!debounced.trim() && (
            <div className="p-6 text-sm text-gray-500">
              Type a query to search across body text and attachments. Try the
              <code className="mx-1 bg-gray-100 px-1 rounded">semantic</code>
              mode for paraphrases.
            </div>
          )}
          {showEmptyMsg && (
            <div className="p-3 text-sm text-gray-500">
              No results
              {filtersActive ? ' (try clearing filters)' : ''}.
            </div>
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
                {h.summary && (
                  // LLM-generated topical summary above the snippet. Plain
                  // text, no markup to sanitize.
                  <div className="text-xs text-gray-800 mt-1 italic line-clamp-2">
                    {h.summary}
                  </div>
                )}
                <div
                  className="text-xs text-gray-700 mt-1 line-clamp-3"
                  // Snippet may contain attachment-derived text — sanitize
                  // down to plain text + <mark> only.
                  dangerouslySetInnerHTML={{ __html: sanitizeSnippet(h.snippet) }}
                />
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="bg-white rounded shadow-sm overflow-hidden lg:col-span-7 xl:col-span-8 min-w-0">
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

/** Apply a single stream event to the accumulator state.
 *
 * Pulled out of the consuming effect so the merge rule ("keep the
 * highest-scoring hit per email_id across legs") is easy to find. Per-leg
 * events arrive in completion order — the score is the only thing that
 * decides position.
 */
function applyStreamEvent(
  event: SearchStreamEvent,
  setState: React.Dispatch<React.SetStateAction<SearchStreamState>>,
) {
  switch (event.type) {
    case 'meta': {
      setState((s) => ({
        ...s,
        expectedSources: event.sources,
        // Reset accumulators defensively (we already reset to EMPTY_STATE
        // before kicking off the request).
        arrivedSources: new Set(),
        hitsById: new Map(),
        traces: {},
        errors: {},
      }));
      return;
    }
    case 'hits': {
      setState((s) => {
        const nextHits = new Map(s.hitsById);
        for (const hit of event.hits) {
          const existing = nextHits.get(hit.email_id);
          // Keep the higher-scoring hit. Snippet / matched_in /
          // attachment name ride along — we don't merge fields across
          // legs.
          if (existing == null || hit.score > existing.score) {
            nextHits.set(hit.email_id, hit);
          }
        }
        const nextArrived = new Set(s.arrivedSources);
        nextArrived.add(event.source);
        const nextTraces = { ...s.traces, [event.source]: event.trace };
        // Only mark "done" when meta has landed AND every expected leg
        // has arrived. Otherwise stay in flight.
        const stillWaiting =
          s.expectedSources.length === 0 ||
          nextArrived.size < s.expectedSources.length;
        return {
          ...s,
          hitsById: nextHits,
          arrivedSources: nextArrived,
          traces: nextTraces,
          inFlight: stillWaiting,
        };
      });
      return;
    }
    case 'error': {
      setState((s) => {
        const nextArrived = new Set(s.arrivedSources);
        // Count errored legs as "arrived" so progress converges.
        if (event.source !== 'unknown') {
          nextArrived.add(event.source);
        }
        const stillWaiting =
          s.expectedSources.length === 0 ||
          nextArrived.size < s.expectedSources.length;
        return {
          ...s,
          arrivedSources: nextArrived,
          errors: { ...s.errors, [event.source]: event.message },
          inFlight: stillWaiting,
        };
      });
      return;
    }
    case 'done': {
      setState((s) => ({
        ...s,
        inFlight: false,
        durationMs: event.duration_ms,
      }));
      return;
    }
  }
}
