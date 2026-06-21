import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { askStream, getEmail } from '../api/client';
import type {
  AskMode,
  AskParsedFilters,
  AskStreamEvent,
  SearchHit,
} from '../api/types';
import EmailPreview from '../components/EmailPreview';

/** Inline `[N]` citation markers in the answer text. Group 1 is the
 *  1-based source index. The same regex is used to (a) split the
 *  rendered answer into text + citation buttons and (b) determine
 *  which sources are "cited" for the References footnote ring. */
const CITATION_RE = /\[(\d+)\]/g;

function fmtDate(epoch: number): string {
  return new Date(epoch * 1000).toLocaleString();
}

function fmtRange(start: number | null, end: number | null): string {
  // `start_at` / `end_at` are epoch seconds. We only render a date
  // window — the time-of-day is always 00:00 from the parser so
  // showing hours/minutes would be misleading noise.
  const dayOnly = (epoch: number) =>
    new Date(epoch * 1000).toLocaleDateString();
  if (start != null && end != null) return `${dayOnly(start)} → ${dayOnly(end)}`;
  if (start != null) return `from ${dayOnly(start)}`;
  if (end != null) return `before ${dayOnly(end)}`;
  return '';
}

/** Aggregated state we accumulate as per-event chunks arrive from
 *  /api/ask/stream. Modeled like SearchPage's `SearchStreamState` so
 *  the page-level data flow is recognizable across both tabs. */
interface AskStreamState {
  /** The submitted question. Lags the input box while a request is
   *  in flight — the input is for the NEXT question, not the
   *  currently-streaming one. */
  question: string;
  mode: AskMode;
  /** Filler-stripped query the agent fed to the search tool. `null`
   *  until the `parsed` event arrives. */
  parsedQuery: string | null;
  /** Filters extracted from the question by the parser. `null` until
   *  the `parsed` event arrives. */
  parsedFilters: AskParsedFilters | null;
  /** Sources returned by the search tool. Populated by the `sources`
   *  event; inline `[N]` citations resolve into this array. */
  sources: SearchHit[];
  /** True once the `sources` event has been processed — distinguishes
   *  "search returned zero hits" (legitimately empty list) from
   *  "search hasn't completed yet" (also empty list). Drives the
   *  stage indicator between events. */
  sourcesArrived: boolean;
  /** 1-based hit indexes the triage hop chose to read in FULL. `null`
   *  until the `triage` event arrives. Used by the References
   *  footnote to ring the triaged sources distinctly from the merely
   *  retrieved ones. */
  triagedIndexes: number[] | null;
  /** True when the triage hop actually ran (vs. was skipped because
   *  there was nothing to narrow). Drives the "Read N of M emails"
   *  badge in the stage indicator + References header. */
  triaged: boolean;
  /** Streaming answer text, accumulated from `answer_delta` events. */
  answer: string;
  /** True between `meta` and `done`/`error`. */
  inFlight: boolean;
  /** Server-side error event (e.g. parse / search / synthesis
   *  failure), or fetch/transport error. */
  error: string | null;
  /** Total wall-clock from the `done` event. */
  durationMs: number | null;
}

const EMPTY_STATE: AskStreamState = {
  question: '',
  mode: 'hybrid',
  parsedQuery: null,
  parsedFilters: null,
  sources: [],
  sourcesArrived: false,
  triagedIndexes: null,
  triaged: false,
  answer: '',
  inFlight: false,
  error: null,
  durationMs: null,
};

export default function AskPage() {
  const [input, setInput] = useState('');
  const [streamState, setStreamState] = useState<AskStreamState>(EMPTY_STATE);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  // AbortController for the in-flight stream. We hold a ref so the
  // submit handler can cancel a prior request before starting a new
  // one, AND so the unmount cleanup can abort.
  const inFlightAbortRef = useRef<AbortController | null>(null);

  // Cancel any in-flight request on unmount.
  useEffect(() => {
    return () => {
      inFlightAbortRef.current?.abort();
    };
  }, []);

  const submit = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed) return;

    // Cancel any prior in-flight stream — the user is asking
    // something new and we don't want stale deltas trickling in.
    inFlightAbortRef.current?.abort();
    inFlightAbortRef.current = null;

    const ac = new AbortController();
    inFlightAbortRef.current = ac;

    // Reset state synchronously. We DON'T wait for the first event —
    // that would leave the prior answer visible until the network
    // round-trip lands, which is confusing.
    setStreamState({
      ...EMPTY_STATE,
      question: trimmed,
      inFlight: true,
    });
    setSelectedSourceId(null);

    (async () => {
      try {
        for await (const event of askStream(trimmed, {}, ac.signal)) {
          if (ac.signal.aborted) break;
          applyAskEvent(event, setStreamState);
        }
      } catch (err) {
        if (ac.signal.aborted) return; // expected when superseded
        const message = err instanceof Error ? err.message : String(err);
        setStreamState((s) => ({ ...s, inFlight: false, error: message }));
      }
    })();
  }, [input]);

  // Set of 1-based source indexes referenced from inside the answer.
  // Used by the References footnote to ring the actually-cited sources.
  const citedIndexes = useMemo(() => {
    const idxs = new Set<number>();
    for (const match of streamState.answer.matchAll(CITATION_RE)) {
      const n = Number(match[1]);
      if (!Number.isNaN(n) && n >= 1 && n <= streamState.sources.length) {
        idxs.add(n);
      }
    }
    return idxs;
  }, [streamState.answer, streamState.sources.length]);

  const previewQuery = useQuery({
    queryKey: ['email', selectedSourceId],
    queryFn: () => getEmail(selectedSourceId!),
    enabled: !!selectedSourceId,
  });

  const handleCitationClick = useCallback(
    (index: number) => {
      const src = streamState.sources[index - 1];
      if (src) setSelectedSourceId(src.email_id);
    },
    [streamState.sources],
  );

  const filtersActive =
    !!streamState.parsedFilters &&
    (streamState.parsedFilters.start_at != null ||
      streamState.parsedFilters.end_at != null ||
      streamState.parsedFilters.from_address != null);

  // Stage indicator for the long phases between observable events.
  // Each phase below is a real LLM round-trip (parse hop, triage hop)
  // or a composite (search = parallel distill + augment + embed + DB),
  // so a 30-60s ask can sit on any one of them for many seconds with
  // nothing else moving in the UI. `null` once answer tokens start
  // arriving — the streaming text + caret IS the progress at that
  // point.
  let currentStage: string | null = null;
  if (streamState.inFlight && !streamState.answer) {
    if (!streamState.parsedQuery) {
      currentStage = 'Thinking… parsing question';
    } else if (!streamState.sourcesArrived) {
      currentStage = 'Searching your emails…';
    } else if (streamState.triagedIndexes == null) {
      currentStage = 'Selecting most relevant emails…';
    } else {
      const n = streamState.triagedIndexes.length;
      currentStage = streamState.triaged
        ? `Reading ${n} of ${streamState.sources.length} email${
            streamState.sources.length === 1 ? '' : 's'
          }…`
        : 'Writing answer…';
    }
  }

  // References footnote is only rendered AFTER the stream finishes
  // (per UX spec — see plan §References footnote). The inline `[N]`
  // markers in the answer body work as soon as `sources` arrives.
  const showReferences = !streamState.inFlight && streamState.sources.length > 0;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 h-[calc(100vh-7rem)]">
      <section className="flex flex-col bg-white rounded shadow-sm overflow-hidden lg:col-span-7 xl:col-span-7 min-w-0">
        {/* Composer ---------------------------------------------------- */}
        <div className="p-3 border-b flex gap-2 items-start">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              // Submit on Enter; allow Shift+Enter for newlines so
              // longer multi-line questions stay possible.
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder='Ask a question about your emails (e.g. "when is the garage day this month?")'
            className="flex-1 min-w-0 border rounded px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={2}
            autoFocus
          />
          <button
            type="button"
            onClick={submit}
            disabled={!input.trim() || streamState.inFlight}
            className="px-4 py-2 rounded bg-blue-600 text-white text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700"
          >
            {streamState.inFlight ? 'Asking…' : 'Ask'}
          </button>
        </div>

        {/* Conversation panel ---------------------------------------- */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {!streamState.question && (
            <div className="text-sm text-gray-500">
              Type a question above to get a grounded answer from your indexed emails.
              The agent searches for relevant emails (you'll see the inferred query
              and filters), then writes an answer with inline <code className="bg-gray-100 px-1 rounded">[N]</code>
              citations linking to the source emails.
            </div>
          )}

          {streamState.question && (
            <>
              {/* Echo the user's question so the panel reads like a Q&A. */}
              <div className="text-sm">
                <span className="font-semibold text-gray-700">Q: </span>
                <span className="text-gray-900">{streamState.question}</span>
              </div>

              {/* Inferred query + filter pills. */}
              {(streamState.parsedQuery || filtersActive) && (
                <div className="flex flex-wrap gap-2 text-xs">
                  {streamState.parsedQuery && (
                    <span className="px-2 py-1 rounded-full bg-blue-50 text-blue-900 border border-blue-100">
                      🔎 {streamState.parsedQuery}
                    </span>
                  )}
                  {streamState.parsedFilters?.start_at != null ||
                  streamState.parsedFilters?.end_at != null ? (
                    <span className="px-2 py-1 rounded-full bg-emerald-50 text-emerald-900 border border-emerald-100">
                      📅 {fmtRange(
                        streamState.parsedFilters.start_at,
                        streamState.parsedFilters.end_at,
                      )}
                    </span>
                  ) : null}
                  {streamState.parsedFilters?.from_address && (
                    <span className="px-2 py-1 rounded-full bg-amber-50 text-amber-900 border border-amber-100">
                      ✉️ from {streamState.parsedFilters.from_address}
                    </span>
                  )}
                </div>
              )}

              {/* Error banner — surfaces both server-side error events
                  and transport failures. */}
              {streamState.error && (
                <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-3">
                  {streamState.error}
                </div>
              )}

              {/* Stage indicator — shown only when in-flight AND the
                  answer hasn't started streaming yet. Once answer
                  tokens arrive, the streaming text + caret IS the
                  progress. */}
              {currentStage && (
                <div
                  className="flex items-center gap-2 text-sm text-gray-600"
                  role="status"
                  aria-live="polite"
                >
                  <ThinkingDots />
                  <span>{currentStage}</span>
                </div>
              )}

              {/* Answer body. Renders inline `[N]` markers as clickable
                  buttons as soon as `sources` has landed. We only show
                  the "A:" row once we actually have answer text —
                  otherwise the stage indicator above carries the
                  progress. */}
              {streamState.answer && (
                <div className="text-sm">
                  <span className="font-semibold text-gray-700">A: </span>
                  <AnswerBody
                    answer={streamState.answer}
                    sources={streamState.sources}
                    onCitationClick={handleCitationClick}
                  />
                  {streamState.inFlight && (
                    <span className="inline-block ml-1 w-1 h-4 align-text-bottom bg-gray-400 animate-pulse" />
                  )}
                </div>
              )}

              {/* References footnote — held until `done` per UX spec.
                  Lists ALL retrieved sources. Two independent visual
                  cues:
                    - subtle ring on emails the LLM CITED in the
                      answer (signal: "this is where the answer came
                      from")
                    - dimmed opacity on emails the triage hop did NOT
                      pick (signal: "the LLM didn't even read this").
                  Together they tell the user "of 8 retrieved, we
                  read 3, and 1 of those 3 ended up in the answer". */}
              {showReferences && (
                <div className="mt-6 pt-4 border-t">
                  <h3 className="text-xs font-semibold text-gray-700 uppercase mb-2">
                    References ({streamState.sources.length})
                    {streamState.triaged &&
                      streamState.triagedIndexes != null && (
                        <span className="ml-2 font-normal text-gray-500 normal-case">
                          · read {streamState.triagedIndexes.length} of{' '}
                          {streamState.sources.length}
                        </span>
                      )}
                    {streamState.durationMs != null && (
                      <span className="ml-2 font-normal text-gray-500">
                        · {(streamState.durationMs / 1000).toFixed(1)}s
                      </span>
                    )}
                  </h3>
                  <ol className="space-y-2">
                    {streamState.sources.map((hit, i) => {
                      const n = i + 1;
                      const cited = citedIndexes.has(n);
                      const selected = selectedSourceId === hit.email_id;
                      // Treat all as "read" when triage was skipped so
                      // we don't dim every source on the small-result
                      // path.
                      const wasRead =
                        !streamState.triaged ||
                        (streamState.triagedIndexes?.includes(n) ?? true);
                      let title = 'Retrieved but not read';
                      if (cited) {
                        title = 'Cited in the answer';
                      } else if (wasRead) {
                        title = 'Read for the answer';
                      }
                      return (
                        <li
                          key={hit.email_id}
                          onClick={() => setSelectedSourceId(hit.email_id)}
                          className={
                            'flex gap-2 p-2 rounded cursor-pointer text-xs hover:bg-blue-50 ' +
                            (selected ? 'bg-blue-100 ' : '') +
                            (cited ? 'ring-1 ring-blue-300 ' : '') +
                            (wasRead ? '' : 'opacity-50')
                          }
                          title={title}
                        >
                          <span className="font-mono text-blue-700 flex-shrink-0">
                            [{n}]
                          </span>
                          <div className="min-w-0 flex-1">
                            <div className="font-medium text-gray-900 truncate">
                              {hit.subject || '(no subject)'}
                            </div>
                            <div className="text-gray-600 truncate">
                              {hit.from_name
                                ? `${hit.from_name} <${hit.from_address}>`
                                : hit.from_address}
                              {' · '}
                              {fmtDate(hit.received_at)}
                            </div>
                          </div>
                        </li>
                      );
                    })}
                  </ol>
                </div>
              )}

              {/* No-hits + done state — explicit so the user knows the
                  request finished, not that the spinner just hung. */}
              {!streamState.inFlight &&
                !streamState.error &&
                streamState.sources.length === 0 && (
                  <div className="text-xs text-gray-500 italic">
                    No emails matched the search.
                  </div>
                )}
            </>
          )}
        </div>
      </section>

      {/* Email preview pane --------------------------------------- */}
      <section className="bg-white rounded shadow-sm overflow-hidden lg:col-span-5 xl:col-span-5 min-w-0">
        {!selectedSourceId && (
          <div className="p-6 text-sm text-gray-500">
            Click a <code className="bg-gray-100 px-1 rounded">[N]</code> citation or a
            reference row to preview the source email here.
          </div>
        )}
        {selectedSourceId && previewQuery.isLoading && (
          <div className="p-6 text-sm text-gray-500">Loading…</div>
        )}
        {previewQuery.data && <EmailPreview email={previewQuery.data} />}
      </section>
    </div>
  );
}

/** Three animated dots that visibly cycle so a stationary status
 *  string still reads as "actively working". CSS-only — the
 *  Tailwind `animate-bounce` utility with staggered delays is
 *  enough; no need for a third-party spinner library or a sprite.
 */
function ThinkingDots() {
  return (
    <span className="inline-flex gap-0.5" aria-hidden="true">
      <span className="w-1 h-1 rounded-full bg-gray-500 animate-bounce [animation-delay:-0.3s]" />
      <span className="w-1 h-1 rounded-full bg-gray-500 animate-bounce [animation-delay:-0.15s]" />
      <span className="w-1 h-1 rounded-full bg-gray-500 animate-bounce" />
    </span>
  );
}

/** Render the answer text with inline `[N]` citations as buttons.
 *
 * Split on the citation regex; alternating segments are plain text
 * vs citation markers. Out-of-range citations (LLM hallucinated a
 * `[7]` when only 4 sources came back) render as plain text — better
 * to show the marker harmlessly than to silently strip it.
 */
function AnswerBody({
  answer,
  sources,
  onCitationClick,
}: {
  answer: string;
  sources: SearchHit[];
  onCitationClick: (index: number) => void;
}) {
  const parts = useMemo(() => {
    // String.split with a capturing group keeps the captured digit
    // groups in the output array — alternating text / digit / text / digit.
    return answer.split(/\[(\d+)\]/g);
  }, [answer]);

  return (
    <span className="text-gray-900 whitespace-pre-wrap">
      {parts.map((part, i) => {
        // Even indexes (0, 2, 4, ...) are plain text; odd indexes
        // are the captured citation numbers.
        if (i % 2 === 0) return <span key={i}>{part}</span>;
        const n = Number(part);
        const inRange = !Number.isNaN(n) && n >= 1 && n <= sources.length;
        if (!inRange) return <span key={i}>{`[${part}]`}</span>;
        return (
          <button
            key={i}
            type="button"
            onClick={() => onCitationClick(n)}
            className="inline-flex items-baseline px-1 mx-0.5 text-xs font-mono text-blue-700 bg-blue-50 rounded hover:bg-blue-100"
            title={`Open source ${n}: ${sources[n - 1].subject || '(no subject)'}`}
          >
            [{n}]
          </button>
        );
      })}
    </span>
  );
}

/** Apply a single Ask stream event to the accumulator state.
 *
 * Pulled out of the submit handler so the per-event reducer logic
 * is easy to find. `error` events lock the stream into a terminal
 * error state but DON'T clobber whatever was already received (so
 * the user can still see the sources / partial answer when synthesis
 * is what failed).
 */
function applyAskEvent(
  event: AskStreamEvent,
  setState: React.Dispatch<React.SetStateAction<AskStreamState>>,
) {
  switch (event.type) {
    case 'meta': {
      setState((s) => ({ ...s, mode: event.mode }));
      return;
    }
    case 'parsed': {
      setState((s) => ({
        ...s,
        parsedQuery: event.query,
        parsedFilters: event.filters,
      }));
      return;
    }
    case 'sources': {
      setState((s) => ({ ...s, sources: event.hits, sourcesArrived: true }));
      return;
    }
    case 'triage': {
      setState((s) => ({
        ...s,
        triagedIndexes: event.selected_indexes,
        triaged: event.triaged,
      }));
      return;
    }
    case 'answer_delta': {
      setState((s) => ({ ...s, answer: s.answer + event.text }));
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
    case 'error': {
      setState((s) => ({ ...s, inFlight: false, error: event.message }));
      return;
    }
  }
}
