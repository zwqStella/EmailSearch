import { useEffect, useMemo, useRef, useState } from 'react';

/** One row in the dropdown. ``label`` is the primary string the trigger
 *  shows when this option is selected; ``detail`` and ``meta`` are
 *  secondary text the dropdown lists alongside it. All three contribute
 *  to the substring filter. */
export interface SelectOption {
  value: string;
  label: string;
  detail?: string;
  meta?: string;
}

interface Props {
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
  /** Label for the synthetic "no filter" row that always sits on top
   *  and clears the selection (passes ``""`` to ``onChange``). */
  allLabel: string;
  ariaLabel: string;
  /** Tailwind classes applied to the wrapper. The trigger button fills
   *  this width — pass ``flex-1 min-w-0`` etc. here. */
  className?: string;
  searchPlaceholder?: string;
}

/** Combobox built from a button + filtered listbox. We roll our own
 *  instead of a native <select> because the sender facet can have
 *  hundreds of entries, and the native control has no text filter on
 *  Windows / Linux. */
export default function SearchableSelect({
  value,
  onChange,
  options,
  allLabel,
  ariaLabel,
  className = '',
  searchPlaceholder = 'Filter…',
}: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const wrapperRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const currentLabel = useMemo(() => {
    if (!value) return allLabel;
    const match = options.find((o) => o.value === value);
    // Fall back to the raw value when the selected option isn't in the
    // facet list (e.g. the user picked a sender, then the facets refetch
    // dropped it because the corpus changed).
    return match ? match.label : value;
  }, [value, options, allLabel]);

  // Outside click + Esc close. Only wired up while the popover is open
  // so the listeners don't leak.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  // Reset the filter + focus the input every time the menu opens, so a
  // stale query from a previous session doesn't hide everything.
  useEffect(() => {
    if (open) {
      setQuery('');
      inputRef.current?.focus();
    }
  }, [open]);

  const q = query.trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!q) return options;
    return options.filter((o) => {
      const hay = `${o.label} ${o.detail ?? ''} ${o.meta ?? ''}`.toLowerCase();
      return hay.includes(q);
    });
  }, [options, q]);

  const select = (v: string) => {
    onChange(v);
    setOpen(false);
  };

  return (
    <div ref={wrapperRef} className={`relative ${className}`}>
      <button
        type="button"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="w-full border rounded px-2 py-1 bg-white text-left flex items-center justify-between gap-1"
      >
        <span className={`truncate ${value ? '' : 'text-gray-700'}`}>
          {currentLabel}
        </span>
        <span aria-hidden className="text-gray-500 flex-shrink-0">▾</span>
      </button>
      {open && (
        <div
          className="absolute z-20 mt-1 left-0 min-w-[18rem] max-w-[calc(100vw-2rem)] bg-white border rounded shadow-lg"
          role="listbox"
        >
          <div className="p-2 border-b">
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={searchPlaceholder}
              autoComplete="off"
              className="w-full border rounded px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <ul className="max-h-64 overflow-y-auto py-1">
            <li>
              <button
                type="button"
                onClick={() => select('')}
                className={`w-full text-left px-3 py-1.5 hover:bg-blue-50 ${
                  value === '' ? 'bg-blue-100 font-medium' : ''
                }`}
              >
                {allLabel}
              </button>
            </li>
            {filtered.map((o) => (
              <li key={o.value}>
                <button
                  type="button"
                  onClick={() => select(o.value)}
                  className={`w-full text-left px-3 py-1.5 hover:bg-blue-50 ${
                    o.value === value ? 'bg-blue-100' : ''
                  }`}
                >
                  <span className="font-medium">{o.label}</span>
                  {o.detail && (
                    <span className="ml-1 text-gray-600">{o.detail}</span>
                  )}
                  {o.meta && (
                    <span className="ml-2 text-gray-500">· {o.meta}</span>
                  )}
                </button>
              </li>
            ))}
            {filtered.length === 0 && (
              <li className="px-3 py-2 text-gray-500">No matches</li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
