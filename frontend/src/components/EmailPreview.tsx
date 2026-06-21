import { useMemo, useState } from 'react';
import DOMPurify from 'dompurify';
import type { AttachmentRecord, EmailRow } from '../api/types';

function fmtDate(epoch: number): string {
  return new Date(epoch * 1000).toLocaleString();
}

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function statusColor(s: AttachmentRecord['status']): string {
  switch (s) {
    case 'ok':
      return 'bg-emerald-100 text-emerald-900';
    case 'empty':
      return 'bg-gray-100 text-gray-700';
    case 'failed':
      return 'bg-red-100 text-red-800';
    default:
      return 'bg-amber-100 text-amber-900';
  }
}

export default function EmailPreview({ email }: { email: EmailRow }) {
  // body_html is guaranteed set whenever body_text has content (see
  // extract/pipeline.py). DOMPurify strips any active content.
  const sanitized = useMemo(() => DOMPurify.sanitize(email.body_html), [email.body_html]);

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b">
        <div className="flex items-start justify-between gap-3">
          <h2 className="text-lg font-semibold text-gray-900">
            {email.subject || '(no subject)'}
          </h2>
          {email.web_link && (
            <a
              href={email.web_link}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-blue-700 underline whitespace-nowrap mt-1"
            >
              Open in Outlook
            </a>
          )}
        </div>
        <div className="text-sm text-gray-700 mt-1">
          {email.from_name ? `${email.from_name} <${email.from_address}>` : email.from_address}
        </div>
        <div className="text-xs text-gray-500 mt-0.5">{fmtDate(email.received_at)}</div>
        {email.to_addresses.length > 0 && (
          <div className="text-xs text-gray-600 mt-1">
            To: {email.to_addresses.map((a) => a.address).join(', ')}
          </div>
        )}
        {email.summary && (
          // LLM-generated topical summary card. Plain text — no markup
          // to sanitize.
          <div className="mt-3 p-2 rounded bg-blue-50 border border-blue-100 text-xs text-blue-900">
            <span className="font-semibold mr-1">Summary:</span>
            {email.summary}
          </div>
        )}
      </div>
      {/* Body — sandboxed to prevent any active content. */}
      <iframe
        title="email-body"
        srcDoc={sanitized}
        sandbox=""
        className="flex-1 border-0 bg-white"
      />
      {email.attachments.length > 0 && (
        <div className="border-t bg-gray-50 max-h-64 overflow-y-auto">
          <h3 className="px-4 py-2 text-xs font-semibold text-gray-700 uppercase">
            Attachments
          </h3>
          <ul className="divide-y">
            {email.attachments.map((a) => (
              <AttachmentCard key={a.att_id} att={a} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function AttachmentCard({ att }: { att: AttachmentRecord }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="px-4 py-2 text-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span>📎</span>
          <span className="font-medium text-gray-900">{att.name}</span>
          <span className="text-xs text-gray-500">{fmtSize(att.size)}</span>
          {att.ocr_used && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-purple-100 text-purple-900">
              OCR
            </span>
          )}
          <span
            className={`text-xs px-1.5 py-0.5 rounded ${statusColor(att.status)}`}
          >
            {att.status}
          </span>
        </div>
        {att.extracted_text && (
          <button
            onClick={() => setOpen((v) => !v)}
            className="text-xs text-blue-700 underline"
          >
            {open ? 'hide' : 'show text'}
          </button>
        )}
      </div>
      {open && att.extracted_text && (
        <pre className="mt-2 p-2 bg-white border rounded text-xs whitespace-pre-wrap max-h-40 overflow-y-auto">
          {att.extracted_text}
        </pre>
      )}
    </li>
  );
}
