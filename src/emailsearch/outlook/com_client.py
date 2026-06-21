"""Read mail from the local Classic Outlook app via COM automation.

Why COM?
- Outlook is already authenticated to your mailbox — no tokens, no Conditional
  Access, no Entra app registration.
- Reads from Outlook's local OST cache, so it works offline (for cached items).
- Sees every store the user has mounted (primary mailbox, shared mailboxes,
  PST archives), every folder, including custom ones.

Limitations:
- Windows + Classic Outlook only. New Outlook for Windows ("Monarch") does not
  expose COM and will not work here.
- Outlook must be installed and runnable. COM auto-launches it on first call.
- COM is apartment-threaded; each thread that touches `Outlook.Application`
  must call `pythoncom.CoInitialize()`. The `OutlookClient` does this for you
  on construction and uninitializes on close.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from emailsearch.outlook.raw import RawAttachment, RawMessage
from emailsearch.util import to_utc

log = logging.getLogger(__name__)


# --- Outlook constants we care about. Documented values; safe to inline. ----

OL_MAIL_ITEM_TYPE = 0  # DefaultItemType for mail folders
OL_RECIPIENT_TO = 1
OL_RECIPIENT_CC = 2

# `Class` discriminator: 43 = MailItem. We skip non-mail (meeting requests = 53,
# tasks = 48, contacts = 40, etc.) — they live in mail folders sometimes.
OL_OBJECT_CLASS_MAIL = 43

# DASL property URIs for items we can't get via the high-level API.
PR_INTERNET_MESSAGE_ID = "http://schemas.microsoft.com/mapi/proptag/0x1035001F"
PR_ATTACH_CONTENT_ID = "http://schemas.microsoft.com/mapi/proptag/0x3712001F"
PR_SMTP_ADDRESS = "http://schemas.microsoft.com/mapi/proptag/0x39FE001F"


class OutlookUnavailableError(RuntimeError):
    """Raised when we can't talk to Outlook (not Windows / Outlook not installed / new-Outlook-only)."""


# ---------------------------------------------------------------------------
# Folder model (small, JSON-friendly)
# ---------------------------------------------------------------------------


class FolderInfo:
    __slots__ = ("id", "name", "path", "store_name", "total_count")

    def __init__(self, *, id: str, name: str, store_name: str, path: str, total_count: int) -> None:
        self.id = id
        self.name = name
        self.store_name = store_name
        self.path = path
        self.total_count = total_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "displayName": f"{self.store_name} / {self.path}",
            "store": self.store_name,
            "path": self.path,
            "totalItemCount": self.total_count,
        }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OutlookClient:
    """Connection to the running (or auto-launched) Classic Outlook instance.

    Use as a context manager so COM is uninitialized cleanly:
        with OutlookClient() as oc:
            for msg in oc.iter_messages(start, end): ...
    """

    def __init__(self, *, max_attachment_bytes: int = 25 * 1024 * 1024) -> None:
        self._max_attachment_bytes = max_attachment_bytes
        self._app: Any | None = None
        self._ns: Any | None = None
        self._co_initialized = False
        self._tempdir: tempfile.TemporaryDirectory | None = None
        self._connect()

    # ---------- lifecycle ----------

    def _connect(self) -> None:
        try:
            import pythoncom
            import win32com.client
        except ImportError as exc:
            raise OutlookUnavailableError(
                "pywin32 is not installed. Outlook COM is Windows-only."
            ) from exc

        try:
            pythoncom.CoInitialize()
            self._co_initialized = True
        except Exception as exc:  # pragma: no cover
            raise OutlookUnavailableError(f"CoInitialize failed: {exc}") from exc

        try:
            self._app = win32com.client.Dispatch("Outlook.Application")
            self._ns = self._app.GetNamespace("MAPI")
        except Exception as exc:
            raise OutlookUnavailableError(
                "Could not connect to Outlook.Application. "
                "Is Classic Outlook installed and runnable on this machine?"
            ) from exc

        self._tempdir = tempfile.TemporaryDirectory(prefix="emailsearch-att-")

    def close(self) -> None:
        # Drop COM references before CoUninitialize; pywin32 reference-counts.
        self._ns = None
        self._app = None
        if self._tempdir is not None:
            with suppress(Exception):
                self._tempdir.cleanup()
            self._tempdir = None
        if self._co_initialized:
            try:
                import pythoncom

                pythoncom.CoUninitialize()
            except Exception:  # pragma: no cover
                pass
            self._co_initialized = False

    def __enter__(self) -> OutlookClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---------- lightweight probes ----------

    def current_user_display(self) -> str | None:
        """Return a human-readable name for the connected mailbox owner.

        Used by the status probe — does NOT walk folders or touch messages.
        Falls back to the primary store's display name if `CurrentUser.Name`
        isn't populated.
        """
        try:
            assert self._ns is not None
            name = str(self._ns.CurrentUser.Name)
            if name and name.strip():
                return name
        except Exception:
            pass
        try:
            assert self._ns is not None
            stores = self._ns.Stores
            if int(stores.Count) >= 1:
                return str(stores.Item(1).DisplayName)
        except Exception:
            pass
        return None

    def trigger_send_receive(self) -> tuple[bool, str]:
        """Kick off Outlook's Send/Receive on all accounts.

        Fire-and-forget — Outlook handles the sync in the background. Only
        refreshes items within the account's Cached Exchange Mode window;
        older items not in the local OST cache will NOT be pulled. To get
        truly old emails the user must widen the "Mail to keep offline"
        slider in File → Account Settings.
        """
        try:
            assert self._ns is not None
            sync_objects = self._ns.SyncObjects
            count = int(sync_objects.Count)
            if count == 0:
                return False, "No sync groups configured in Outlook."
            # Item(1) is conventionally the "All Accounts" group.
            sync_objects.Item(1).Start()
            return True, f"Send/Receive started on {count} sync group(s)."
        except Exception as exc:
            log.warning("trigger_send_receive failed: %s", exc)
            return False, f"failed: {exc}"

    def display_email(self, entry_id: str) -> tuple[bool, str]:
        """Open an Outlook item by EntryID in the normal Outlook inspector.

        Replaces the broken ``outlook:<EntryID>`` URL scheme — which is
        not a registered Windows protocol handler and fails with "scheme
        does not have a registered handler" when launched from a browser.
        Routing through COM is the only reliable way to jump to a
        specific item by EntryID on Classic Outlook.

        ``Display()`` is non-blocking — the inspector window pops up
        asynchronously while this call returns. The Outlook window may
        appear behind the browser; the user clicks the taskbar to bring
        it forward (we deliberately don't ``Activate()`` it because that
        can steal focus during long-running flows like a Load job).
        """
        if not entry_id:
            return False, "missing EntryID"
        try:
            assert self._ns is not None
            item = self._ns.GetItemFromID(entry_id)
        except Exception as exc:
            # Most common cause: the message moved folders since indexing,
            # which invalidates the EntryID. We surface this verbatim so
            # the user knows to re-load.
            log.warning("GetItemFromID(%r) failed: %s", entry_id, exc)
            return False, f"could not find message in Outlook: {exc}"
        try:
            item.Display()
        except Exception as exc:
            log.warning("Display() failed for EntryID %r: %s", entry_id, exc)
            return False, f"failed to open: {exc}"
        return True, "opened in Outlook"

    # ---------- folder discovery ----------

    def list_folders(self) -> list[FolderInfo]:
        """Walk every mail-typed folder across every mounted store.

        Skips non-mail folders (Calendar, Contacts, Tasks, Notes, Journal)
        by checking ``DefaultItemType == 0`` (olMailItem).
        """
        assert self._ns is not None
        out: list[FolderInfo] = []
        for store in self._ns.Stores:
            try:
                root = store.GetRootFolder()
            except Exception as exc:  # pragma: no cover
                log.warning("skipping store %s: %s", store, exc)
                continue
            store_name = store.DisplayName
            self._walk_folders(root, store_name, "", out)
        return out

    def _walk_folders(
        self, folder: Any, store_name: str, parent_path: str, out: list[FolderInfo]
    ) -> None:
        try:
            kind = folder.DefaultItemType
        except Exception:
            kind = -1
        path = f"{parent_path}/{folder.Name}" if parent_path else folder.Name
        if kind == OL_MAIL_ITEM_TYPE:
            try:
                count = int(folder.Items.Count)
            except Exception:
                count = 0
            out.append(
                FolderInfo(
                    id=folder.EntryID,
                    name=folder.Name,
                    store_name=store_name,
                    path=path,
                    total_count=count,
                )
            )
        # Recurse regardless — Calendar etc. can have mail-typed sub-folders rarely.
        try:
            children = list(folder.Folders)
        except Exception:
            children = []
        for child in children:
            self._walk_folders(child, store_name, path, out)

    # ---------- message iteration ----------

    def iter_messages(
        self,
        *,
        start: datetime,
        end: datetime,
        folder_ids: list[str] | None = None,
    ) -> Iterator[RawMessage]:
        """Yield messages in [start, end) from the given folders (or all mail folders).

        Restricts at the Outlook layer using DASL syntax — much faster than
        Python-side filtering on a large mailbox.
        """
        assert self._ns is not None
        start_utc = to_utc(start)
        end_utc = to_utc(end)

        if folder_ids:
            folders = []
            for fid in folder_ids:
                try:
                    folders.append(self._ns.GetFolderFromID(fid))
                except Exception as exc:
                    log.warning("could not open folder %s: %s", fid, exc)
        else:
            folders = []
            for fi in self.list_folders():
                with suppress(Exception):
                    folders.append(self._ns.GetFolderFromID(fi.id))

        for folder in folders:
            try:
                store_name = folder.Store.DisplayName
            except Exception:
                store_name = ""

            try:
                items = folder.Items
                items.Sort("[ReceivedTime]", True)  # newest first
                # DASL on ``urn:schemas:httpmail:datereceived`` is the most
                # robust filter across Outlook builds. The property is
                # stored UTC but Outlook compares in the user's locale; for
                # hourly granularity that's accurate enough.
                restrict = (
                    f"@SQL=\"urn:schemas:httpmail:datereceived\" >= '{_to_dasl(start_utc)}' AND "
                    f"\"urn:schemas:httpmail:datereceived\" < '{_to_dasl(end_utc)}'"
                )
                filtered = items.Restrict(restrict)
            except Exception as exc:
                log.warning("could not restrict folder %s: %s — falling back to scan", folder, exc)
                filtered = folder.Items

            try:
                count = int(filtered.Count)
            except Exception:
                count = 0
            for i in range(1, count + 1):
                try:
                    item = filtered.Item(i)
                except Exception:
                    continue
                # Skip non-mail (meeting requests, etc.)
                try:
                    if int(item.Class) != OL_OBJECT_CLASS_MAIL:
                        continue
                except Exception:
                    continue
                try:
                    raw = self._mail_to_raw(item, folder, store_name)
                except Exception as exc:
                    log.warning("failed to convert mail item: %s", exc)
                    continue
                if raw is None:
                    continue
                # Defensive: some Outlook builds return items at the boundary;
                # enforce the window explicitly.
                if not (start_utc.timestamp() <= raw.received_at < end_utc.timestamp()):
                    continue
                yield raw

    # ---------- internals ----------

    def _mail_to_raw(self, item: Any, folder: Any, store_name: str) -> RawMessage | None:
        msg_id = self._stable_message_id(item)
        if not msg_id:
            return None

        received = _com_dt_to_epoch(_safe(lambda: item.ReceivedTime))
        if received is None:
            return None
        sent = _com_dt_to_epoch(_safe(lambda: item.SentOn))

        from_addr = _safe(lambda: item.SenderEmailAddress) or ""
        from_name = _safe(lambda: item.SenderName) or None
        # Exchange returns an X.500 DN for internal senders. Try to upgrade
        # to SMTP via the AddressEntry property accessor.
        if from_addr and "@" not in from_addr:
            smtp = self._sender_smtp(item)
            if smtp:
                from_addr = smtp

        subject = _safe(lambda: item.Subject) or ""
        body_html = _safe(lambda: item.HTMLBody) or ""
        body_text = _safe(lambda: item.Body) or ""
        body_preview = body_text[:255] if body_text else ""
        conversation_id = _safe(lambda: item.ConversationID)
        entry_id = _safe(lambda: item.EntryID) or ""

        to_list = self._recipients(item, OL_RECIPIENT_TO)
        cc_list = self._recipients(item, OL_RECIPIENT_CC)

        attachments: list[RawAttachment] = []
        try:
            atts_collection = item.Attachments
            n_atts = int(atts_collection.Count)
        except Exception:
            n_atts = 0
        for j in range(1, n_atts + 1):
            try:
                a = atts_collection.Item(j)
            except Exception:
                continue
            ra = self._attachment_to_raw(a)
            if ra is not None:
                attachments.append(ra)

        try:
            folder_id = folder.EntryID
            folder_name = f"{store_name} / {folder.Name}" if store_name else folder.Name
        except Exception:
            folder_id, folder_name = None, None

        web_link = f"outlook:{entry_id}" if entry_id else None

        return RawMessage(
            id=msg_id,
            subject=subject,
            from_address=from_addr,
            from_name=from_name,
            to=to_list,
            cc=cc_list,
            received_at=received,
            sent_at=sent,
            body_html=body_html,
            body_text=body_text,
            body_preview=body_preview,
            conversation_id=conversation_id,
            folder_id=folder_id,
            folder_name=folder_name,
            web_link=web_link,
            has_attachments=bool(attachments),
            attachments=attachments,
        )

    @staticmethod
    def _stable_message_id(item: Any) -> str | None:
        """Internet Message-Id (RFC822) is stable across folder moves; EntryID isn't."""
        try:
            mid = item.PropertyAccessor.GetProperty(PR_INTERNET_MESSAGE_ID)
            if mid:
                return str(mid)
        except Exception:
            pass
        try:
            return str(item.EntryID)
        except Exception:
            return None

    @staticmethod
    def _sender_smtp(item: Any) -> str | None:
        try:
            entry = item.Sender
            if entry is None:
                return None
            try:
                v = entry.PropertyAccessor.GetProperty(PR_SMTP_ADDRESS)
                if v:
                    return str(v)
            except Exception:
                pass
            # Try GetExchangeUser SMTP
            try:
                eu = entry.GetExchangeUser()
                if eu is not None:
                    return str(eu.PrimarySmtpAddress)
            except Exception:
                pass
        except Exception:
            return None
        return None

    @staticmethod
    def _recipients(item: Any, kind: int) -> list[tuple[str, str | None]]:
        out: list[tuple[str, str | None]] = []
        try:
            recips = item.Recipients
            n = int(recips.Count)
        except Exception:
            return out
        for i in range(1, n + 1):
            try:
                r = recips.Item(i)
                if int(r.Type) != kind:
                    continue
                addr = _safe(lambda r=r: r.Address) or ""
                name = _safe(lambda r=r: r.Name) or None
                if addr and "@" not in addr:
                    # Try SMTP upgrade
                    try:
                        v = r.PropertyAccessor.GetProperty(PR_SMTP_ADDRESS)
                        if v:
                            addr = str(v)
                    except Exception:
                        pass
                if addr:
                    out.append((addr, name))
            except Exception:
                continue
        return out

    def _attachment_to_raw(self, a: Any) -> RawAttachment | None:
        try:
            att_name = _safe(lambda: a.FileName) or _safe(lambda: a.DisplayName) or "unnamed"
            size = int(_safe(lambda: a.Size) or 0)
            att_type = int(_safe(lambda: a.Type) or 0)  # 1=byval, 5=embedded item, 6=ole
        except Exception:
            return None

        # Per-attachment unique id within the parent message. Outlook's
        # `Attachment.Index` is 1-based and unique within a single message
        # (unlike e.g. PR_INTERNET_MESSAGE_ID which is a *message* property
        # and identical for every attachment in the same message).
        try:
            att_idx = int(_safe(lambda: a.Index) or 0)
            parent_entry_id = _safe(lambda: a.Parent.EntryID) or ""
            att_id = f"{parent_entry_id}::{att_idx}" if parent_entry_id else f"att::{att_idx}"
        except Exception:
            att_id = att_name

        content_id = _safe(lambda: a.PropertyAccessor.GetProperty(PR_ATTACH_CONTENT_ID))
        is_inline = bool(content_id)

        # Cheap content-type guess from extension. Outlook doesn't expose MIME
        # type on the high-level API; the property accessor sometimes does, but
        # extension-based is good enough for the extractor registry.
        ext = os.path.splitext(att_name)[1].lower()
        content_type = _CONTENT_TYPE_BY_EXT.get(ext, "application/octet-stream")

        ra = RawAttachment(
            att_id=att_id,
            name=att_name,
            content_type=content_type,
            size=size,
            is_inline=is_inline,
            content_id=content_id,
        )

        # Skip embedded item-attachments (forwarded emails as attachments).
        if att_type == 5:
            ra.skipped_reason = "unsupported_type:embedded_item"
            return ra

        if size > self._max_attachment_bytes:
            ra.skipped_reason = f"too_large:{size}>{self._max_attachment_bytes}"
            return ra

        # Save to a temp file then read bytes. SaveAsFile is the most
        # reliable path for arbitrary attachment types.
        try:
            assert self._tempdir is not None
            fd, path = tempfile.mkstemp(prefix="att-", suffix=ext or ".bin", dir=self._tempdir.name)
            os.close(fd)
            a.SaveAsFile(path)
            with open(path, "rb") as f:
                ra.content_bytes = f.read()
            with suppress(OSError):
                os.unlink(path)
        except Exception as exc:
            log.warning("attachment SaveAsFile failed for %s: %s", att_name, exc)
            ra.skipped_reason = "save_failed"
            ra.content_bytes = None
        return ra


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_CONTENT_TYPE_BY_EXT: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".doc": "application/msword",
    ".xls": "application/vnd.ms-excel",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".html": "text/html",
    ".htm": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".heic": "image/heic",
}


def _to_dasl(dt: datetime) -> str:
    """Outlook DASL date literal: 'YYYY-MM-DD HH:MM' in local time.

    The ``urn:schemas:httpmail:datereceived`` property is stored UTC but
    Outlook converts to local for the comparison. Hourly granularity is
    accurate enough for our "all messages in [start,end)" use case.
    """
    return dt.strftime("%Y-%m-%d %H:%M")


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


def _com_dt_to_epoch(dt: Any) -> int | None:
    """Convert a pywin32 datetime (or naive datetime) to UTC unix seconds."""
    if dt is None:
        return None
    try:
        # pywin32's PyTime is convertible via int() to a Unix timestamp on modern builds,
        # but the safer path is to grab year/month/.../second and reconstruct.
        # The COM datetime arrives in the user's local timezone with no tzinfo —
        # attach the local zone, then convert to UTC.
        py = datetime(
            int(dt.year),
            int(dt.month),
            int(dt.day),
            int(dt.hour),
            int(dt.minute),
            int(dt.second),
        )
        local = py.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return int(local.astimezone(UTC).timestamp())
    except Exception:
        # Fall back: hope int() works on PyTime.
        try:
            return int(dt)
        except Exception:
            return None
