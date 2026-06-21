"""Status route: backend health for the Settings tab."""

from __future__ import annotations

import logging
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from emailsearch.config import get_settings
from emailsearch.db.connection import connect
from emailsearch.db.repositories import get_email

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["status"])


# Custom URL scheme stored in ``EmailRow.web_link`` for COM-loaded
# messages. The scheme is NOT registered with Windows (browsers report
# "scheme does not have a registered handler"), so we strip the prefix
# and drive Outlook directly via COM through :func:`open_email_in_outlook`.
_COM_LINK_PREFIX = "outlook:"


class OutlookStatus(BaseModel):
    available: bool
    detail: str  # human-readable explanation
    backend: str = "outlook-com"


@router.get("/outlook/status", response_model=OutlookStatus)
def outlook_status() -> OutlookStatus:
    """Check whether we can talk to local Classic Outlook via COM.

    Lightweight: opens a COM session, reads the user name, closes. Does
    NOT walk folders — folder enumeration only happens when the user
    explicitly asks for it on the Load tab.
    """
    if sys.platform != "win32":
        return OutlookStatus(
            available=False,
            detail="Outlook COM is Windows-only.",
        )
    try:
        from emailsearch.outlook.com_client import OutlookClient, OutlookUnavailableError
    except Exception as exc:
        return OutlookStatus(available=False, detail=f"import failed: {exc}")

    try:
        with OutlookClient() as client:
            # Touching the namespace's current user is enough to confirm COM
            # is responsive — no folder walk, no message iteration.
            who = client.current_user_display() or "(unknown)"
        return OutlookStatus(available=True, detail=f"Connected as {who}.")
    except OutlookUnavailableError as exc:
        return OutlookStatus(available=False, detail=str(exc))
    except Exception as exc:
        log.exception("outlook status probe failed")
        return OutlookStatus(available=False, detail=f"probe failed: {exc}")


class SyncResponse(BaseModel):
    ok: bool
    detail: str


@router.post("/outlook/sync", response_model=SyncResponse)
def outlook_sync() -> SyncResponse:
    """Trigger Outlook's Send/Receive on all accounts.

    Asks Outlook to pull fresh items within its Cached Exchange Mode
    window. This will NOT retrieve items older than the cache slider —
    the user must widen that in Outlook's account settings.
    """
    if sys.platform != "win32":
        return SyncResponse(ok=False, detail="Outlook COM is Windows-only.")
    try:
        from emailsearch.outlook.com_client import OutlookClient, OutlookUnavailableError
    except Exception as exc:
        return SyncResponse(ok=False, detail=f"import failed: {exc}")
    try:
        with OutlookClient() as client:
            ok, detail = client.trigger_send_receive()
        return SyncResponse(ok=ok, detail=detail)
    except OutlookUnavailableError as exc:
        return SyncResponse(ok=False, detail=str(exc))
    except Exception as exc:
        log.exception("outlook sync trigger failed")
        return SyncResponse(ok=False, detail=f"failed: {exc}")


class OpenEmailResponse(BaseModel):
    ok: bool
    detail: str


@router.post("/outlook/open/{email_id}", response_model=OpenEmailResponse)
def open_email_in_outlook(email_id: str) -> OpenEmailResponse:
    """Open an indexed email in the local Outlook app.

    Resolves the email's stored EntryID (kept in ``EmailRow.web_link``
    under the ``outlook:<EntryID>`` prefix by the COM loader) and asks
    Outlook to display it. Replaces the broken ``outlook:<EntryID>`` URL
    scheme used in the previous version — that scheme is NOT registered
    with Windows and fails browser-side with "scheme does not have a
    registered handler".

    The 404 path covers both "no such email" and "indexed via a non-COM
    backend (no EntryID)" — neither is recoverable from the UI, so the
    frontend just surfaces the detail string verbatim.
    """
    with connect(get_settings().resolved_db_path) as conn:
        email = get_email(conn, email_id)
    if email is None:
        raise HTTPException(status_code=404, detail="email not found")
    link = email.web_link or ""
    if not link.startswith(_COM_LINK_PREFIX):
        # Not a COM-loaded message — we can't drive Outlook to it. The
        # frontend already opens http(s) ``web_link`` URLs directly so
        # this path only fires for malformed / missing links.
        raise HTTPException(
            status_code=404,
            detail="no Outlook EntryID stored for this email",
        )
    entry_id = link[len(_COM_LINK_PREFIX):]
    if sys.platform != "win32":
        return OpenEmailResponse(ok=False, detail="Outlook COM is Windows-only.")
    try:
        from emailsearch.outlook.com_client import OutlookClient, OutlookUnavailableError
    except Exception as exc:
        return OpenEmailResponse(ok=False, detail=f"import failed: {exc}")
    try:
        with OutlookClient() as client:
            ok, detail = client.display_email(entry_id)
        return OpenEmailResponse(ok=ok, detail=detail)
    except OutlookUnavailableError as exc:
        return OpenEmailResponse(ok=False, detail=str(exc))
    except Exception as exc:
        log.exception("outlook open failed for email_id=%s", email_id)
        return OpenEmailResponse(ok=False, detail=f"failed: {exc}")
