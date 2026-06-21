"""Status route: backend health for the Settings tab."""

from __future__ import annotations

import logging
import sys

from fastapi import APIRouter
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["status"])


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
