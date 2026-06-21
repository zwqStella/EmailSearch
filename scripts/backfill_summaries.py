r"""Backfill LLM-generated summaries onto existing emails in the index.

Usage:
    .\.venv\Scripts\python.exe scripts\backfill_summaries.py
    .\.venv\Scripts\python.exe scripts\backfill_summaries.py --limit 50
    .\.venv\Scripts\python.exe scripts\backfill_summaries.py --dry-run
    .\.venv\Scripts\python.exe scripts\backfill_summaries.py --force
    .\.venv\Scripts\python.exe scripts\backfill_summaries.py --rechunk-only

Behaviour
---------
* By default, iterates emails with NULL summary, calls `summarize_email` (which
  now reads body + subject + sender + every attachment's extracted text under a
  bounded character budget), and writes the result via `set_email_summary`.
* Each row commits independently — Ctrl+C between calls loses at most the
  in-flight row. Re-running the script picks up where it left off.
* Progress is logged to stdout with rate + ETA so you can plan around the run.
* `--dry-run` runs the LLM but does NOT write to the DB; useful for sanity-
  checking the model's output before committing.
* `--force` re-summarizes emails that ALREADY have a summary. The default skips
  them.
* `--limit N` stops after N successful summaries — handy for verifying the
  pipeline works on a small batch before kicking off a full corpus run.
* `--rechunk-only` SKIPS the LLM entirely and just re-runs `set_email_summary`
  with the email's existing stored summary. Use this after upgrading the
  schema/code so summary text already in `emails.summary` gets embedded into
  `vec_email_chunks` as `source_type='summary'` rows. Cheap (one embed per
  email, no LLM round-trip) and idempotent. Implies `--force` semantics over
  the candidate set (everything with a non-NULL summary).

Sanity checks before the loop
-----------------------------
* `EMAILSEARCH_LLM_ENABLED` must be true (skipped when `--rechunk-only`).
* The configured `llm_base_url` must respond to `/models` within 5s (also
  skipped when `--rechunk-only`).
Failures here are fatal — better to abort upfront than burn through 8k emails
calling a dead endpoint and storing thousands of NULL summaries.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
import urllib.error
import urllib.request

from emailsearch.config import get_settings
from emailsearch.db.connection import connect
from emailsearch.db.repositories import set_email_summary
from emailsearch.summarize import summarize_email


def _probe(base_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Confirm the LLM endpoint is reachable before starting the loop."""
    url = base_url.rstrip("/") + "/models"
    try:
        # nosec B310 — URL is operator-controlled config.
        with urllib.request.urlopen(url, timeout=timeout):  # noqa: S310
            return True, "ok"
    except urllib.error.URLError as exc:
        return False, f"URLError: {exc.reason}"
    except (TimeoutError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _fmt_eta(seconds: float) -> str:
    """Render a duration as ``HhMmSs`` skipping the leading zeros."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=0, help="stop after N successful summaries (0 = no limit)")
    p.add_argument("--dry-run", action="store_true", help="call the LLM but don't write to the DB")
    p.add_argument("--force", action="store_true", help="re-summarize emails that already have a summary")
    p.add_argument(
        "--rechunk-only",
        action="store_true",
        help=(
            "skip the LLM and just re-embed every existing summary as "
            "vec_email_chunks rows. Use after upgrading the schema/code."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="log every prompt/response (INFO level on summarize.client)")
    args = p.parse_args()

    # Default to a single concise line per email; --verbose unlocks the
    # prompt/response logging inside the summarize client.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.verbose:
        logging.getLogger("emailsearch.summarize.client").setLevel(logging.INFO)

    settings = get_settings()
    # --rechunk-only doesn't call the LLM at all, so the enabled/reachability
    # checks would be misleading noise. Skip them entirely.
    if not args.rechunk_only:
        if not settings.llm_enabled:
            print(
                "EMAILSEARCH_LLM_ENABLED is false. Set it to true and ensure "
                "the local LLM endpoint is running before running the backfill.",
                file=sys.stderr,
            )
            return 2

        print(f"# llm_base_url={settings.llm_base_url}  model={settings.llm_model}")
        print(f"# probing {settings.llm_base_url}/models ...")
        ok, detail = _probe(settings.llm_base_url)
        if not ok:
            print(f"# LLM endpoint unreachable: {detail}", file=sys.stderr)
            print("# Start the model server (or fix EMAILSEARCH_LLM_BASE_URL) and retry.", file=sys.stderr)
            return 2
        print("# endpoint reachable; starting backfill\n")
    else:
        print("# --rechunk-only: skipping LLM (re-embedding existing summaries only)\n")

    # Wire up cooperative cancel so Ctrl+C drains in-flight work cleanly.
    cancel_requested = {"flag": False}

    def _on_sigint(signum: int, _frame: object | None) -> None:
        if cancel_requested["flag"]:
            # Second Ctrl+C — get out NOW.
            print("\n# second SIGINT — exiting immediately", file=sys.stderr)
            sys.exit(130)
        cancel_requested["flag"] = True
        print("\n# SIGINT received — finishing in-flight call, then exiting (Ctrl+C again to force)", file=sys.stderr)

    signal.signal(signal.SIGINT, _on_sigint)

    with connect(settings.resolved_db_path) as conn:
        # Build the candidate list.
        if args.rechunk_only:
            # Re-embed existing summaries only — NULL/empty summaries are
            # skipped because there's nothing to chunk.
            id_sql = (
                "SELECT id FROM emails "
                "WHERE summary IS NOT NULL AND summary != '' "
                "ORDER BY received_at DESC"
            )
        elif args.force:
            id_sql = "SELECT id FROM emails ORDER BY received_at DESC"
        else:
            id_sql = (
                "SELECT id FROM emails "
                "WHERE summary IS NULL OR summary = '' "
                "ORDER BY received_at DESC"
            )
        rows = conn.execute(id_sql).fetchall()
        total_candidates = len(rows)
        label = "re-chunk" if args.rechunk_only else "summarize"
        print(f"# {total_candidates} email(s) need {label}")
        if total_candidates == 0:
            return 0

        ok_count = 0
        fail_count = 0
        skip_count = 0
        started_at = time.time()
        # Print throughput every N rows; smaller batches when --limit is set
        # so the user sees something quickly.
        progress_every = max(1, min(20, args.limit) if args.limit else 20)

        # Import here so we can use repositories.get_email without circular
        # gymnastics — and so dry-run mode doesn't even touch get_email if
        # candidates is empty.
        from emailsearch.db.repositories import get_email

        for i, row in enumerate(rows, start=1):
            if cancel_requested["flag"]:
                break
            if args.limit and ok_count >= args.limit:
                break

            eid = row["id"]
            email = get_email(conn, eid)
            if email is None:
                # Vanishingly unlikely race (concurrent delete) — log and
                # continue rather than abort the whole job.
                print(f"  [{i}/{total_candidates}] {eid}: row disappeared, skipping")
                skip_count += 1
                continue

            t0 = time.time()
            if args.rechunk_only:
                # No LLM call — just re-run set_email_summary with the
                # already-stored text so the embed + insert pipeline runs.
                existing = (email.summary or "").strip()
                if not existing:
                    # Defensive: the SELECT filtered these out, but a
                    # concurrent UPDATE could have cleared the column
                    # between the SELECT and this read.
                    skip_count += 1
                    continue
                summary = existing
            else:
                try:
                    summary = summarize_email(email)
                except Exception as exc:  # pragma: no cover — safety net
                    print(f"  [{i}/{total_candidates}] {eid}: UNEXPECTED {type(exc).__name__}: {exc}")
                    fail_count += 1
                    continue
            dt = time.time() - t0

            if summary is None:
                # LLM call failed / no content. The client already logged the
                # reason at WARNING. Don't write NULL — leaves the row open
                # for a future retry.
                fail_count += 1
                preview = "(no summary)"
            else:
                if not args.dry_run:
                    set_email_summary(conn, eid, summary)
                ok_count += 1
                preview = summary[:80].replace("\n", " ")
                if len(summary) > 80:
                    preview += "…"

            subject_preview = (email.subject or "(no subject)")[:40]
            print(f"  [{i}/{total_candidates}] {eid}  ({dt:5.1f}s)  {subject_preview!r:42} -> {preview}")

            if i % progress_every == 0:
                elapsed = time.time() - started_at
                rate = i / elapsed if elapsed else 0
                remaining = total_candidates - i
                eta = remaining / rate if rate else 0
                print(
                    f"  -- {i}/{total_candidates}  ok={ok_count}  fail={fail_count}  "
                    f"skip={skip_count}  rate={rate:.2f}/s  eta={_fmt_eta(eta)}\n"
                )

    elapsed = time.time() - started_at
    print()
    print(f"# done. processed {ok_count + fail_count + skip_count}/{total_candidates}")
    print(f"#   ok={ok_count}  fail={fail_count}  skip={skip_count}  elapsed={_fmt_eta(elapsed)}")
    if args.dry_run:
        print("# (dry-run: nothing was written to the database)")
    if cancel_requested["flag"]:
        print("# (interrupted; re-run the script to resume — already-summarized rows are skipped)")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
