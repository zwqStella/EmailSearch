r"""Preview a few LLM-generated summaries from the live index.

Run:
    .\.venv\Scripts\python.exe scripts\preview_summaries.py
    .\.venv\Scripts\python.exe scripts\preview_summaries.py --limit 20
    .\.venv\Scripts\python.exe scripts\preview_summaries.py --like alpha
"""

from __future__ import annotations

import argparse

from emailsearch.db.connection import connect


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=5, help="rows to print (default 5)")
    p.add_argument(
        "--like",
        default=None,
        help="optional substring filter on subject OR summary",
    )
    p.add_argument(
        "--width", type=int, default=200, help="max chars of summary to print"
    )
    args = p.parse_args()

    sql = (
        "SELECT id, subject, from_address, summary "
        "FROM emails "
        "WHERE summary IS NOT NULL AND summary != ''"
    )
    params: tuple = ()
    if args.like:
        sql += " AND (subject LIKE ? OR summary LIKE ?)"
        params = (f"%{args.like}%", f"%{args.like}%")
    sql += " ORDER BY received_at DESC LIMIT ?"
    params = (*params, args.limit)

    with connect() as conn:
        total_emails = conn.execute(
            "SELECT COUNT(*) AS n FROM emails"
        ).fetchone()["n"]
        with_summary = conn.execute(
            "SELECT COUNT(*) AS n FROM emails "
            "WHERE summary IS NOT NULL AND summary != ''"
        ).fetchone()["n"]
        rows = conn.execute(sql, params).fetchall()

    # Diagnostic line so an empty result tells you WHICH state you're in:
    #   - "0 / 0"     -> no emails indexed at all
    #   - "0 / N>0"   -> loader ran but didn't summarize (LLM was off / failed)
    #   - "M / N"     -> some emails have summaries; tweak --like / --limit
    print(
        f"# {with_summary}/{total_emails} email(s) have a summary; showing {len(rows)}"
    )
    if with_summary == 0:
        if total_emails == 0:
            print("# No emails indexed. Run a Load job from the UI first.")
        else:
            from emailsearch.config import get_settings

            s = get_settings()
            print("# No summaries stored. Common causes:")
            print(f"#   - llm_enabled is {s.llm_enabled} (must be True at LOAD time)")
            print(f"#   - llm_base_url={s.llm_base_url} — was the model server up?")
            print("#   - emails were loaded BEFORE you turned llm_enabled on.")
            print("#     Run scripts/backfill_summaries.py to fill them in without")
            print("#     re-loading from Outlook.")

    for r in rows:
        summary = r["summary"]
        if len(summary) > args.width:
            summary = summary[: args.width] + "…"
        print()
        print(f"-- {r['subject']!r}")
        print(f"   from: {r['from_address']}")
        print(f"   id:   {r['id']}")
        print(f"   summary: {summary}")


if __name__ == "__main__":
    main()
