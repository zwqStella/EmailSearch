"""CLI: ``python -m emailsearch.eval run --queries eval/queries.toml``.

Two subcommands:

- ``run`` — execute the eval set against the live DB, write a markdown
  report (and optionally a JSON dump for downstream diffing).
- ``validate`` — parse the TOML and check every relevance ID actually
  exists in the DB. Catches typos in hand-authored relevance lists
  before they silently zero out the metric.
"""

from __future__ import annotations

import json
import logging
import sys
import tomllib
from pathlib import Path

import typer

from emailsearch.db.connection import apply_schema, open_connection
from emailsearch.eval.report import render
from emailsearch.eval.runner import DEFAULT_MODES, RUN_LIMIT, run_eval
from emailsearch.eval.schema import EvalSet
from emailsearch.search.service import SearchMode

app = typer.Typer(
    no_args_is_help=True,
    help="Search-quality evaluation harness.",
    add_completion=False,
)


def _load_eval_set(path: Path) -> EvalSet:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return EvalSet.model_validate(data)


@app.command()
def run(
    queries: Path = typer.Option(
        Path("eval/queries.toml"),
        help="Path to the query set TOML.",
    ),
    out: Path = typer.Option(
        Path("eval/report.md"),
        help="Where to write the markdown report.",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        help="Optional JSON dump of the full report payload.",
    ),
    modes: str = typer.Option(
        ",".join(DEFAULT_MODES),
        help="Comma-separated modes to evaluate. Subset of keyword,semantic,hybrid.",
    ),
    limit: int = typer.Option(
        RUN_LIMIT,
        help="Per-query retrieval cap fed to search().",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable info-level logs."),
) -> None:
    """Run the eval set and write a markdown report."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not queries.exists():
        typer.secho(f"queries file not found: {queries}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    eval_set = _load_eval_set(queries)
    mode_list = _parse_modes(modes)

    typer.secho(
        f"Running {len(eval_set.queries)} query/ies × {len(mode_list)} mode(s) "
        f"(limit={limit})...",
        fg=typer.colors.CYAN,
    )

    conn = open_connection()
    try:
        apply_schema(conn)
        report = run_eval(conn, eval_set, modes=mode_list, limit=limit)
    finally:
        conn.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(report), encoding="utf-8")
    typer.secho(f"Wrote markdown report → {out}", fg=typer.colors.GREEN)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        typer.secho(f"Wrote JSON payload → {json_out}", fg=typer.colors.GREEN)


@app.command()
def validate(
    queries: Path = typer.Option(
        Path("eval/queries.toml"),
        help="Path to the query set TOML.",
    ),
) -> None:
    """Check that every relevance ID in the query set actually exists in
    the DB. Exits non-zero if any are missing.
    """
    if not queries.exists():
        typer.secho(f"queries file not found: {queries}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    eval_set = _load_eval_set(queries)
    all_ids = {eid for q in eval_set.queries for eid in q.relevant}

    typer.echo(f"queries: {len(eval_set.queries)}, distinct relevance IDs: {len(all_ids)}")
    if not all_ids:
        typer.secho(
            "No relevance IDs in the eval set — nothing to validate.",
            fg=typer.colors.YELLOW,
        )
        return

    conn = open_connection()
    try:
        apply_schema(conn)
        placeholders = ",".join("?" * len(all_ids))
        rows = conn.execute(
            f"SELECT id FROM emails WHERE id IN ({placeholders})", list(all_ids)
        ).fetchall()
    finally:
        conn.close()
    present = {r["id"] for r in rows}
    missing = all_ids - present

    if not missing:
        typer.secho("All relevance IDs resolve in the DB.", fg=typer.colors.GREEN)
        return
    typer.secho(f"{len(missing)} missing ID(s):", fg=typer.colors.RED, err=True)
    for eid in sorted(missing):
        # Show which queries reference each missing ID for fast fix-up.
        owners = [q.id for q in eval_set.queries if eid in q.relevant]
        typer.secho(f"  {eid}  (queries: {', '.join(owners)})", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _parse_modes(s: str) -> list[SearchMode]:
    raw = [m.strip() for m in s.split(",") if m.strip()]
    allowed = {"keyword", "semantic", "hybrid"}
    bad = [m for m in raw if m not in allowed]
    if bad:
        typer.secho(f"unknown mode(s): {bad}; allowed: {sorted(allowed)}", fg=typer.colors.RED)
        sys.exit(2)
    return [m for m in raw]  # type: ignore[return-value]


if __name__ == "__main__":
    app()
