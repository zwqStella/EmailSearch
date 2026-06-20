"""CLI entrypoint: `python -m emailsearch serve` etc."""

from __future__ import annotations

# Workaround for a Python + LlamaIndex shutdown race:
# `llama_index.core.ingestion.pipeline` does `from concurrent.futures import
# ProcessPoolExecutor`, which lazily imports `concurrent.futures.process`,
# which tries to register an `atexit` handler. If that import happens FOR THE
# FIRST TIME during interpreter shutdown (e.g. via a module finalizer), the
# atexit registration fails with `RuntimeError: can't register atexit after
# shutdown` and pollutes the Ctrl+C output. Pre-importing here registers the
# handler early — subsequent imports are no-ops.
import concurrent.futures.process  # noqa: F401
import signal
import sys

import typer
import uvicorn

from emailsearch.config import get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="Local Outlook email search.",
    add_completion=False,
)


def _serve_with_graceful_shutdown(config: uvicorn.Config) -> None:
    """Run uvicorn with custom signal handling that flips the cancel flag on
    all active sync jobs the moment Ctrl+C is pressed — *before* uvicorn
    starts tearing down the event loop.

    Without this, uvicorn's own SIGINT handler cancels the ASGI lifespan task
    before sending it the "shutdown" message, so our `_lifespan` cleanup
    never runs and the user sees a `KeyboardInterrupt` + `CancelledError`
    traceback.

    We also suppress `install_signal_handlers` so uvicorn doesn't fight ours,
    and swallow any leftover `KeyboardInterrupt` from `Server.run()` so the
    CLI exits with a clean status.
    """
    server = uvicorn.Server(config)
    # Tell uvicorn not to install its own SIGINT/SIGTERM handlers — ours below
    # already drive `server.should_exit`, which is the same lever uvicorn uses.
    server.install_signal_handlers = lambda: None  # type: ignore[attr-defined]

    interrupt_count = {"n": 0}

    def _handle_shutdown(signum: int, _frame: object | None) -> None:
        interrupt_count["n"] += 1
        if interrupt_count["n"] == 1:
            # First Ctrl+C: cooperatively cancel every active job + ask
            # uvicorn to shut down cleanly. Wrapped in a bare try/except
            # because signal handlers must never raise.
            try:
                from emailsearch.sync.jobs import get_registry

                cancelled = get_registry().request_cancel_all_active()
                if cancelled:
                    typer.secho(
                        f"\nCancelling {len(cancelled)} active job(s) "
                        "(press Ctrl+C again to force exit)...",
                        fg=typer.colors.YELLOW,
                        err=True,
                    )
            except Exception:
                pass
            server.should_exit = True
        else:
            # Second Ctrl+C: hard exit. Don't even attempt cleanup; the user
            # has told us twice they want out.
            server.force_exit = True
            # And in case uvicorn is genuinely stuck, fall through to a hard
            # process exit on a 3rd interrupt.
            if interrupt_count["n"] >= 3:
                sys.exit(130)

    signal.signal(signal.SIGINT, _handle_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_shutdown)
    if hasattr(signal, "SIGBREAK"):
        # Windows-only: produced by CTRL_BREAK_EVENT (e.g. cross-process
        # shutdown from a parent launcher). Real Ctrl+C in a console produces
        # SIGINT instead — both should be handled the same way.
        signal.signal(signal.SIGBREAK, _handle_shutdown)

    try:
        server.run()
    except KeyboardInterrupt:
        # Our handler already initiated graceful shutdown; just exit silently.
        pass


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind host (default: from .env / 127.0.0.1)."),
    port: int | None = typer.Option(None, help="Bind port (default: from .env / 8765)."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)."),
    open_browser: bool = typer.Option(False, "--open-browser", help="Open browser on start."),
) -> None:
    """Start the local web server."""
    settings = get_settings()
    h = host or settings.host
    p = port or settings.port

    if open_browser and not reload:
        import threading
        import webbrowser

        def _open() -> None:
            import time

            time.sleep(1.0)
            webbrowser.open(f"http://{h}:{p}/")

        threading.Thread(target=_open, daemon=True).start()

    config = uvicorn.Config(
        "emailsearch.web.app:create_app",
        host=h,
        port=p,
        reload=reload,
        factory=True,
    )
    if reload:
        # `Server.run()` doesn't support reload — that's only available via
        # `uvicorn.run()`'s subprocess supervisor. Fall back when the user
        # asked for dev reload; the graceful-shutdown niceties don't matter
        # in a dev loop anyway.
        uvicorn.run(
            "emailsearch.web.app:create_app",
            host=h,
            port=p,
            reload=True,
            factory=True,
        )
    else:
        _serve_with_graceful_shutdown(config)


@app.command()
def info() -> None:
    """Print resolved configuration paths."""
    s = get_settings()
    typer.echo(f"data_dir:    {s.data_dir}")
    typer.echo(f"db_path:     {s.resolved_db_path}")
    typer.echo(f"models_dir:  {s.models_cache_dir}")
    typer.echo(f"embed_model: {s.embed_model} (dim={s.embed_dim})")
    typer.echo(f"ocr_enabled: {s.ocr_enabled}")
    typer.echo(f"max_attachment_mb: {s.max_attachment_mb}")
    typer.echo(f"server:      http://{s.host}:{s.port}")


if __name__ == "__main__":  # pragma: no cover
    app()
