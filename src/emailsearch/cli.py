"""CLI entrypoint: `python -m emailsearch serve` etc."""

from __future__ import annotations

# Pre-import ``concurrent.futures.process`` so its lazy atexit handler
# registers now rather than during interpreter shutdown — LlamaIndex
# triggers that lazy import inside a shutdown path, which raises
# ``RuntimeError: can't register atexit after shutdown``.
import concurrent.futures.process  # noqa: F401
import signal
import subprocess
import sys
from pathlib import Path

import typer
import uvicorn

from emailsearch.config import get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="Local Outlook email search.",
    add_completion=False,
)


def _serve_with_graceful_shutdown(config: uvicorn.Config) -> None:
    """Run uvicorn with custom signal handling: flip the cancel flag on
    every active sync job BEFORE uvicorn tears down the event loop.

    Without this, uvicorn's SIGINT handler cancels the ASGI lifespan
    task before sending the "shutdown" message, so our ``_lifespan``
    cleanup never runs and the user sees a CancelledError traceback.
    """
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[attr-defined]

    interrupt_count = {"n": 0}

    def _handle_shutdown(signum: int, _frame: object | None) -> None:
        interrupt_count["n"] += 1
        if interrupt_count["n"] == 1:
            # First Ctrl+C: cooperatively cancel jobs + ask uvicorn to shut down.
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
            # Second Ctrl+C: hard exit. Third forces a process exit.
            server.force_exit = True
            if interrupt_count["n"] >= 3:
                sys.exit(130)

    signal.signal(signal.SIGINT, _handle_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_shutdown)
    if hasattr(signal, "SIGBREAK"):
        # Windows-only: produced by CTRL_BREAK_EVENT.
        signal.signal(signal.SIGBREAK, _handle_shutdown)

    try:
        server.run()
    except KeyboardInterrupt:
        # Our handler already initiated graceful shutdown.
        pass


def _open_browser_when_ready(host: str, port: int) -> None:
    """Poll ``/api/health`` until it responds, then open the browser.

    A blind ``time.sleep`` isn't enough — uvicorn's lifespan startup
    preloads the embedding model (multi-second on cold start) and
    doesn't accept requests until that completes. After ``deadline`` we
    open anyway as a best-effort fallback.
    """
    import time
    import urllib.error
    import urllib.request
    import webbrowser

    url = f"http://{host}:{port}/"
    health_url = f"{url}api/health"
    deadline = time.monotonic() + 30.0
    poll_interval_s = 0.25

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as resp:  # noqa: S310
                if 200 <= resp.status < 500:
                    break
        except (urllib.error.URLError, TimeoutError, OSError, ConnectionError):
            pass
        time.sleep(poll_interval_s)

    webbrowser.open(url)


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

        threading.Thread(
            target=_open_browser_when_ready, args=(h, p), daemon=True
        ).start()

    config = uvicorn.Config(
        "emailsearch.web.app:create_app",
        host=h,
        port=p,
        reload=reload,
        factory=True,
    )
    if reload:
        # ``Server.run()`` doesn't support reload — that's only available
        # via ``uvicorn.run()``'s subprocess supervisor. Fall back for dev;
        # graceful shutdown niceties don't matter in a dev loop anyway.
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
def start(
    host: str | None = typer.Option(None, help="Bind host (default: from .env / 127.0.0.1)."),
    port: int | None = typer.Option(None, help="Bind port (default: from .env / 8765)."),
    skip_build: bool = typer.Option(
        False, "--skip-build", help="Skip the `npm run build` step."
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Don't open the browser on start."
    ),
) -> None:
    """Stop any stale server, rebuild the frontend, then start fresh + open browser.

    Convenience one-liner for the dev inner-loop. Use ``serve`` directly
    if you want ``--reload``, want to skip the build, or want to iterate
    with the Vite dev server on :5173.
    """
    settings = get_settings()
    h = host or settings.host
    p = port or settings.port

    _stop_server_on_port(p)

    if not skip_build:
        _build_frontend()

    if not no_browser:
        import threading

        threading.Thread(
            target=_open_browser_when_ready, args=(h, p), daemon=True
        ).start()

    config = uvicorn.Config(
        "emailsearch.web.app:create_app",
        host=h,
        port=p,
        reload=False,
        factory=True,
    )
    _serve_with_graceful_shutdown(config)


def _stop_server_on_port(port: int) -> None:
    """Best-effort: kill the process bound to ``port`` so the new server
    can claim it. Windows-only (uses Get-NetTCPConnection + Stop-Process)
    because this product targets Windows + Classic Outlook COM anyway.
    """
    import socket
    import time

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) != 0:
            return  # nothing listening — fast path

    typer.secho(
        f"Port {port} is in use; stopping the existing server…",
        fg=typer.colors.YELLOW,
    )

    if sys.platform != "win32":
        typer.secho(
            f"  auto-stop is Windows-only — free port {port} manually and retry.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # ``-ErrorAction SilentlyContinue`` so the absent-row case exits 0.
    ps_cmd = (
        f"Get-NetTCPConnection -LocalPort {port} -State Listen "
        f"-ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        typer.secho(f"  could not query owning process: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    pids = {int(line.strip()) for line in out.stdout.splitlines() if line.strip().isdigit()}
    if not pids:
        # Socket said "in use" but Get-NetTCPConnection saw nothing —
        # likely a TIME_WAIT remnant. Don't bail; the wait loop retries.
        typer.secho(
            "  no owning process found — port may be in TIME_WAIT; waiting…",
            fg=typer.colors.YELLOW,
        )
    else:
        for pid in pids:
            typer.secho(f"  stopping PID {pid}", fg=typer.colors.YELLOW)
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"],
                check=False,
                timeout=5,
            )

    # Poll until the socket frees (3s budget — TIME_WAIT can push past 1s).
    for _ in range(15):
        time.sleep(0.2)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
    typer.secho(
        f"  port {port} still in use after stop attempt — aborting.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1)


def _build_frontend() -> None:
    """Run ``npm run build`` in the repo's ``frontend/`` dir, streaming
    output. Assumes ``npm install`` is done. A failing build is fatal —
    starting on top of a stale ``dist/`` is exactly the bug this command
    exists to prevent.
    """
    # cli.py lives at src/emailsearch/cli.py; frontend/ is two parents up.
    frontend = Path(__file__).resolve().parents[2] / "frontend"
    if not (frontend / "package.json").is_file():
        typer.secho(
            f"frontend/ not found at {frontend}; skipping build.",
            fg=typer.colors.YELLOW,
        )
        return

    typer.secho(f"Building frontend in {frontend}…", fg=typer.colors.CYAN)
    # Windows `npm` is a `.cmd` shim; on POSIX it's a plain binary.
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    try:
        result = subprocess.run([npm, "run", "build"], cwd=str(frontend), check=False)
    except FileNotFoundError as e:
        typer.secho(
            f"  `{npm}` not found in PATH — install Node.js or run with --skip-build.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from e
    if result.returncode != 0:
        typer.secho(
            f"  npm run build failed (exit {result.returncode}).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=result.returncode)
    typer.secho("Frontend build complete.", fg=typer.colors.GREEN)


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
