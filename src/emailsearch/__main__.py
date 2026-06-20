"""Allow `python -m emailsearch ...` to invoke the Typer CLI."""

from emailsearch.cli import app

if __name__ == "__main__":  # pragma: no cover
    app()
