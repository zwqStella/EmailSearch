"""Shared test fixtures.

Process-wide caches (LLM result memos, embedding memo, connection pool)
must be reset between tests so a value stashed by one case doesn't leak
into a later one that uses different mocks or DB paths.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_process_caches() -> None:
    """Clear every process-lifetime cache between tests.

    Tests routinely monkeypatch ``urllib.request.urlopen`` and
    ``get_settings`` differently per case. Without this fixture, a
    cached LLM response from an earlier test would skip the patched
    transport on later calls, masking real coverage gaps.

    Imports are local so a test that exercises a subset of modules
    doesn't force-load the others.
    """
    from emailsearch.ask import parser as ask_parser
    from emailsearch.db import connection as db_conn
    from emailsearch.embed import encoder as embed_encoder
    from emailsearch.summarize import client as llm_client

    llm_client.clear_query_caches()
    ask_parser.clear_parse_cache()
    embed_encoder.clear_embed_query_cache()
    db_conn.clear_connection_pools()
    db_conn.reset_schema_cache()
    yield
    # Post-test cleanup so the next test starts clean even if it
    # bypassed this fixture's setup phase.
    llm_client.clear_query_caches()
    ask_parser.clear_parse_cache()
    embed_encoder.clear_embed_query_cache()
    db_conn.clear_connection_pools()
    db_conn.reset_schema_cache()
