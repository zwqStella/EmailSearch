"""Ask agent: parse a natural-language question, run a single search
tool-call, and stream a grounded answer with inline ``[N]`` citations.

The public surface is intentionally narrow:

  - :func:`ask_question` — the agent generator. Yields :class:`AskEvent`
    objects in a fixed order (``meta`` → ``parsed`` → ``sources`` →
    zero-or-more ``answer_delta`` → ``done``, or a single ``error``).
  - :func:`parse_ask_question` — the single LLM hop that turns the
    question into a :class:`ParsedAskRequest` (query + filters).
  - :class:`ParsedAskRequest` / :class:`AskEvent` — data shapes the
    HTTP layer wraps as NDJSON.

Implementation details (prompt templates, the search-tool façade,
error-routing helpers) are private to the submodules.
"""

from emailsearch.ask.parser import ParsedAskRequest, parse_ask_question
from emailsearch.ask.service import AskEvent, AskEventType, ask_question

__all__ = [
    "AskEvent",
    "AskEventType",
    "ParsedAskRequest",
    "ask_question",
    "parse_ask_question",
]
