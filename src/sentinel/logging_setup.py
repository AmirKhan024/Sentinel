"""Single place where logging is configured.

Rationale: the ingestion pipeline must be auditable from its logs alone
(which page was requested, how many rows came back, what was retried, where
output landed). `print()` is not used anywhere in `src/sentinel` because it
cannot be levelled, filtered, timestamped, or redirected by the caller.
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for CLI entry points.

    Idempotent: calling twice replaces the handler rather than duplicating it,
    which keeps log output clean when tests or notebooks re-configure.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level.upper())

    # httpx logs every request at INFO, which duplicates our own page logging.
    logging.getLogger("httpx").setLevel(logging.WARNING)
