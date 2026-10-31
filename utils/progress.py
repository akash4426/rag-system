"""Small, transport-agnostic progress events for long-running pipelines.

Retrieval and evidence stages optionally accept one of these callbacks. The API
turns them into Server-Sent Events, while tests can collect them in a list.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

ProgressCallback = Callable[[dict[str, Any]], None]

logger = logging.getLogger(__name__)


def emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    status: str = "info",
    message: str = "",
    **details: Any,
) -> None:
    """Send one progress event without allowing a listener to break retrieval."""
    if callback is None:
        return
    event = {
        "stage": stage,
        "status": status,
        "message": message,
        "details": details,
        "ts": time.time(),
    }
    try:
        callback(event)
    except Exception as exc:  # progress must never fail the request
        logger.debug("Progress callback failed for stage %s: %s", stage, exc)
