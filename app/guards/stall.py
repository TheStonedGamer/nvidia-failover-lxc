"""Stream-stall watchdog: a reasoning model can go silent for a long time while
"thinking" before emitting visible output, so the timeout must be generous and
reset on reasoning heartbeats, not just visible content."""

from app.config import _STREAM_STALL_S

STREAM_STALL_S = _STREAM_STALL_S


class StreamStall(Exception):
    """Raised when a stream has committed content but gone idle too long."""
