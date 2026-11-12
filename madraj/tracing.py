"""LangSmith tracing (Lab 9).

`traceable` records a function's inputs, outputs, timing and errors in
LangSmith when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set.
If LangSmith is not installed, the decorator does nothing, so the
simulation still runs offline (for example during the exhibition).
"""
import os

try:
    from langsmith import traceable as _traceable
except ImportError:  # pragma: no cover
    _traceable = None


def tracing_enabled():
    return bool(_traceable) and os.getenv("LANGSMITH_TRACING", "").lower() == "true" \
        and bool(os.getenv("LANGSMITH_API_KEY"))


def traceable(name=None, run_type="chain"):
    def decorator(fn):
        if _traceable is None:
            return fn
        return _traceable(name=name or fn.__name__, run_type=run_type)(fn)
    return decorator
