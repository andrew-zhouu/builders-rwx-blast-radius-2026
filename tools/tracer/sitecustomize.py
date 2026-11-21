"""Activates the tracer in interpreters spawned by a traced test."""
import os

_dir = os.environ.get("BR_TRACE_SUBDIR")
_root = os.environ.get("BR_TRACE_ROOT")
if _dir and _root and os.path.isdir(_dir):
    try:
        import brtrace

        brtrace.start_subprocess(_dir, _root)
    except Exception:  # never break the traced program
        pass
