"""Typed exception hierarchy for Python-API consumers.

CLI consumers rely on 0/1/2 exit codes (0 = clean, 1 = findings,
2 = tool error). Python-API consumers rely on ``except SciWriteLintError``.
Keep the hierarchy narrow — only add typed subclasses when a specific
raise site justifies it.
"""

from __future__ import annotations


class SciWriteLintError(Exception):
    """Base for all sciwrite-lint domain errors."""


class LLMConnectionError(SciWriteLintError):
    """vLLM server unreachable — at startup, during batch, or mid-request."""
