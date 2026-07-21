"""Test-only dependency shims shared by conftest.py and mp.spawn workers.

``mp.spawn`` children re-import the spawning test module to unpickle the
worker function, but they never execute pytest's ``conftest.py`` hooks. Any
worker (or module a worker imports) that pulls in the verl import chain must
call :func:`install` BEFORE the first ``verl`` import, otherwise the child
dies on optional-dependency drift (e.g. newer ``transformers`` without
``AutoModelForVision2Seq``) even though the parent process runs fine.

This module is test infrastructure only — production code must never import
it. The leading underscore keeps pytest from collecting it.
"""

from __future__ import annotations


def install() -> None:
    """Install compatibility shims for optional dependency drift."""
    import transformers

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        transformers.AutoModelForVision2Seq = object
