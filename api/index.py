"""Vercel serverless entry point.

Vercel expects a WSGI-compatible ``app`` object in ``api/index.py``.
We import the Flask app from ``web.app`` and expose it here.

The ``matrix_operations`` package is resolved because the project root
is on ``sys.path`` (Vercel adds it automatically).
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on path so both ``matrix_operations`` and ``web``
# are importable regardless of how Vercel resolves the working directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web.app import app  # noqa: E402, F401

# Vercel looks for an ``app`` object that is WSGI-compatible.
# Flask's app object is already WSGI-compatible, so this just works.
