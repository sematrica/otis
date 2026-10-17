"""Verbatim transcription engine.

The engine runs as a long-lived JSON-RPC server (``verbatim-engine serve``) that
owns the SQLite database and schedules transcription jobs, each of which runs in
its own worker subprocess (``verbatim-engine worker``).
"""

__version__ = "0.4.0"

#: Version of the JSON-lines protocol spoken with the macOS app. Bump on any
#: incompatible change to method names, parameters or result shapes.
PROTOCOL_VERSION = 4
