"""Thin wrapper around ``snowflake-connector-python``.

Reads connection parameters from ``~/.snowflake/connections.toml`` (via
:pymod:`arcticswarm.config`) and exposes the session connection plus the
REST-auth helpers used by Cortex Search (``_get_rest_url`` / ``_get_token`` /
``_get_account``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import snowflake.connector

# Suppress noisy "TelemetryClient is closed" warnings that fire when the
# Snowflake connector closes a connection — the SDK's own teardown sequence
# races with internal telemetry logging and there is nothing we can do about it.
logger = logging.getLogger(__name__)


def _get_snowflake_connector() -> Any:
    """Import snowflake.connector lazily.

    Some datasets do not use Snowflake at all, so importing the connector
    only when a real connection is needed avoids unnecessary binary dependency
    failures in minimal task containers.
    """
    import snowflake.connector

    logging.getLogger("snowflake.connector.telemetry").setLevel(logging.CRITICAL)
    return snowflake.connector


class SnowflakeClient:
    """Lazy Snowflake connection manager.

    Provides the session connection and the REST-auth helpers Cortex Search
    needs (``_get_rest_url`` / ``_get_token`` / ``_get_account``).
    """

    def __init__(self, params: dict[str, Any], sql_timeout: int = 0) -> None:
        self._params = params
        self._sql_timeout = sql_timeout
        self._conn: Any | None = None

    # -- connection lifecycle ------------------------------------------------

    def _connect(self) -> Any:
        if self._conn is None or self._conn.is_closed():
            connector = _get_snowflake_connector()
            self._conn = connector.connect(**self._params)
            cur = self._conn.cursor()
            # Explicitly set warehouse if specified in params
            if 'warehouse' in self._params and self._params['warehouse']:
                cur.execute(f"USE WAREHOUSE {self._params['warehouse']}")
            if self._sql_timeout > 0:
                cur.execute(
                    f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {self._sql_timeout}"
                )
            cur.close()
        return self._conn

    def close(self) -> None:
        if self._conn and not self._conn.is_closed():
            self._conn.close()
            self._conn = None

    # -- REST API helpers (Cortex Search auth) -------------------------------

    def _get_rest_url(self) -> str:
        """Derive the Snowflake REST URL (host) from the connection."""
        conn = self._connect()
        return str(conn.host)

    def _get_token(self) -> str:
        """Get the current session token from the active connection."""
        conn = self._connect()
        return str(conn.rest.token)

    def _get_account(self) -> str:
        """Get the account identifier from the active connection."""
        conn = self._connect()
        return str(conn.account)
