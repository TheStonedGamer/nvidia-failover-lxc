"""SQLite persistence primitives: WAL-mode connection, kv/stats tables,
and locked-down file permissions (the DB holds provider API keys)."""

import os
import sqlite3
from typing import Dict

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.environ.get("PROXY_DB_FILE", os.path.join(_HERE, "proxy.db"))


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS stats (model TEXT PRIMARY KEY, data TEXT)")
    return conn


def secure_db_file() -> None:
    for suffix in ("", "-wal", "-shm"):
        try:
            os.chmod(DB_FILE + suffix, 0o600)
        except OSError:
            pass


def kv_get_all(conn: sqlite3.Connection) -> Dict[str, str]:
    return {k: v for k, v in conn.execute("SELECT key, value FROM kv")}


def kv_set(conn: sqlite3.Connection, mapping: Dict[str, str]) -> None:
    conn.executemany(
        "INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        list(mapping.items()),
    )


CONFIG_FILE = os.environ.get("PROXY_CONFIG_FILE", "proxy_config.json")
