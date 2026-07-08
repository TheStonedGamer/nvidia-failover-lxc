"""Test isolation: each test gets a fresh, empty proxy.db so tests never
touch a real proxy.db (dev-local or the deployed CT 3000 instance) and don't
leak provider/ladder state between tests."""

import importlib
import sys

import pytest


@pytest.fixture
def app_modules(tmp_path, monkeypatch):
    """Point PROXY_DB_FILE at a fresh temp file, then reload every app module
    that caches state at import time (db.DB_FILE, ladder_config, stats,
    cascade) so each test starts from a clean slate."""
    db_file = tmp_path / "proxy.db"
    monkeypatch.setenv("PROXY_DB_FILE", str(db_file))
    monkeypatch.setenv("PROXY_STATS_FILE", str(tmp_path / "proxy_stats.json"))
    monkeypatch.setenv("PROXY_CONFIG_FILE", str(tmp_path / "proxy_config.json"))

    names = [
        "app.config",
        "app.db",
        "app.ladder",
        "app.stats",
        "app.cascade",
        "app.state",
        "app.routes.models",
        "app.routes.dashboard",
        "app.routes.commands",
        "app.routes.chat",
        "app.routes.config_api",
        "app.main",
    ]
    for name in names:
        sys.modules.pop(name, None)

    mods = {}
    for name in names:
        mods[name] = importlib.import_module(name)
    return mods
