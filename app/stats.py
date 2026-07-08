"""Per-model live/learned stats: request counters, EWMA-learned RPM/TPM
ceilings (with floors to prevent a low-traffic 429 death-spiral), rolling
token accounting, and money-saved estimation. Persisted to the `stats` table
in the shared kv/stats SQLite file."""

import time
from typing import Dict, List, Optional

from app.config import (
    _EWMA_ALPHA,
    _RPM_WINDOW_S,
    _THROTTLE_RATIO,
    _RPM_LIMIT_FLOOR,
    _TPM_IN_LIMIT_FLOOR,
    _TPM_OUT_LIMIT_FLOOR,
    _SERVING_WINDOW_S,
    _STICKY_LADDER,
    saved_usd,
)
from app.db import get_db, secure_db_file, kv_get_all

import json
import os

STATS_PATH = os.environ.get("PROXY_STATS_FILE", "proxy_stats.json")


def _load_json_stats() -> Dict[str, dict]:
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


class Stats:
    def __init__(self):
        self.models: Dict[str, dict] = {}
        self._recent: Dict[str, List[float]] = {}
        self._recent_tok_in: Dict[str, List[tuple]] = {}
        self._recent_tok_out: Dict[str, List[tuple]] = {}
        self._last_save = 0.0
        self._serving_model: Optional[str] = None
        self._serving_at: float = 0.0

    # --- serving-now tracking -------------------------------------------------
    def note_serving(self, model: str) -> None:
        self._serving_model = model
        self._serving_at = time.monotonic()

    def sticky_model(self) -> Optional[str]:
        if not _STICKY_LADDER:
            return None
        return self._serving_model

    def current_serving(self, now: Optional[float] = None) -> Optional[str]:
        now = now if now is not None else time.monotonic()
        if self._serving_model and (now - self._serving_at) <= _SERVING_WINDOW_S:
            return self._serving_model
        return None

    def reset_all(self) -> None:
        self.models = {}
        self._recent = {}
        self._recent_tok_in = {}
        self._recent_tok_out = {}
        self._serving_model = None
        self._serving_at = 0.0
        conn = get_db()
        try:
            conn.execute("DELETE FROM stats")
            conn.commit()
        finally:
            conn.close()
        secure_db_file()

    def _m(self, model: str) -> dict:
        d = self.models.setdefault(model, {})
        d.setdefault("requests", 0)
        d.setdefault("successes", 0)
        d.setdefault("rate_limited", 0)
        d.setdefault("errors", 0)
        d.setdefault("last_429", None)
        d.setdefault("last_success", None)
        d.setdefault("retry_after_ewma", None)
        d.setdefault("limit_rpm", None)
        d.setdefault("peak_rpm", 0)
        d.setdefault("limit_tpm_in", None)
        d.setdefault("limit_tpm_out", None)
        d.setdefault("peak_tpm_in", 0)
        d.setdefault("peak_tpm_out", 0)
        d.setdefault("tokens_in", 0)
        d.setdefault("tokens_out", 0)
        d.setdefault("tokens_total", 0)
        d.setdefault("last_in", 0)
        d.setdefault("last_out", 0)
        return d

    def _rpm(self, model: str, now: float) -> int:
        window = self._recent.setdefault(model, [])
        cutoff = now - _RPM_WINDOW_S
        while window and window[0] < cutoff:
            window.pop(0)
        return len(window)

    def _tpm(self, window_map: Dict[str, List[tuple]], model: str, now: float) -> int:
        window = window_map.setdefault(model, [])
        cutoff = now - _RPM_WINDOW_S
        while window and window[0][0] < cutoff:
            window.pop(0)
        return sum(tok for _, tok in window)

    def _snapshot_tpm(self, model: str, now: Optional[float] = None) -> tuple:
        now = now if now is not None else time.monotonic()
        return (
            self._tpm(self._recent_tok_in, model, now),
            self._tpm(self._recent_tok_out, model, now),
        )

    def live_tpm_in(self, model: str) -> int:
        return self._tpm(self._recent_tok_in, model, time.monotonic())

    def live_tpm_out(self, model: str) -> int:
        return self._tpm(self._recent_tok_out, model, time.monotonic())

    def learned_tpm_in(self, model: str) -> Optional[float]:
        return self._m(model).get("limit_tpm_in")

    def learned_tpm_out(self, model: str) -> Optional[float]:
        return self._m(model).get("limit_tpm_out")

    def is_near_limit(self, model: str) -> bool:
        d = self._m(model)
        now = time.monotonic()
        rpm = self._rpm(model, now)
        limit_rpm = d.get("limit_rpm")
        if limit_rpm and rpm >= max(limit_rpm, _RPM_LIMIT_FLOOR) * _THROTTLE_RATIO:
            return True
        tpm_in, tpm_out = self._snapshot_tpm(model, now)
        limit_in = d.get("limit_tpm_in")
        if limit_in and tpm_in >= max(limit_in, _TPM_IN_LIMIT_FLOOR) * _THROTTLE_RATIO:
            return True
        limit_out = d.get("limit_tpm_out")
        if limit_out and tpm_out >= max(limit_out, _TPM_OUT_LIMIT_FLOOR) * _THROTTLE_RATIO:
            return True
        return False

    def record_request(self, model: str) -> None:
        d = self._m(model)
        d["requests"] += 1
        now = time.monotonic()
        self._recent.setdefault(model, []).append(now)
        d["peak_rpm"] = max(d["peak_rpm"], self._rpm(model, now))
        self._maybe_save()

    def record_success(self, model: str) -> None:
        d = self._m(model)
        d["successes"] += 1
        now = time.time()
        if d.get("last_429"):
            gap = now - d["last_429"]
            if gap > 0:
                self._learn_cooldown(model, gap)
        d["last_success"] = now
        self._maybe_save()

    def record_429(self, model: str, retry_after: Optional[float] = None) -> None:
        d = self._m(model)
        d["rate_limited"] += 1
        d["last_429"] = time.time()
        now = time.monotonic()
        rpm = self._rpm(model, now)
        tpm_in, tpm_out = self._snapshot_tpm(model, now)
        prev_rpm = d.get("limit_rpm")
        d["limit_rpm"] = rpm if prev_rpm is None else (_EWMA_ALPHA * rpm + (1 - _EWMA_ALPHA) * prev_rpm)
        prev_in = d.get("limit_tpm_in")
        d["limit_tpm_in"] = tpm_in if prev_in is None else (_EWMA_ALPHA * tpm_in + (1 - _EWMA_ALPHA) * prev_in)
        prev_out = d.get("limit_tpm_out")
        d["limit_tpm_out"] = tpm_out if prev_out is None else (_EWMA_ALPHA * tpm_out + (1 - _EWMA_ALPHA) * prev_out)
        if retry_after:
            self._learn_cooldown(model, retry_after)
        self._maybe_save()

    def record_error(self, model: str) -> None:
        d = self._m(model)
        d["errors"] += 1
        self._maybe_save()

    def record_usage(self, model: str, usage: dict) -> None:
        d = self._m(model)
        tin = int((usage or {}).get("prompt_tokens") or 0)
        tout = int((usage or {}).get("completion_tokens") or 0)
        ttotal = int((usage or {}).get("total_tokens") or (tin + tout))
        d["tokens_in"] += tin
        d["tokens_out"] += tout
        d["tokens_total"] += ttotal
        d["last_in"] = tin
        d["last_out"] = tout
        now = time.monotonic()
        if tin:
            self._recent_tok_in.setdefault(model, []).append((now, tin))
        if tout:
            self._recent_tok_out.setdefault(model, []).append((now, tout))
        tpm_in, tpm_out = self._snapshot_tpm(model, now)
        d["peak_tpm_in"] = max(d["peak_tpm_in"], tpm_in)
        d["peak_tpm_out"] = max(d["peak_tpm_out"], tpm_out)
        self._maybe_save()

    def _learn_cooldown(self, model: str, secs: float) -> None:
        d = self._m(model)
        prev = d.get("retry_after_ewma")
        d["retry_after_ewma"] = secs if prev is None else (_EWMA_ALPHA * secs + (1 - _EWMA_ALPHA) * prev)

    def learned_cooldown(self, model: str) -> Optional[float]:
        return self._m(model).get("retry_after_ewma")

    def live_rpm(self, model: str) -> int:
        return self._rpm(model, time.monotonic())

    def saved_usd_total(self) -> float:
        total = 0.0
        for model, d in self.models.items():
            total += saved_usd(model, d.get("tokens_in", 0), d.get("tokens_out", 0))
        return total

    def _maybe_save(self) -> None:
        now = time.time()
        if now - self._last_save >= 10:
            self._save()
            self._last_save = now

    def _save(self) -> None:
        conn = get_db()
        try:
            rows = [(model, json.dumps(d)) for model, d in self.models.items()]
            conn.executemany(
                "INSERT INTO stats(model, data) VALUES(?, ?) "
                "ON CONFLICT(model) DO UPDATE SET data=excluded.data",
                rows,
            )
            conn.commit()
        finally:
            conn.close()
        secure_db_file()

    @classmethod
    def load(cls) -> "Stats":
        inst = cls()
        conn = get_db()
        try:
            rows = list(conn.execute("SELECT model, data FROM stats"))
        finally:
            conn.close()
        if rows:
            for model, data in rows:
                try:
                    inst.models[model] = json.loads(data)
                except Exception:
                    pass
        else:
            legacy = _load_json_stats()
            if legacy:
                inst.models = legacy
                inst._save()
        return inst
