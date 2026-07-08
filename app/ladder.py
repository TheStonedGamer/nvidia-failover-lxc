"""LadderConfig: persisted provider registry + failover ladder order/disabled
set. Backed by the shared kv table so a copied proxy.db carries all providers,
keys, and ladder state over unchanged."""

import json
from typing import Dict, List, Optional, Set

from app.config import PROVIDER_ENV, NVIDIA_BASE_URL, LOCAL_BASE_URL, LOCAL_MODEL, resolve_api_key
from app.db import get_db, kv_get_all, kv_set, CONFIG_FILE


class LadderConfig:
    def __init__(self):
        self.order: List[str] = []
        self.disabled: Set[str] = set()
        self.providers: Dict[str, dict] = {}
        self.nvidia_key: Optional[str] = None
        self.local_tail: Optional[str] = None
        self.load()

    # --- persistence -----------------------------------------------------
    def load(self) -> None:
        conn = get_db()
        try:
            kv = kv_get_all(conn)
        finally:
            conn.close()

        if kv.get("order"):
            try:
                self.order = json.loads(kv["order"])
            except Exception:
                pass
        if kv.get("disabled"):
            try:
                self.disabled = set(json.loads(kv["disabled"]))
            except Exception:
                pass
        if kv.get("providers"):
            try:
                self.providers = json.loads(kv["providers"])
            except Exception:
                pass
        self.nvidia_key = kv.get("nvidia_key") or None
        self.local_tail = kv.get("local_tail") or None

        first_run = not kv
        if first_run:
            self._import_json()

        if not kv.get("migrated_nvidia_v1"):
            self._migrate_nvidia_provider(kv)

        if not kv.get("seeded_providers_v1"):
            self._seed_providers_from_env()

        self._write()

    def _migrate_nvidia_provider(self, kv: Dict[str, str]) -> None:
        """One-time: fold the legacy standalone nvidia_key/order into the
        first-class "nvidia" provider entry, but only if legacy state exists —
        a brand-new install has nothing to migrate."""
        legacy_key = kv.get("nvidia_key") or self.nvidia_key
        if (legacy_key or self.order) and "nvidia" not in self.providers:
            self.providers["nvidia"] = {
                "base_url": NVIDIA_BASE_URL,
                "api_key": legacy_key or "",
                "models": list(self.order),
            }

    def _seed_providers_from_env(self) -> None:
        """First-run only: populate providers from <PREFIX>_API_KEY / _MODELS /
        _BASE_URL env vars for every entry in PROVIDER_ENV, plus legacy
        fallbacks (NVIDIA_API_KEY / LOCAL_OLLAMA_URL / LOCAL_MODEL /
        ROUTER_NVIDIA_MODELS). Ollama needs no key and its models are never
        added to the cloud failover order."""
        for name, default_base in PROVIDER_ENV.items():
            prefix = name.upper()
            env_key = None
            env_models: List[str] = []
            env_base = None
            import os

            if name == "nvidia":
                env_key = resolve_api_key()
                models_env = os.environ.get("ROUTER_NVIDIA_MODELS")
                if models_env:
                    env_models = [m.strip() for m in models_env.split(",") if m.strip()]
            else:
                env_key = os.environ.get(f"{prefix}_API_KEY")
                models_env = os.environ.get(f"{prefix}_MODELS")
                if models_env:
                    env_models = [m.strip() for m in models_env.split(",") if m.strip()]

            env_base = os.environ.get(f"{prefix}_BASE_URL")
            if name == "ollama":
                env_base = env_base or os.environ.get("LOCAL_OLLAMA_URL") or default_base
                if not env_models and os.environ.get("LOCAL_MODEL"):
                    env_models = [os.environ["LOCAL_MODEL"]]

            has_signal = env_key or env_models or (name == "ollama")
            if not has_signal or name in self.providers:
                continue

            self.providers[name] = {
                "base_url": env_base or default_base,
                "api_key": env_key or "",
                "models": env_models,
            }
            if name != "ollama":
                for m in env_models:
                    if m not in self.order:
                        self.order.append(m)

        if not self.local_tail and LOCAL_MODEL:
            ollama = self.providers.get("ollama")
            if ollama and ollama.get("models"):
                self.local_tail = ollama["models"][0]
            else:
                self.local_tail = LOCAL_MODEL

    def _import_json(self) -> None:
        """Legacy proxy_config.json import, for upgrades that predate the
        SQLite kv store."""
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        self.order = data.get("order") or self.order
        self.disabled = set(data.get("disabled") or [])
        self.providers = data.get("providers") or self.providers
        self.nvidia_key = data.get("nvidia_key") or self.nvidia_key
        self.local_tail = data.get("local_tail") or self.local_tail

    def _write(self) -> None:
        conn = get_db()
        try:
            kv_set(
                conn,
                {
                    "order": json.dumps(self.order),
                    "disabled": json.dumps(sorted(self.disabled)),
                    "providers": json.dumps(self.providers),
                    "nvidia_key": self.nvidia_key or "",
                    "local_tail": self.local_tail or "",
                    "migrated_nvidia_v1": "1",
                    "seeded_providers_v1": "1",
                },
            )
            conn.commit()
        finally:
            conn.close()

    def save(self) -> None:
        self._write()

    # --- mutation ----------------------------------------------------------
    def update(self, order: List[str], disabled: List[str]) -> None:
        self.order = list(order)
        self.disabled = set(disabled)
        self._write()

    def set_nvidia_key(self, key: str) -> None:
        self.nvidia_key = key
        prov = self.providers.setdefault("nvidia", {"base_url": NVIDIA_BASE_URL, "models": []})
        prov["api_key"] = key
        self._write()

    def add_provider(self, name: str, base_url: str = "", api_key: str = "", models: Optional[List[str]] = None) -> None:
        """Merge semantics: a blank field keeps the previous value; models are
        unioned, not replaced, so repeated discovery calls are additive."""
        existing = self.providers.get(name, {})
        new_base = base_url or existing.get("base_url", "")
        new_key = api_key or existing.get("api_key", "")
        merged_models = list(existing.get("models", []))
        for m in models or []:
            if m not in merged_models:
                merged_models.append(m)
        self.providers[name] = {"base_url": new_base, "api_key": new_key, "models": merged_models}
        self._write()

    def remove_provider(self, name: str) -> None:
        self.providers.pop(name, None)
        self._write()

    def set_local_tail(self, model: str) -> None:
        self.local_tail = model
        self._write()

    # --- queries -------------------------------------------------------------
    def is_enabled(self, model: str) -> bool:
        return model not in self.disabled

    def active_ladder(self) -> List[str]:
        return [m for m in self.order if self.is_enabled(m)]

    def custom_models(self) -> List[str]:
        return list(self.order)

    def model_provider(self, model: str):
        for name, p in self.providers.items():
            if model in (p.get("models") or []):
                return p.get("base_url"), p.get("api_key")
        # Not in the curated ladder — check the live-discovery cache so any
        # individual model a provider offers is still routable by name, not
        # just the ones explicitly added to the failover order.
        from app.discovery import cached_provider_for

        name = cached_provider_for(model)
        if name and name in self.providers:
            p = self.providers[name]
            return p.get("base_url"), p.get("api_key")
        return None

    def resolved_nvidia_key(self) -> Optional[str]:
        prov = self.providers.get("nvidia")
        if prov and prov.get("api_key"):
            return prov["api_key"]
        return self.nvidia_key or resolve_api_key()


ladder_config = LadderConfig()
