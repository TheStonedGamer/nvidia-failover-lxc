def test_fresh_config_seeds_ollama_provider(app_modules):
    ladder_config = app_modules["app.ladder"].ladder_config
    # Ollama always seeds (has_signal is unconditionally true for it) even
    # with no env vars set, since it needs no API key.
    assert "ollama" in ladder_config.providers


def test_add_provider_merges_models_not_replaces(app_modules):
    ladder_config = app_modules["app.ladder"].ladder_config
    ladder_config.add_provider("openrouter", base_url="https://openrouter.ai/api/v1", api_key="k1", models=["model-a"])
    ladder_config.add_provider("openrouter", api_key="", models=["model-b"])
    prov = ladder_config.providers["openrouter"]
    assert prov["models"] == ["model-a", "model-b"]
    # Blank api_key on the second call must not clobber the first.
    assert prov["api_key"] == "k1"


def test_remove_provider(app_modules):
    ladder_config = app_modules["app.ladder"].ladder_config
    ladder_config.add_provider("groq", base_url="https://api.groq.com/openai/v1", api_key="k", models=["m1"])
    assert "groq" in ladder_config.providers
    ladder_config.remove_provider("groq")
    assert "groq" not in ladder_config.providers


def test_update_order_and_disabled_persists_across_reload(app_modules, monkeypatch):
    ladder_config = app_modules["app.ladder"].ladder_config
    ladder_config.add_provider("groq", base_url="https://api.groq.com/openai/v1", api_key="k", models=["m1", "m2"])
    ladder_config.update(order=["m1", "m2"], disabled=["m2"])

    # Recreate a LadderConfig against the same DB file and confirm it reloads
    # the same order/disabled/providers instead of the constructor defaults.
    LadderConfig = app_modules["app.ladder"].LadderConfig
    reloaded = LadderConfig()
    assert reloaded.order == ["m1", "m2"]
    assert reloaded.disabled == {"m2"}
    assert reloaded.active_ladder() == ["m1"]
    assert "groq" in reloaded.providers


def test_model_provider_lookup(app_modules):
    ladder_config = app_modules["app.ladder"].ladder_config
    ladder_config.add_provider("mistral", base_url="https://api.mistral.ai/v1", api_key="mk", models=["mistral-large"])
    assert ladder_config.model_provider("mistral-large") == ("https://api.mistral.ai/v1", "mk")
    assert ladder_config.model_provider("unknown-model") is None


def test_set_local_tail(app_modules):
    ladder_config = app_modules["app.ladder"].ladder_config
    ladder_config.set_local_tail("qwen3:4b")
    assert ladder_config.local_tail == "qwen3:4b"


def test_env_seeded_provider_added_to_order(app_modules, monkeypatch, tmp_path):
    # Env seeding only runs once per DB (seeded_providers_v1 flag), so this
    # needs a brand-new, never-seeded DB file plus the env vars set first.
    monkeypatch.setenv("PROXY_DB_FILE", str(tmp_path / "fresh.db"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GROQ_MODELS", "llama-3.3-70b-versatile")

    import importlib
    db = importlib.reload(app_modules["app.db"])
    ladder_mod = importlib.reload(app_modules["app.ladder"])

    fresh = ladder_mod.LadderConfig()
    assert fresh.providers["groq"]["api_key"] == "gsk-test"
    assert "llama-3.3-70b-versatile" in fresh.order
