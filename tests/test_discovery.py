import pytest


_GROQ_URL = "https://api.groq.com/openai/v1"


@pytest.mark.asyncio
async def test_discover_all_populates_cache_and_model_provider_map(app_modules):
    discovery = app_modules["app.discovery"]
    ladder_config = app_modules["app.ladder"].ladder_config
    ladder_config.add_provider("groq", base_url=_GROQ_URL, api_key="k")

    async def fake_fetch(base_url, key):
        if base_url == _GROQ_URL:
            return {"models": ["groq/some-model", "groq/other-model"]}
        return {"error": "not this provider"}

    discovery._fetch_model_ids = fake_fetch
    await discovery.discover_all(ladder_config.providers)

    assert "groq/some-model" in discovery.all_discovered_models()
    assert discovery.cached_provider_for("groq/some-model") == "groq"


@pytest.mark.asyncio
async def test_discover_all_throttles_repeat_calls_within_interval(app_modules):
    discovery = app_modules["app.discovery"]
    ladder_config = app_modules["app.ladder"].ladder_config
    ladder_config.add_provider("groq", base_url=_GROQ_URL, api_key="k")

    calls = {"n": 0}

    async def counting_fetch(base_url, key):
        if base_url != _GROQ_URL:
            return {"error": "not this provider"}
        calls["n"] += 1
        return {"models": ["groq/m1"]}

    discovery._fetch_model_ids = counting_fetch

    await discovery.discover_all(ladder_config.providers)
    await discovery.discover_all(ladder_config.providers)
    assert calls["n"] == 1  # second call served from the throttle cache

    await discovery.discover_all(ladder_config.providers, force=True)
    assert calls["n"] == 2
