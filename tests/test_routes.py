from fastapi.testclient import TestClient


def _client(app_modules):
    return TestClient(app_modules["app.main"].app)


def test_health(app_modules):
    client = _client(app_modules)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "dead" in body
    assert "cooling" in body
    assert "models" in body


def test_v1_models_lists_special_ids_and_ladder(app_modules):
    client = _client(app_modules)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "nvidia-auto" in ids
    assert "nvidia-only" in ids
    assert "local-only" in ids


def test_stats_endpoint_returns_model_snapshot(app_modules):
    client = _client(app_modules)
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["models"], list)
    assert "saved_usd_total" in body


def test_dashboard_renders_html(app_modules):
    client = _client(app_modules)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "failover proxy" in resp.text
    assert "<table>" in resp.text


def test_dashboard_static_assets_are_served(app_modules):
    client = _client(app_modules)
    css = client.get("/static/dashboard.css")
    js = client.get("/static/dashboard.js")
    assert css.status_code == 200
    assert js.status_code == 200


def test_config_get_and_post_roundtrip(app_modules):
    client = _client(app_modules)
    ladder_config = app_modules["app.ladder"].ladder_config
    ladder_config.add_provider("groq", base_url="https://api.groq.com/openai/v1", api_key="k", models=["m1", "m2"])

    resp = client.get("/_config")
    assert resp.status_code == 200
    order = resp.json()["order"]
    assert "m1" in order and "m2" in order

    resp = client.post("/_config", json={"order": ["m2", "m1"], "disabled": ["m1"]})
    assert resp.status_code == 200
    assert resp.json()["order"] == ["m2", "m1"]
    assert resp.json()["disabled"] == ["m1"]
    assert resp.json()["active"] == ["m2"]


def test_config_post_rejects_bad_types(app_modules):
    client = _client(app_modules)
    resp = client.post("/_config", json={"order": "not-a-list", "disabled": []})
    assert resp.status_code == 400


def test_reset_cooldowns_and_reset_stats(app_modules):
    client = _client(app_modules)
    cascade = app_modules["app.state"].cascade
    cascade.cool(cascade._serving_ladder()[0], 60)

    resp = client.post("/_reset_cooldowns")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert cascade.model_until == {}

    resp = client.post("/_reset_stats")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_chat_completions_intercepts_slash_command(app_modules):
    client = _client(app_modules)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "/health"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "proxy-commands"
    assert "Proxy Health" in body["choices"][0]["message"]["content"]


def test_chat_completions_invalid_json_body(app_modules):
    client = _client(app_modules)
    resp = client.post(
        "/v1/chat/completions",
        data=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
