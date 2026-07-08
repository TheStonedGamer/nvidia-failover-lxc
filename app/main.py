"""FastAPI app factory: route registration, static files, DB permission lockdown."""

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.db import secure_db_file
from app.discovery import discover_all
from app.ladder import ladder_config
from app.routes import chat, models, config_api, dashboard

app = FastAPI(title="nvidia-failover-proxy")

secure_db_file()

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

app.include_router(dashboard.router)
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(config_api.router)


@app.on_event("startup")
async def _warm_model_discovery():
    """So an individual (non-curated) model can be routed correctly on the
    very first chat request, not just after a client has already hit
    /v1/models once."""
    try:
        await discover_all(ladder_config.providers)
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PROXY_PORT", "5002"))
    host = os.environ.get("PROXY_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port)
