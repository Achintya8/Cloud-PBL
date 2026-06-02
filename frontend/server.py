"""
frontend/server.py
==================
Lightweight FastAPI server to host the farmer-facing frontend
on an EC2 instance. Serves static files, HTML templates, and
proxies API calls to the backend.
"""

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.logging_config import get_logger
from config.settings import app as app_cfg, grafana as grafana_cfg

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR   = BASE_DIR / "static"

app = FastAPI(title="Areca Nut Price Frontend", docs_url=None, redoc_url=None)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

BACKEND_URL = os.getenv(
    "BACKEND_API_URL",
    f"http://localhost:{app_cfg.api_port}/api/v1"
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main farmer dashboard HTML."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "api_base_url": BACKEND_URL,
            "grafana_embed_url": grafana_cfg.embed_url,
        },
    )


@app.get("/health")
async def health():
    """Frontend health check."""
    return {"status": "ok", "service": "areca-frontend"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "frontend.server:app",
        host="0.0.0.0",
        port=app_cfg.frontend_port,
        reload=False,
        log_level=app_cfg.log_level.lower(),
        workers=2,
    )
