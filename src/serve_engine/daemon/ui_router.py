from __future__ import annotations

from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


def make_ui_router() -> APIRouter | None:
    """Return a router that serves the bundled UI, or None if the dist is missing."""
    ui_dir = files("serve_engine.ui")
    index = ui_dir.joinpath("index.html")
    try:
        index_text = index.read_text()
    except FileNotFoundError:
        return None
    router = APIRouter()
    assets_dir = str(ui_dir.joinpath("assets"))
    router.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @router.get("/", response_class=HTMLResponse)
    def index_html() -> HTMLResponse:
        return HTMLResponse(content=index_text)

    return router
