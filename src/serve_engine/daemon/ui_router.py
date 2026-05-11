from __future__ import annotations

from importlib.resources import files

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


def install_ui(app: FastAPI) -> bool:
    """Mount the bundled UI on `app`. Returns True if the dist was found.

    - `GET /` returns the SPA index.html.
    - `GET /assets/*` is served by StaticFiles.

    StaticFiles must be mounted on the FastAPI app, not on an APIRouter
    (APIRouter.mount() is unreliable for nested static apps).
    """
    ui_dir = files("serve_engine.ui")
    index = ui_dir.joinpath("index.html")
    try:
        index_text = index.read_text()
    except FileNotFoundError:
        return False
    assets_dir = str(ui_dir.joinpath("assets"))

    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def index_html() -> HTMLResponse:
        return HTMLResponse(content=index_text)

    app.include_router(router)
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    return True


# Legacy alias kept for backwards compatibility — returns None now that we
# install directly onto the app. Callers should use install_ui() instead.
def make_ui_router():  # pragma: no cover
    return None
