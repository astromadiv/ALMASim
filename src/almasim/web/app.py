"""The console's FastAPI application and its access token (web console plan, S2.1).

The server runs as one user on a shared login node. It listens on a socket
only that user can open (or on 127.0.0.1), and every request must also carry
the token ``almasim serve`` printed: as the ``almasim_token`` cookie, which
opening the printed link once sets, or as ``Authorization: Bearer <token>``.

Behind a reverse proxy that passes its path prefix on (Open OnDemand's
``/node/<host>/<port>/``), ``base_path`` serves everything under that prefix
(Open OnDemand plan, O1).
"""

from __future__ import annotations

import hmac
import os
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp, Receive, Scope, Send

from almasim.__version__ import __version__
from almasim.jobs.profile import Profile, load_profile
from almasim.jobs.service import find_run

from .api import cluster, commands, files, jobs, simulation
from .api import profile as profile_api
from .dashboard import DashboardRelay
from .events import EventHub
from .events import router as events_router

COOKIE = "almasim_token"
# The built web app: inside the package in a wheel (S2.6); in a source
# checkout, the frontend's own build folder (`npm run build`).
STATIC_DIR = Path(__file__).parent / "static"
CHECKOUT_BUILD = Path(__file__).resolve().parents[3] / "frontend" / "build"
WEB_DIR_ENV = "ALMASIM_WEB_DIR"
_BASE_PATH = re.compile(r"(/[A-Za-z0-9._~-]+)+")

_NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>ALMASim</title>
<p>The ALMASim web console is running, but its pages are not built into this
install. The API is at <a href="{base}/api/v2/health">{base}/api/v2</a>.</p>"""

_LOCKED = """<!doctype html><meta charset="utf-8"><title>ALMASim</title>
<p>Open the link <code>almasim serve</code> printed (it carries the access
token), or run <code>almasim serve --status</code> to see it again.</p>"""


def normalize_base_path(value: str | None) -> str:
    """``value`` as a URL prefix: "" for none, else ``/a/b`` (no slash at the end)."""
    path = (value or "").strip().rstrip("/")
    if not path:
        return ""
    if not _BASE_PATH.fullmatch(path) or any(p in (".", "..") for p in path.split("/")):
        raise ValueError(f"{value!r} is not a URL path prefix (e.g. /node/host/8765)")
    return path


def with_base(html: str, base: str) -> str:
    """The built page with its root paths moved under ``base``: SvelteKit's
    base path (read at run time from the page's ``base: ""``), and the
    scripts and images the page links from the root."""
    if not base:
        return html
    return (
        html.replace('base: ""', f'base: "{base}"', 1)
        .replace('"/_app/', f'"{base}/_app/')
        .replace('"/assets/', f'"{base}/assets/')
    )


class AppFiles(StaticFiles):
    """The built web app. Its pages are routed in the browser, so a path that
    is not a file (``/jobs/01J…``) gets the app's fallback page; the browser
    then shows the right page. API paths never fall back."""

    FALLBACK = "200.html"

    def __init__(self, *, directory: Path, base: str = "") -> None:
        super().__init__(directory=directory, html=True)
        self.base = base

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or path.split("/", 1)[0] == "api":
                raise
            if not self.base:
                return await super().get_response(self.FALLBACK, scope)
            page = (Path(str(self.directory)) / self.FALLBACK).read_text(encoding="utf-8")
            return HTMLResponse(with_base(page, self.base))


class BasePath:
    """Serve the app under ``base``, for a proxy that passes the prefix on.

    A request under it is routed as if the prefix were the root (ASGI's
    ``root_path``); the bare prefix redirects to its folder, so the page's
    relative links resolve; anything else is not this console's.
    """

    def __init__(self, app: ASGIApp, base: str) -> None:
        self.app = app
        self.base = base

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or not self.base:
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if scope["type"] == "websocket":  # the dashboard relay's (almasim.web.dashboard)
            if path.startswith(self.base + "/"):
                await self.app({**scope, "root_path": self.base}, receive, send)
            else:
                await send({"type": "websocket.close", "code": 1008})
        elif path == self.base:
            query = scope.get("query_string", b"").decode()
            target = self.base + "/" + (f"?{query}" if query else "")
            await RedirectResponse(target, status_code=307)(scope, receive, send)
        elif path.startswith(self.base + "/"):
            await self.app({**scope, "root_path": self.base}, receive, send)
        else:
            await PlainTextResponse("Not found", status_code=404)(scope, receive, send)


class TokenMiddleware(BaseHTTPMiddleware):
    """Let a request through only with the server's token."""

    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self.token = token

    def _valid(self, value: str | None) -> bool:
        if not value:
            return False
        return hmac.compare_digest(value.encode(), self.token.encode())

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        offered = request.query_params.get("token")
        if offered is not None and request.method == "GET":
            if not self._valid(offered):
                return HTMLResponse(_LOCKED, status_code=401)
            # Trade the link's token for a cookie, and drop it from the address bar.
            # The cookie goes to this console's own prefix only: behind a shared
            # proxy, other users' apps live under the same address.
            rest = {k: v for k, v in request.query_params.items() if k != "token"}
            target = request.scope["path"] + (f"?{urlencode(rest)}" if rest else "")
            response = RedirectResponse(target, status_code=303)
            response.set_cookie(
                COOKIE,
                self.token,
                httponly=True,
                samesite="strict",
                path=request.scope.get("root_path", "") + "/",
                max_age=30 * 86400,
            )
            return response

        bearer = request.headers.get("authorization", "")
        presented = request.cookies.get(COOKIE) or (
            bearer[7:] if bearer.lower().startswith("bearer ") else None
        )
        if not self._valid(presented):
            route = request.scope["path"][len(request.scope.get("root_path", "")) :]
            if route.startswith("/api/"):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)
            return HTMLResponse(_LOCKED, status_code=401)
        return await call_next(request)


def web_dir(environ: Mapping[str, str] | None = None) -> Path | None:
    """Where the built web app is: ``$ALMASIM_WEB_DIR``, else the packaged
    copy, else a source checkout's ``frontend/build``; None when there is none."""
    env = os.environ if environ is None else environ
    if env.get(WEB_DIR_ENV):
        return Path(env[WEB_DIR_ENV]).expanduser()
    for candidate in (STATIC_DIR, CHECKOUT_BUILD):
        if (candidate / AppFiles.FALLBACK).is_file():
            return candidate
    return None


def create_app(
    token: str,
    static_dir: Path | None = None,
    *,
    profile: Profile | None = None,
    events_interval: float = 15.0,
    base_path: str = "",
) -> FastAPI:
    """The console app, serving the built web app from ``static_dir`` (default: ``web_dir()``).

    ``profile`` is the job profile the server submits with (default: the
    one ``almasim`` reads from the environment); ``events_interval`` is how
    often, in seconds, it looks at the runs between actions; ``base_path``
    is the URL prefix it is served under (default: the root).
    """
    if not token:
        raise ValueError("the console needs an access token")
    base = normalize_base_path(base_path)
    profile = profile if profile is not None else load_profile()
    hub = EventHub(profile, interval=events_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        hub.start()
        try:
            yield
        finally:
            hub.stop()

    app = FastAPI(
        title="ALMASim web console", version=__version__, docs_url="/api/docs", lifespan=lifespan
    )
    app.state.profile = profile
    app.state.events = hub
    app.state.cluster = cluster.ClusterView(profile)
    # Inside the token check, which covers its HTTP half; it checks websockets itself.
    app.add_middleware(
        DashboardRelay,
        token=token,
        find_run=lambda run_id: find_run(profile, run_id),
        cookie=COOKIE,
    )
    app.add_middleware(TokenMiddleware, token=token)
    app.add_middleware(BasePath, base=base)  # added last: runs first

    @app.get("/api/v2/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(commands.router)
    app.include_router(jobs.router)
    app.include_router(files.router)
    app.include_router(cluster.router)
    app.include_router(simulation.router)
    app.include_router(profile_api.router)
    app.include_router(events_router)

    static_dir = static_dir if static_dir is not None else web_dir()
    if static_dir is not None and (static_dir / AppFiles.FALLBACK).is_file():
        app.mount("/", AppFiles(directory=static_dir, base=base), name="web")
    else:

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def not_built() -> str:
            return _NOT_BUILT.format(base=base)

    return app
