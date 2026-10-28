"""Relay a running job's Dask dashboard through the console.

A job's Dask scheduler serves its dashboard on a free port of its own node
and notes where in the run folder (``dashboard.json``, see
:func:`almasim.jobs.rundir.read_dashboard`). The browser reaches the console
only (an SSH tunnel, or the portal's proxy), so the console passes
``/api/v2/jobs/<id>/dashboard/<path>`` on to ``http://<host>:<port>/<path>``,
websockets included.

The dashboard is served without a prefix: every URL it emits is relative to
the page (``static/js/...``, ``../../status``) and Bokeh opens its websocket
at the page's own path plus ``/ws``, so cutting the route off is enough and
the job needs to know nothing about the console's address. The one absolute
URL, a redirect's ``Location``, is put back under the route.

This is plain ASGI rather than a FastAPI route: the token check
(:class:`almasim.web.app.TokenMiddleware`) sees HTTP requests only, so the
websocket half checks the token itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import re
from collections.abc import Callable
from http.cookies import SimpleCookie
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import httpx
from starlette.responses import JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from almasim.jobs.rundir import DashboardAddress, read_dashboard
from almasim.jobs.service import RunError, view

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

ROUTE = re.compile(r"^/api/v2/jobs/(?P<run_id>[^/]+)/dashboard(?P<rest>/.*)?$")
# Not passed on either way: they describe one hop, or carry the console's token.
_HOP = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
    b"host",
    b"cookie",
    b"authorization",
}


class DashboardUnavailable(Exception):
    """Why a run's dashboard cannot be relayed, with the HTTP status to say it with."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def dashboard_of(run_dir: Path) -> DashboardAddress:
    """The address to relay a run's dashboard to, or :class:`DashboardUnavailable`."""
    run = view(run_dir)
    if run.state.finished:
        raise DashboardUnavailable(409, "The run has finished; its Dask dashboard ended with it.")
    address = read_dashboard(run_dir)
    if address is None:
        raise DashboardUnavailable(404, "This run has no Dask dashboard (yet).")
    return address


class DashboardRelay:
    """Pass ``/api/v2/jobs/<id>/dashboard/...`` on to that run's Dask dashboard."""

    def __init__(
        self,
        app: ASGIApp,
        token: str,
        find_run: Callable[[str], Path],
        cookie: str,
    ) -> None:
        self.app = app
        self.token = token
        self.find_run = find_run
        self.cookie = cookie

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        root = scope.get("root_path", "")
        match = ROUTE.match(scope["path"][len(root) :])
        if match is None:
            await self.app(scope, receive, send)
            return
        here = f"{root}/api/v2/jobs/{match['run_id']}/dashboard"
        rest = match["rest"]
        if scope["type"] == "http" and not rest:
            await RedirectResponse(f"{here}/status", status_code=307)(scope, receive, send)
            return
        try:
            address = await anyio.to_thread.run_sync(self._address, match["run_id"])
        except DashboardUnavailable as exc:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008, "reason": exc.detail})
            else:
                await JSONResponse({"detail": exc.detail}, status_code=exc.status)(
                    scope, receive, send
                )
            return
        if scope["type"] == "http":
            await _relay_http(scope, receive, send, address, rest or "/", here)
        elif self._authenticated(scope):
            await _relay_websocket(scope, receive, send, address, rest or "/")
        else:
            await send({"type": "websocket.close", "code": 1008, "reason": "Not authenticated"})

    def _address(self, run_id: str) -> DashboardAddress:
        try:
            return dashboard_of(self.find_run(run_id))
        except RunError as exc:
            raise DashboardUnavailable(404, str(exc)) from exc

    def _authenticated(self, scope: Scope) -> bool:
        cookies = SimpleCookie()
        for name, value in scope.get("headers", []):
            if name == b"cookie":
                cookies.load(value.decode("latin-1"))
        presented = cookies[self.cookie].value if self.cookie in cookies else ""
        return bool(presented) and hmac.compare_digest(presented.encode(), self.token.encode())


def _forwarded(headers: list[tuple[bytes, bytes]]) -> list[tuple[str, str]]:
    return [
        (name.decode("latin-1"), value.decode("latin-1"))
        for name, value in headers
        if name.lower() not in _HOP
    ]


async def _relay_http(
    scope: Scope, receive: Receive, send: Send, address: DashboardAddress, path: str, here: str
) -> None:
    body = b""
    more = True
    while more:
        message = await receive()
        body += message.get("body", b"")
        more = message.get("more_body", False)
    query = scope.get("query_string", b"").decode("latin-1")
    url = f"http://{address.host}:{address.port}{path}" + (f"?{query}" if query else "")
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
        try:
            request = client.build_request(
                scope["method"], url, headers=_forwarded(scope["headers"]), content=body
            )
            upstream = await client.send(request, stream=True)
        except httpx.HTTPError as exc:
            detail = f"The Dask dashboard at {address.host}:{address.port} does not answer: {exc}"
            await JSONResponse({"detail": detail}, status_code=502)(scope, receive, send)
            return
        try:
            headers = [
                (name, value)
                for name, value in upstream.headers.raw
                if name.lower() not in _HOP and name.lower() != b"set-cookie"
            ]
            location = upstream.headers.get("location", "")
            if location.startswith("/"):
                headers = [(n, v) for n, v in headers if n.lower() != b"location"]
                headers.append((b"location", f"{here}{location}".encode("latin-1")))
            await send(
                {"type": "http.response.start", "status": upstream.status_code, "headers": headers}
            )
            async for chunk in upstream.aiter_raw():
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b""})
        finally:
            await upstream.aclose()


async def _relay_websocket(
    scope: Scope, receive: Receive, send: Send, address: DashboardAddress, path: str
) -> None:
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidHandshake
    from websockets.typing import Subprotocol

    message = await receive()  # websocket.connect
    if message["type"] != "websocket.connect":
        return
    query = scope.get("query_string", b"").decode("latin-1")
    url = f"ws://{address.host}:{address.port}{path}" + (f"?{query}" if query else "")
    # Bokeh sends its session token as a second subprotocol: pass them all on.
    offered = [Subprotocol(p) for p in scope.get("subprotocols", [])]
    try:
        upstream = await connect(url, subprotocols=offered or None, open_timeout=10)
    except (OSError, InvalidHandshake, TimeoutError):
        await send({"type": "websocket.close", "code": 1011})
        return
    async with upstream:
        accept: Message = {"type": "websocket.accept"}
        if upstream.subprotocol:
            accept["subprotocol"] = upstream.subprotocol
        await send(accept)
        tasks = [
            asyncio.ensure_future(_browser_to_dashboard(receive, upstream)),
            asyncio.ensure_future(_dashboard_to_browser(upstream, send)),
        ]
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


async def _browser_to_dashboard(receive: Receive, upstream: ClientConnection) -> None:
    while True:
        incoming = await receive()
        if incoming["type"] == "websocket.disconnect":
            return
        data = incoming.get("text")
        await upstream.send(data if data is not None else incoming.get("bytes", b""))


async def _dashboard_to_browser(upstream: ClientConnection, send: Send) -> None:
    from websockets.exceptions import ConnectionClosed

    try:
        async for data in upstream:
            key = "text" if isinstance(data, str) else "bytes"
            await send({"type": "websocket.send", key: data})
    except ConnectionClosed:
        pass
    # The browser may have gone first: then there is no one to tell.
    with contextlib.suppress(Exception):
        await send({"type": "websocket.close", "code": 1000})
