from collections.abc import Callable
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .context import request_id


def _default_gen() -> str:
    return uuid4().hex


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp, header: str | None = None, gen: Callable[[], str] | None = None) -> None:
        self._app = app
        self._header = "X-Request-ID" if header is None else header
        self._gen = _default_gen if gen is None else gen

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        val = Headers(scope=scope).get(self._header)
        request_id.set(self._gen() if val is None else val)

        async def add_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append(self._header, request_id.get())
            await send(message)

        await self._app(scope, receive, add_header)
