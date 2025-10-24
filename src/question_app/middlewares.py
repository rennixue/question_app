import asyncio
import logging
from collections.abc import Callable
from datetime import date
from uuid import uuid4

import httpx
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .context import request_id
from .dependencies import get_or_create_settings

logger = logging.getLogger(__name__)


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


class GenerateCounterMiddleware:
    def __init__(self, app: ASGIApp, mult: int) -> None:
        self._app = app
        self._mult = mult
        self._curr_count = 0
        self._last_date = date.today()

    def increment(self) -> int:
        today = date.today()
        if today != self._last_date:
            self._curr_count = 0
            self._last_date = today
        self._curr_count += 1
        return self._curr_count

    @classmethod
    async def send_alert(cls, date_: date, count: int) -> None:
        logger.warning("feishu counter sent on %s of %d times", date_, count)
        try:
            url = str((await get_or_create_settings()).feishu_webhook_url)
            req_body = {
                "msg_type": "post",
                "content": {
                    "post": {
                        "zh_cn": {
                            "title": "题库生成",
                            "content": [
                                [{"tag": "text", "text": f"日期: {date_}"}],
                                [{"tag": "text", "text": f"计数: {count}"}],
                            ],
                        }
                    }
                },
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(url, json=req_body)
        except Exception as exc:
            logger.error("fail to request feishu: %r", exc)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/api/question/generate-blocks":
            self.increment()
            if self._curr_count % self._mult == 0:
                asyncio.create_task(self.send_alert(self._last_date, self._curr_count))
        await self._app(scope, receive, send)
