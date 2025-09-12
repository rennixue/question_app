import logging
from typing import Any

from httpx import AsyncClient, Request

from ..context import request_id
from ..models import Question

logger = logging.getLogger(__name__)


class CallbackService:
    def __init__(self, base_url: str, skip: bool = False) -> None:
        self._client = AsyncClient(base_url=base_url, headers={"X-Request-Id": request_id.get()})
        self._skip = skip

    async def _send_req(self, req: Request) -> None:
        if self._skip:
            logger.info(
                f"skip callback\n{req.method} {req.url}\n{'\n'.join(k + ': ' + v for k, v in req.headers.items())}\n\n{req.content!r}"
            )
        else:
            try:
                resp = await self._client.send(req)
                resp.raise_for_status()
            except Exception as exc:
                logger.error("callback failure: %r, %s %r %r %r", exc, req.method, req.url, req.headers, req.content)
            else:
                logger.debug("callback success: %r", resp.content)

    async def notify_generate_err(self, db_id: int, err_msg: str) -> None:
        req = self._client.build_request(
            "POST",
            "/courseware_platform/question/callback/generate",
            json={"status": 0, "taskId": db_id, "error": err_msg},
        )
        await self._send_req(req)

    async def notify_generate_ok(self, db_id: int, qs: list[Question], err_msg: str | None = None) -> None:
        questions: list[dict[str, Any]] = []
        for no, q in enumerate(qs, 1):
            question = {
                "questionNo": q.id.hex,
                "questionType": q.type.to_int(),
                "genType": q.source.to_int(),
                "genNo": no,
                "genQuestion": q.content,
            }
            if q.meta_info:
                question["genQuestionInfo"] = q.meta_info
            questions.append(question)
        req_body: dict[str, Any] = {"status": 1, "taskId": db_id, "questions": questions}
        if err_msg:
            req_body["error"] = err_msg
        req = self._client.build_request("POST", "/courseware_platform/question/callback/generate", json=req_body)
        await self._send_req(req)

    async def notify_rewrite_err(self, db_id: int, err_msg: str) -> None:
        req = self._client.build_request(
            "POST",
            "/courseware_platform/question/callback/rewritten",
            json={"status": 0, "questionId": db_id, "error": err_msg},
        )
        await self._send_req(req)

    async def notify_rewrite_ok(self, db_id: int, q: Question) -> None:
        req_body: dict[str, Any] = {
            "status": 1,
            "questionId": db_id,
            "question": {
                "questionNo": q.id.hex,
                "questionType": q.type.to_int(),
                "genType": q.source.to_int(),
                "genNo": -1,
                "genQuestion": q.content,
            },
        }
        if q.meta_info:
            req_body["question"]["genQuestionInfo"] = q.meta_info
        req = self._client.build_request("POST", "/courseware_platform/question/callback/rewritten", json=req_body)
        await self._send_req(req)
