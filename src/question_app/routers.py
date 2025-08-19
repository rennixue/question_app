import asyncio
import json
import logging
import re
from asyncio import Task, TaskGroup
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .dependencies import (
    AgentDep,
    CallbackDep,
    ElasticsearchDep,
    MysqlDep,
    OllamaDep,
    QdrantDep,
    QuestionGenerateDep,
    QuestionImitateDep,
    QuestionRewriteDep,
    QuestionSearchDep,
)
from .models import Question, QuestionGenerateReq, QuestionRewriteReq, QuestionSource, QuestionType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/health")
async def get_health(elasticsearch: ElasticsearchDep, mysql: MysqlDep, ollama: OllamaDep, qdrant: QdrantDep):
    tasks: dict[str, Task[bool]] = {}
    async with elasticsearch, ollama, qdrant:
        async with TaskGroup() as group:
            tasks["elasticsearch"] = group.create_task(elasticsearch.is_healthy())
            tasks["mysql"] = group.create_task(mysql.is_healthy())
            tasks["ollama"] = group.create_task(ollama.is_healthy())
            tasks["qdrant"] = group.create_task(qdrant.is_healthy())
    flags: dict[str, str] = {}
    for key, task in tasks.items():
        try:
            res = task.result()
        except Exception:
            res = False
        flags[key] = "UP" if res else "DOWN"
    status = "UP" if all(it == "UP" for it in flags.values()) else "DOWN"
    return {"status": status, "dependencies": flags}


async def do_generate(
    r: QuestionGenerateReq,
    question_generate: QuestionGenerateDep,
    question_imitate: QuestionImitateDep,
    question_search: QuestionSearchDep,
) -> list[Question]:
    async with question_search._elasticsearch:  # type: ignore
        qs_same_course, qs_historical = await question_search.find_questions(
            r.exam_kp, r.context, r.question_type, r.major_name, r.course_name, r.course_code, r.university_name, 10, 20
        )
    logger.debug(f"{len(qs_same_course)=}, {len(qs_historical)=}")
    async with asyncio.TaskGroup() as group:
        task_verify = group.create_task(question_imitate.verify(qs_historical, r.course_id, r.exam_kp, r.context))
        task_generate = group.create_task(
            question_generate.generate(
                r.course_id, r.exam_kp, r.context, r.question_type, r.major_name, r.course_name, 10
            )
        )
    qs_verified = task_verify.result()
    qs_generate = task_generate.result()
    logger.debug(f"{len(qs_verified)=}")
    questions = [*qs_same_course, *qs_verified, *qs_generate]
    if r.question_type != QuestionType.Any:
        questions = [it for it in questions if it.type == r.question_type]
    logger.debug(f"{len(questions)=}")
    return questions


async def wrapped_generate(
    max_exec_secs: float,
    req_body: QuestionGenerateReq,
    callback: CallbackDep,
    question_generate: QuestionGenerateDep,
    question_imitate: QuestionImitateDep,
    question_search: QuestionSearchDep,
):
    task = asyncio.create_task(do_generate(req_body, question_generate, question_imitate, question_search))
    try:
        questions = await asyncio.wait_for(task, max_exec_secs)
    except asyncio.TimeoutError:
        logger.warning("generate cancelled")
        await callback.notify_generate_err(req_body.task_id, "exceeds max execution time")
    except Exception as exc:
        logger.error("generate error: %r", exc)
        await callback.notify_generate_err(req_body.task_id, "error when generate")
    else:
        logger.debug("generate completed")
        await callback.notify_generate_ok(req_body.task_id, questions)


@router.post("/question/generate", status_code=202)
async def post_question_generate(
    req_body: QuestionGenerateReq,
    callback: CallbackDep,
    question_generate: QuestionGenerateDep,
    question_imitate: QuestionImitateDep,
    question_search: QuestionSearchDep,
):
    logger.debug("request body: %r", req_body)
    asyncio.create_task(
        wrapped_generate(180.0, req_body, callback, question_generate, question_imitate, question_search)
    )
    return {"taskId": req_body.task_id, "status": "created", "message": "ok"}


async def do_rewrite(r: QuestionRewriteReq, question_rewrite: QuestionRewriteDep) -> Question:
    return await question_rewrite.rewrite(r.rewritten_prompt, r.question, r.exam_kp, r.context, r.question_type)


async def wrapped_rewrite(
    max_exec_secs: float,
    req_body: QuestionRewriteReq,
    callback: CallbackDep,
    question_rewrite: QuestionRewriteDep,
):
    task = asyncio.create_task(do_rewrite(req_body, question_rewrite))
    try:
        question = await asyncio.wait_for(task, max_exec_secs)
    except asyncio.TimeoutError:
        logger.warning("rewrite cancelled")
        await callback.notify_rewrite_err(req_body.rewritten_from, "exceeds max execution time")
    except Exception as exc:
        logger.error("rewrite error: %r", exc)
        await callback.notify_rewrite_err(req_body.rewritten_from, "error when rewrite")
    else:
        logger.debug("rewrite completed")
        await callback.notify_rewrite_ok(req_body.rewritten_from, question)


@router.post("/question/rewrite", status_code=202)
async def post_question_rewrite(
    req_body: QuestionRewriteReq,
    callback: CallbackDep,
    question_rewrite: QuestionRewriteDep,
):
    logger.debug("request body: %r", req_body)
    asyncio.create_task(wrapped_rewrite(120.0, req_body, callback, question_rewrite))
    return {"rewritten_from": req_body.rewritten_from, "status": "created", "message": "ok"}


@router.get("/prepared/{course_id}/kps")
async def get_order_kps(course_id: int, mysql: MysqlDep, file_limit: int | None = None, kp_limit: int | None = None):
    file_limit = file_limit or 30
    kp_limit = kp_limit or 20
    precessed, files = await mysql.select_order_kps(course_id, file_limit, kp_limit)
    return {"course_id": course_id, "precessed": precessed, "files": files}


def encode_chunk(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


async def iter_chunks(
    r: QuestionGenerateReq,
    callback: CallbackDep,
    agent: AgentDep,
    ollama: OllamaDep,
    qdrant: QdrantDep,
    question_search: QuestionSearchDep,
):
    try:
        yield encode_chunk({"done": False, "message": "开始处理。\n"})
        # step 1: search questions
        async with question_search._elasticsearch:  # type: ignore
            qs_same_course, qs_historical = await question_search.find_questions(
                r.exam_kp,
                r.context,
                r.question_type,
                r.major_name,
                r.course_name,
                r.course_code,
                r.university_name,
                10,
                20,
            )
        logger.debug(f"{len(qs_same_course)=}, {len(qs_historical)=}")
        yield encode_chunk({"done": False, "message": "搜索真题完成。\n"})
        # step 2: query chunks
        kp = r.exam_kp.strip().lower()
        text = f"Definition or explanation of {kp}."
        if r.context:
            text += f"\nKnowledge of {kp} related to the following context:\n{r.context.strip()}"
        vec = await ollama.embed_one(text)
        pairs = await qdrant.query_chunks(kp, vec, r.course_id, 8)
        logger.debug("len(chunks)=%d", len(pairs))
        yield encode_chunk({"done": False, "message": "搜索课程材料完成。\n"})
        # step 3: generate key points
        key_points = await agent.analyze_chunks(kp, [it[1] for it in pairs])
        relevances = ["weak", "medium", "strong"]
        key_points.sort(key=lambda it: relevances.index(it.relevance), reverse=True)
        key_points = [it for it in key_points if it.relevance != "weak"]
        yield encode_chunk({"done": False, "message": "理解知识点完成。\n"})
        # step 4: create verify task, this is usually quicker
        if qs_historical:
            task_verify = asyncio.create_task(agent.verify_questions(qs_historical, r.exam_kp, key_points))
        else:
            task_verify = None
        # step 5: generate questions
        qs_generate: list[Question] = []
        chunks: list[str] = []
        async for chunk in agent.generate_stream(
            r.exam_kp, r.context, r.question_type, r.major_name, r.course_name, key_points, 10
        ):
            yield encode_chunk({"done": False, "message": chunk})
            chunks.append(chunk)
        asst_msg = "".join(chunks)
        for it in re.finditer(r"(?s)<question>(.+?)</question>", asst_msg):
            q_content, q_type = agent._parse_question(it.group())  # type: ignore
            qs_generate.append(Question(content=q_content, type=q_type, source=QuestionSource.Generated))
        yield encode_chunk({"done": False, "message": "\n生成题目完成。"})
        logger.debug("generate completed")
        # step 6: await verify task
        if task_verify is not None:
            qs_verified = await task_verify
        else:
            qs_verified = []
        # step 7: concat and filter
        questions = [*qs_same_course, *qs_verified, *qs_generate]
        if r.question_type != QuestionType.Any:
            questions = [it for it in questions if it.type == r.question_type]
        logger.debug(f"{len(questions)=}")
        # step 8: notify callback
        await callback.notify_generate_ok(r.task_id, questions)
    except Exception as exc:
        logger.error("generate error: %r", exc)
        await callback.notify_generate_err(r.task_id, "error when generate")
    finally:
        yield encode_chunk({"done": True, "message": ""})


@router.post("/question/generate-stream")
async def post_question_generate_stream(
    req_body: QuestionGenerateReq,
    callback: CallbackDep,
    agent: AgentDep,
    ollama: OllamaDep,
    qdrant: QdrantDep,
    question_search: QuestionSearchDep,
):
    logger.debug("request body: %r", req_body)
    return StreamingResponse(
        iter_chunks(req_body, callback, agent, ollama, qdrant, question_search), media_type="application/x-ndjson"
    )


async def iter_chunks_test(req: Request):
    try:
        yield encode_chunk({"done": False, "message": req.method + "\n"})
        for k, v in req.headers.items():
            await asyncio.sleep(0.2)
            yield encode_chunk({"done": False, "message": f"{k}: "})
            await asyncio.sleep(0.2)
            yield encode_chunk({"done": False, "message": f"{v}\n"})
    finally:
        yield encode_chunk({"done": True, "message": ""})


# NOTE must add /api prefix
@router.route(
    "/api/stream-test", methods=["CONNECT", "DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "TRACE"]
)
async def route_stream_test(req: Request):
    return StreamingResponse(iter_chunks_test(req), media_type="application/x-ndjson")
