import asyncio
import logging
from asyncio import Task, TaskGroup

from fastapi import APIRouter

from .dependencies import (
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
from .models import Question, QuestionGenerateReq, QuestionRewriteReq, QuestionType

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
