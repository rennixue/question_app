import asyncio
import json
import logging
import random
import re
import time
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
    SettingsDep,
)
from .models import (
    AnalyzeDescriptionOutput,
    KeyPoint,
    Question,
    QuestionGenerateReq,
    QuestionRewriteReq,
    QuestionSection,
    QuestionSource,
    QuestionType,
    StreamBlock,
)

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
        aiterator = question_search.find_questions(
            r.exam_kp,
            r.context,
            r.question_type,
            r.major_name,
            r.course_name,
            r.course_code,
            r.university_name,
            10,
            10,
            20,
        )
        qs_same_course = await anext(aiterator)
        qs_same_university = await anext(aiterator)
        qs_historical = await anext(aiterator)
        qs_historical = [*qs_same_university, *qs_historical]
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


async def empty_generate(wait_secs: float) -> list[Question]:
    await asyncio.sleep(wait_secs)
    logger.debug("empty generate")
    return []


async def wrapped_generate(
    max_exec_secs: float,
    req_body: QuestionGenerateReq,
    callback: CallbackDep,
    question_generate: QuestionGenerateDep,
    question_imitate: QuestionImitateDep,
    question_search: QuestionSearchDep,
):
    # task = asyncio.create_task(do_generate(req_body, question_generate, question_imitate, question_search))
    task = asyncio.create_task(empty_generate(170.0))
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
    file_limit = file_limit or 100
    kp_limit = kp_limit or 20
    processed, files, files_with_types, order_kps = await mysql.select_order_kps(course_id, file_limit, kp_limit)
    new_files_with_types = [
        {
            "file_type": file_type,
            "files": [
                {"file_name": it.file_name, "kps": it.kps} for it in files_with_types if it.file_type == file_type
            ],
        }
        for file_type in sorted(set(it.file_type for it in files_with_types), key=lambda it: it.to_int())
    ]
    return {
        "code": 0,
        "status": 200,
        "body": {
            "course_id": course_id,
            "processed": processed,
            "files": files,
            "files_with_types": new_files_with_types,
            "order_kps": order_kps,
        },
    }


def encode_chunk(obj: Any) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n\n"


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
        # step 0: analyze description
        if r.context:
            analyzed_context = await agent.analyze_description(exam_kp=r.exam_kp, context=r.context)
            logger.debug("analyzed_context: %r", analyzed_context)
            yield encode_chunk({"done": False, "message": "理解描述完成。\n"})
        else:
            analyzed_context = AnalyzeDescriptionOutput()
        # step 1: search questions
        async with question_search._elasticsearch:  # type: ignore
            aiterator = question_search.find_questions(
                r.exam_kp,
                r.context,
                r.question_type,
                r.major_name,
                r.course_name,
                r.course_code,
                r.university_name,
                10,
                10,
                20,
            )
            qs_same_course = await anext(aiterator)
            qs_same_university = await anext(aiterator)
            qs_historical = await anext(aiterator)
            qs_historical = [*qs_same_university, *qs_historical]
        logger.debug(f"{len(qs_same_course)=}, {len(qs_historical)=}")
        yield encode_chunk({"done": False, "message": "搜索真题完成。\n"})
        # step 2: query chunks
        kp = r.exam_kp.strip().lower()
        text = f"Definition or explanation of {kp}."
        if it := (analyzed_context.key_concepts or r.context):
            text += f"\nKnowledge of {kp} related to the following context:\n{it.strip()}"
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
            r.exam_kp, r.context, analyzed_context, r.question_type, r.major_name, r.course_name, key_points, 10
        ):
            yield encode_chunk({"done": False, "message": chunk})
            chunks.append(chunk)
        asst_msg = "".join(chunks)
        for it in re.finditer(r"(?s)<question>(.+?)</question>", asst_msg):
            q_content, q_type = agent._parse_question(it.group())  # type: ignore
            # if q_type == QuestionType.MultipleChoice:
            #     q_content = reorder_choices(q_content)
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
        iter_chunks(req_body, callback, agent, ollama, qdrant, question_search), media_type="text/event-stream"
    )


async def iter_chunks_test(req: Request):
    try:
        yield encode_chunk({"done": False, "message": req.method + "\n"})
        for k, v in req.headers.items():
            await asyncio.sleep(0.2)
            yield encode_chunk({"done": False, "message": f"{k}: "})
            await asyncio.sleep(0.2)
            yield encode_chunk({"done": False, "message": f"{v}\n"})
    except asyncio.CancelledError as exc:
        logger.error(repr(exc))
    finally:
        yield encode_chunk({"done": True, "message": ""})


# NOTE must add /api prefix
@router.route(
    "/api/stream-test", methods=["CONNECT", "DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "TRACE"]
)
async def route_stream_test(req: Request):
    return StreamingResponse(iter_chunks_test(req), media_type="text/event-stream")


def encode_block(block: StreamBlock) -> str:
    return "data: " + block.model_dump_json() + "\n\n"


async def extract_key_points(
    r: QuestionGenerateReq,
    agent: AgentDep,
    ollama: OllamaDep,
    qdrant: QdrantDep,
) -> tuple[AnalyzeDescriptionOutput, list[KeyPoint], list[tuple[str, str]]]:
    # step 1: analyze context
    if r.context:
        analyzed_context = await agent.analyze_description(exam_kp=r.exam_kp, context=r.context)
        logger.debug("analyzed_context: %r", analyzed_context)
    else:
        analyzed_context = AnalyzeDescriptionOutput()
    # step 2: retrieve chunks
    kp = r.exam_kp.strip().lower()
    text = f"Definition or explanation of {kp}."
    if it := (analyzed_context.key_concepts or r.context):
        text += f"\nKnowledge of {kp} related to the following context:\n{it.strip()}"
    vec = await ollama.embed_one(text)
    pairs = await qdrant.query_chunks(kp, vec, r.course_id, 8)
    logger.debug("len(chunks)=%d", len(pairs))
    # step 3: generate key points
    key_points = await agent.analyze_chunks(kp, [it[1] for it in pairs])
    relevances = ["weak", "medium", "strong"]
    key_points.sort(key=lambda it: relevances.index(it.relevance), reverse=True)
    key_points = [it for it in key_points if it.relevance != "weak"]
    return analyzed_context, key_points, pairs


def sort_by_year(qs: list[Question]) -> list[Question]:
    pairs: list[tuple[int, Question]] = []
    for it in qs:
        year = 0
        if it.meta_info:
            if m := re.search(r"^20\d\d", it.meta_info):
                year = int(m.group(0))
        pairs.append((year, it))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return [it[1] for it in pairs]


def cleanse_question_content(qs: list[Question]) -> list[Question]:
    for it in qs:
        it.content = re.sub(r"(?m)^#{1,4} (.+)$", "**\\1**", it.content)
    return qs


async def iter_blocks(
    request: Request,
    r: QuestionGenerateReq,
    callback: CallbackDep,
    agent: AgentDep,
    mysql: MysqlDep,
    ollama: OllamaDep,
    qdrant: QdrantDep,
    question_search: QuestionSearchDep,
    settings: SettingsDep,
):
    time_start = time.perf_counter()
    questions: list[Question] = []
    count_same_course = 0
    count_same_univ = 0
    count_historical = 0
    count_gen = 0
    count_gen_1 = 0
    count_gen_2 = 0
    elapsed_same_course = 0.0
    elapsed_same_univ = 0.0
    elapsed_historical = 0.0
    elapsed_gen = 0.0
    elapsed_gen_1 = 0.0
    elapsed_gen_2 = 0.0
    try:
        # timer for step 1.1
        time_same_course_start = time.perf_counter()
        await mysql.log_search(r.task_id, settings.is_dev, r.exam_kp, r.context, r.question_type)  # pyright: ignore[reportPrivateUsage]
        try:
            analyze_query_output = await agent.analyze_query(r.exam_kp)
            await mysql.log_search_ext1(r.task_id, analyze_query_output)
            kp_synonyms = analyze_query_output.synonyms
        except Exception as exc:
            logger.warning("fail to extract terms: %r", exc)
            kp_synonyms = []
        else:
            r.exam_kp = analyze_query_output.primary_term
            r.context = (", ".join(analyze_query_output.secondary_terms) + "\n\n" + (r.context or "")).strip()
            logger.info("actual searched exam_kp=%r, synonyms=%r, context=%r", r.exam_kp, kp_synonyms, r.context)
        knowledge_task = asyncio.create_task(extract_key_points(r, agent, ollama, qdrant))
        # step 1: search questions
        async with question_search._elasticsearch:  # type: ignore
            aiterator = question_search.find_questions(
                r.exam_kp,
                r.context,
                r.question_type,
                r.major_name,
                r.course_name,
                r.course_code,
                r.university_name,
                20,
                20,
                20,
                kp_synonyms,
            )
            # step 1.1: search same course questions
            # time_same_course_start = time.perf_counter()
            yield encode_block(StreamBlock(q_src=QuestionSource.SameCourse, status="start"))
            try:
                qs_same_course = await anext(aiterator)
            except Exception as exc:
                logger.error("fail to search same course: %r", exc)
            else:
                if qs_same_course:
                    qs_same_course = sort_by_year(qs_same_course)  # NOTE bad
                    qs_same_course = cleanse_question_content(qs_same_course)
                    yield encode_block(
                        StreamBlock(q_src=QuestionSource.SameCourse, status="progress", questions=qs_same_course)
                    )
                    count_same_course = len(qs_same_course)
                    questions.extend(qs_same_course)
                logger.debug(f"{len(qs_same_course)=}")
            elapsed_same_course = time.perf_counter() - time_same_course_start
            yield encode_block(
                StreamBlock(
                    q_src=QuestionSource.SameCourse, status="finish", count=count_same_course, time=elapsed_same_course
                )
            )
            # step 1.2: search same university questions
            time_same_univ_start = time.perf_counter()
            yield encode_block(StreamBlock(q_src=QuestionSource.SameUniversity, status="start"))
            analyzed_context, key_points, chunks = None, None, []
            try:
                qs_same_univ = await anext(aiterator)
            except Exception as exc:
                logger.error("fail to search same university: %r", exc)
            else:
                if qs_same_univ:
                    try:
                        analyzed_context, key_points, chunks = await knowledge_task
                    except Exception as exc:
                        logger.error("fail to fetch knowledge: %r", exc)
                        analyzed_context, key_points = AnalyzeDescriptionOutput(), []
                    qs_same_univ = sort_by_year(qs_same_univ)  # NOTE bad
                    qs_same_univ = cleanse_question_content(qs_same_univ)
                    qs_same_univ_verified = await agent.verify_questions(qs_same_univ, r.exam_kp, key_points)
                    verified_ids_same_univ = [it.id for it in qs_same_univ_verified]
                    await mysql.log_verify(
                        r.task_id,
                        settings.is_dev,
                        QuestionSource.SameUniversity,
                        [(q.id, q.id in verified_ids_same_univ, q.type, q.content) for q in qs_same_univ],
                    )
                    if qs_same_univ_verified:
                        yield encode_block(
                            StreamBlock(
                                q_src=QuestionSource.SameUniversity, status="progress", questions=qs_same_univ_verified
                            )
                        )
                        count_same_univ = len(qs_same_univ_verified)
                        questions.extend(qs_same_univ_verified)
                logger.debug(f"{len(qs_same_univ)=}, {count_same_univ=}")
            elapsed_same_univ = time.perf_counter() - time_same_univ_start
            yield encode_block(
                StreamBlock(
                    q_src=QuestionSource.SameUniversity, status="finish", count=count_same_univ, time=elapsed_same_univ
                )
            )
            # step 1.3: search other university questions
            time_historical_start = time.perf_counter()
            yield encode_block(StreamBlock(q_src=QuestionSource.Historical, status="start"))
            try:
                qs_historical = await anext(aiterator)
            except Exception as exc:
                logger.error("fail to search same course: %r", exc)
            else:
                if qs_historical:
                    if (analyzed_context, key_points) == (None, None):
                        try:
                            analyzed_context, key_points, chunks = await knowledge_task
                        except Exception as exc:
                            logger.error("fail to fetch knowledge: %r", exc)
                            analyzed_context, key_points, chunks = AnalyzeDescriptionOutput(), [], []
                    else:
                        assert analyzed_context is not None
                        assert key_points is not None
                    qs_historical = cleanse_question_content(qs_historical)
                    qs_historical_verified = await agent.verify_questions(qs_historical, r.exam_kp, key_points)
                    verified_ids_historical = [it.id for it in qs_historical_verified]
                    await mysql.log_verify(
                        r.task_id,
                        settings.is_dev,
                        QuestionSource.Historical,
                        [(q.id, q.id in verified_ids_historical, q.type, q.content) for q in qs_historical],
                    )
                    if qs_historical_verified:
                        yield encode_block(
                            StreamBlock(
                                q_src=QuestionSource.Historical, status="progress", questions=qs_historical_verified
                            )
                        )
                        count_historical = len(qs_historical_verified)
                        questions.extend(qs_historical_verified)
                logger.debug(f"{len(qs_historical)=}, {count_historical=}")
            elapsed_historical = time.perf_counter() - time_historical_start
            yield encode_block(
                StreamBlock(
                    q_src=QuestionSource.Historical, status="finish", count=count_historical, time=elapsed_historical
                )
            )

        # step 2: generate questions
        time_gen_start = time.perf_counter()
        count_gen = 0
        qs_gen: list[Question] = []
        # step 2.1: generate questions first batch
        yield encode_block(StreamBlock(q_src=QuestionSource.Generated, status="start"))
        if (analyzed_context, key_points) == (None, None):
            try:
                analyzed_context, key_points, chunks = await knowledge_task
            except Exception as exc:
                logger.error("fail to fetch knowledge: %r", exc)
                analyzed_context, key_points, chunks = AnalyzeDescriptionOutput(), [], []
        else:
            assert analyzed_context is not None
            assert key_points is not None
            # chunks can be empty
        await mysql.log_search_ext2(r.task_id, analyzed_context, key_points, chunks)
        async for some_questions in agent.generate_stream_first(
            r.exam_kp,
            r.context,
            analyzed_context,
            r.question_type,
            r.major_name,
            r.course_name,
            key_points,
            random.randint(10, 20),
        ):
            if some_questions:
                yield encode_block(
                    StreamBlock(q_src=QuestionSource.Generated, status="progress", questions=some_questions)
                )
                count_gen += len(some_questions)
                questions.extend(some_questions)
                qs_gen.extend(some_questions)
        count_gen_1 = len(qs_gen)
        elapsed_gen_1 = time.perf_counter() - time_gen_start
        yield encode_block(
            StreamBlock(q_src=QuestionSource.Generated, status="checkpoint", count=count_gen_1, time=elapsed_gen_1)
        )
        # step 2.2: generate questions second batch
        async for some_questions in agent.generate_stream_second(
            r.exam_kp,
            r.context,
            analyzed_context,
            r.question_type,
            r.major_name,
            r.course_name,
            key_points,
            random.randint(10, 20),
            qs_gen.copy(),
        ):
            if some_questions:
                for it in some_questions:
                    it.batch_no = 2
                yield encode_block(
                    StreamBlock(q_src=QuestionSource.Generated, status="progress", questions=some_questions)
                )
                count_gen += len(some_questions)
                questions.extend(some_questions)
                qs_gen.extend(some_questions)
        elapsed_gen = time.perf_counter() - time_gen_start
        count_gen_2 = len(qs_gen) - count_gen_1
        elapsed_gen_2 = elapsed_gen - elapsed_gen_1
        yield encode_block(
            StreamBlock(q_src=QuestionSource.Generated, status="finish", count=count_gen, time=elapsed_gen)
        )
        logger.debug(f"{len(qs_gen)=}")

        # step 3: notify callback
        # fmt: off
        sections = [
            QuestionSection(q_src=QuestionSource.SameCourse, batch_no=1, count=count_same_course, elapsed=elapsed_same_course),
            QuestionSection(q_src=QuestionSource.SameUniversity, batch_no=1, count=count_same_univ, elapsed=elapsed_same_univ),
            QuestionSection(q_src=QuestionSource.Historical, batch_no=1, count=count_historical, elapsed=elapsed_historical),
            QuestionSection(q_src=QuestionSource.Generated, batch_no=1, count=count_gen_1, elapsed=elapsed_gen_1),
            QuestionSection(q_src=QuestionSource.Generated, batch_no=2, count=count_gen_2, elapsed=elapsed_gen_2),
        ]
        # fmt: on
        await callback.notify_generate_ok(r.task_id, questions, sections)
    except Exception as exc:
        logger.error("generate error: %r", exc)
        # fmt: off
        sections = [
            QuestionSection(q_src=QuestionSource.SameCourse, batch_no=1, count=count_same_course, elapsed=elapsed_same_course),
            QuestionSection(q_src=QuestionSource.SameUniversity, batch_no=1, count=count_same_univ, elapsed=elapsed_same_univ),
            QuestionSection(q_src=QuestionSource.Historical, batch_no=1, count=count_historical, elapsed=elapsed_historical),
            QuestionSection(q_src=QuestionSource.Generated, batch_no=1, count=count_gen_1, elapsed=elapsed_gen_1),
            QuestionSection(q_src=QuestionSource.Generated, batch_no=2, count=count_gen_2, elapsed=elapsed_gen_2),
        ]
        # fmt: on
        await callback.notify_generate_err(r.task_id, "error when generate", questions, sections)
    except asyncio.CancelledError:  # subclass of BaseException
        logger.warning("request cancelled")
        # fmt: off
        sections = [
            QuestionSection(q_src=QuestionSource.SameCourse, batch_no=1, count=count_same_course, elapsed=elapsed_same_course),
            QuestionSection(q_src=QuestionSource.SameUniversity, batch_no=1, count=count_same_univ, elapsed=elapsed_same_univ),
            QuestionSection(q_src=QuestionSource.Historical, batch_no=1, count=count_historical, elapsed=elapsed_historical),
            QuestionSection(q_src=QuestionSource.Generated, batch_no=1, count=count_gen_1, elapsed=elapsed_gen_1),
            QuestionSection(q_src=QuestionSource.Generated, batch_no=2, count=count_gen_2, elapsed=elapsed_gen_2),
        ]
        # fmt: on
        await callback.notify_generate_ok(r.task_id, questions, sections, "request cancelled")
    finally:
        time_total = time.perf_counter() - time_start
        logger.debug(f"total count: {len(questions)}, total time: {time_total}")
        yield encode_block(StreamBlock(done=True, count=len(questions), time=time_total))


@router.post("/question/generate-blocks")
async def post_question_generate_blocks(
    req: Request,
    req_body: QuestionGenerateReq,
    callback: CallbackDep,
    agent: AgentDep,
    mysql: MysqlDep,
    ollama: OllamaDep,
    qdrant: QdrantDep,
    question_search: QuestionSearchDep,
    settings: SettingsDep,
):
    logger.debug("request body: %r", req_body)
    return StreamingResponse(
        iter_blocks(req, req_body, callback, agent, mysql, ollama, qdrant, question_search, settings),
        media_type="text/event-stream",
    )
