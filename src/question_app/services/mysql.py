import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from textwrap import dedent
from uuid import UUID

import pydantic_core
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import bindparam, text
from sqlalchemy.sql.elements import TextClause

from ..models import (
    AnalyzeDescriptionOutput,
    AnalyzeQueryOutput,
    CourseMaterialType,
    ExtractedFile,
    ExtractedFileWithType,
    KeyPoint,
    KeyPointNameAndFreq,
    QuestionSource,
    QuestionType,
)

logger = logging.getLogger(__name__)


def sql(s: str) -> TextClause:
    return text(dedent(s))


class MysqlService:
    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def is_healthy(self) -> bool:
        try:
            async with self._session_factory() as session:
                cursor = await session.execute(sql("SELECT 1"))
                assert cursor.scalar_one() == 1
        except Exception:
            return False
        return True

    async def select_sim_majors(self, major: str, threshold: float) -> list[str]:
        async with self._session_factory() as session:
            cursor = await session.execute(
                sql("""
                    SELECT major_sim FROM db_major_cluster
                    WHERE major = :major AND similarity >= :threshold
                    ORDER BY similarity DESC
                    LIMIT 40
                """),
                {"major": major, "threshold": threshold},
            )
            majors = cursor.scalars().all()
        return list(majors)

    async def select_question_contents(self, question_ids: Iterable[int]) -> list[tuple[int, str]]:
        async with self._session_factory() as session:
            cursor = await session.execute(
                sql("""
                    SELECT question_id, content FROM db_exam_question_content
                    WHERE question_id IN :question_ids
                """).bindparams(bindparam("question_ids", expanding=True)),
                {"question_ids": question_ids},
            )
            rows = cursor.all()
        return [(it[0], it[1]) for it in rows]

    async def select_question_years(self, question_ids: Iterable[int]) -> list[tuple[int, int]]:
        async with self._session_factory() as session:
            cursor = await session.execute(
                sql("""
                    SELECT question_id, exam_year FROM db_exam_question
                    WHERE question_id IN :question_ids
                """).bindparams(bindparam("question_ids", expanding=True)),
                {"question_ids": question_ids},
            )
            rows = cursor.all()
        goods: list[tuple[int, int]] = []
        for row in rows:
            if row[1]:
                if m := re.search(r"20\d\d", row[1]):
                    year = int(m.group(0))
                    goods.append((row[0], year))
        return goods

    async def select_order_kps(
        self, order_id: int, file_limit: int, kp_limit: int
    ) -> tuple[bool, list[ExtractedFile], list[ExtractedFileWithType], list[KeyPointNameAndFreq]]:
        async with self._session_factory() as session:
            cursor = await session.execute(
                sql("""
                    SELECT file_name, raw_file_type, file_new_type, kp_list FROM db_order_file_kp
                    WHERE order_id = :order_id
                    ORDER BY file_name
                    LIMIT :file_limit
                """),
                {"order_id": order_id, "file_limit": file_limit},
            )
            rows = cursor.all()
            if len(rows) == 0:
                return False, [], [], []
        files: list[ExtractedFile] = []
        files_with_types: list[ExtractedFileWithType] = []
        all_kps: defaultdict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))
        for file_name, raw_file_type, file_new_type, kps_str in rows:
            if not file_name:
                continue
            if file_new_type is None:
                file_type = CourseMaterialType.LectureNote
            else:
                file_type = CourseMaterialType.from_string(file_new_type)
                if file_type == CourseMaterialType.LectureNote and raw_file_type not in (5, 26):
                    file_type = CourseMaterialType.Other
            kps_str: str | None
            if kps_str is None:
                kps: list[str] = []
            else:
                if kps_str.startswith("["):
                    try:
                        kp_pairs = json.loads(kps_str)
                    except Exception:
                        kps = []
                    else:
                        kps = [it[0] for it in kp_pairs]
                        for name, score in kp_pairs:
                            old_freq, old_score = all_kps[name]
                            all_kps[name] = (old_freq + 1, old_score + score)
                else:
                    if kps_str == "<NOTHING_EXTRACTED>":
                        kps = []
                    else:
                        kps = kps_str.split(",")
                        for idx, name in enumerate(kps):
                            old_freq, old_score = all_kps[name]
                            all_kps[name] = (old_freq + 1, old_score + max(1.0 - idx / 20.0, 0.05))
            if kps:
                kps = kps[:kp_limit]
                files.append(ExtractedFile(file_name=file_name, kps=kps))
            files_with_types.append(ExtractedFileWithType(file_name=file_name, file_type=file_type, kps=kps))
        order_kp_triples = [(name, freq, score) for name, (freq, score) in all_kps.items()]
        order_kp_triples.sort(key=lambda it: it[2], reverse=True)
        order_kps = [KeyPointNameAndFreq(name=name, freq=freq) for name, freq, _ in order_kp_triples[:kp_limit]]
        return True, files, files_with_types, order_kps

    async def log_search(
        self, task_id: int, is_dev: bool, exam_kp: str, context: str | None, search_q_type: QuestionType
    ) -> None:
        try:
            async with self._session_factory() as session:
                await session.execute(
                    sql("""
                        INSERT INTO tiku_log_search (task_id, is_dev, exam_kp, context, search_q_type)
                        VALUES (:task_id, :is_dev, :exam_kp, :context, :search_q_type)
                    """),
                    {
                        "task_id": task_id,
                        "is_dev": is_dev,
                        "exam_kp": exam_kp,
                        "context": context,
                        "search_q_type": search_q_type.to_int(),
                    },
                )
                await session.commit()
        except Exception as exc:
            logger.error("fail to mysql log_search %d: %r", task_id, exc)

    async def log_search_ext1(self, task_id: int, analyze_query_output: AnalyzeQueryOutput) -> None:
        async with self._session_factory() as session:
            await session.execute(
                sql("""
                    UPDATE tiku_log_search SET kp_output = :kp_output WHERE task_id = :task_id
                """),
                {
                    "task_id": task_id,
                    "kp_output": analyze_query_output.model_dump_json() if analyze_query_output else None,
                },
            )
            await session.commit()

    async def log_search_ext2(
        self,
        task_id: int,
        analyze_description_output: AnalyzeDescriptionOutput,
        key_points: list[KeyPoint],
        chunks: list[tuple[str, str]],
    ) -> None:
        async with self._session_factory() as session:
            await session.execute(
                sql("""
                    UPDATE tiku_log_search SET ctx_output = :ctx_output, kps = :kps, chunks = :chunks
                    WHERE task_id = :task_id
                """),
                {
                    "task_id": task_id,
                    "ctx_output": analyze_description_output.model_dump_json() if analyze_description_output else None,
                    "kps": pydantic_core.to_json(key_points),
                    "chunks": pydantic_core.to_json(chunks),
                },
            )
            await session.commit()

    async def log_verify(
        self, task_id: int, is_dev: bool, q_src: QuestionSource, qs: list[tuple[UUID, bool, QuestionType, str]]
    ) -> None:
        if not qs:
            return
        try:
            q_src_int = q_src.to_int()
            async with self._session_factory() as session:
                await session.execute(
                    sql("""
                        INSERT INTO tiku_log_verify (task_id, is_dev, q_src, q_id, is_remaining, q_type, q_content)
                        VALUES (:task_id, :is_dev, :q_src, :q_id, :is_remaining, :q_type, :q_content)
                    """),
                    [
                        {
                            "task_id": task_id,
                            "is_dev": is_dev,
                            "q_src": q_src_int,
                            "q_id": q_id.int,
                            "is_remaining": is_remaining,
                            "q_type": q_type.to_int(),
                            "q_content": q_content,
                        }
                        for q_id, is_remaining, q_type, q_content in qs
                    ],
                )
                await session.commit()
        except Exception as exc:
            logger.error("fail to mysql log_verify %d: %r", task_id, exc)
