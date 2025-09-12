import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from textwrap import dedent

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import bindparam, text
from sqlalchemy.sql.elements import TextClause

from ..models import CourseMaterialType, ExtractedFile, ExtractedFileWithType, KeyPointNameAndFreq


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
