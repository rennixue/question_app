from collections.abc import Callable, Iterable
from textwrap import dedent

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import bindparam, text
from sqlalchemy.sql.elements import TextClause

from ..models import ExtractedFile


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

    async def select_order_kps(self, order_id: int, file_limit: int, kp_limit: int) -> list[ExtractedFile]:
        async with self._session_factory() as session:
            cursor = await session.execute(
                sql("""
                    SELECT file_name, kp_list FROM db_order_file_kp
                    WHERE order_id = :order_id AND kp_list IS NOT NULL
                    ORDER BY file_name
                    LIMIT :file_limit
                """),
                {"order_id": order_id, "file_limit": file_limit},
            )
            rows = cursor.all()
        files: list[ExtractedFile] = []
        for file_name, kps_str in rows:
            if file_name and kps_str:
                if kps := kps_str.split(","):
                    files.append(ExtractedFile(file_name=file_name, kps=kps[:kp_limit]))
        return files
