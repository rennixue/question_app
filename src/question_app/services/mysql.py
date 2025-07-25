from collections.abc import Callable
from textwrap import dedent

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
from sqlalchemy.sql.elements import TextClause


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
                    LIMIT 20
                """),
                {"major": major, "threshold": threshold},
            )
            majors = cursor.scalars().all()
        return list(majors)
