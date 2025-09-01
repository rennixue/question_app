import warnings
from types import TracebackType
from typing import Any
from uuid import UUID, uuid4

import numpy
from elastic_transport import SecurityWarning
from elasticsearch import AsyncElasticsearch
from numpy.typing import NDArray

from ..models import Question, QuestionSource, QuestionType

warnings.filterwarnings("ignore", category=SecurityWarning)


def similarity(arr1: NDArray[Any], arr2: NDArray[Any]) -> float:
    return (numpy.dot(arr1, arr2) / numpy.linalg.norm(arr1) / numpy.linalg.norm(arr2)).item()


class ElasticsearchService:
    def __init__(self, client: AsyncElasticsearch) -> None:
        self._client = client

    async def __aenter__(self) -> "ElasticsearchService":
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.close()

    async def is_healthy(self) -> bool:
        try:
            result = await self._client.ping()
        except Exception:
            return False
        return result

    async def search_questions_same_course(
        self,
        *,
        kp: str,
        kp_vec: list[float],
        q_vec: list[float],
        q_type: str | None,
        course_code: str,
        university: str,
        limit: int,
    ) -> list[Question]:
        filters: list[dict[str, Any]] = [
            {"match_phrase": {"content": {"query": kp, "slop": 0}}},
            {"term": {"course_code": {"value": course_code}}},
            {"term": {"university.raw": {"value": university}}},
        ]
        if q_type:
            filters.append({"term": {"question_type": q_type}})
        resp = await self._knn_search(q_vec, self._make_size(limit), filters)
        sources = self._rerank_questions(kp_vec, resp, limit)
        return [self._make_question(it, QuestionSource.SameCourse) for it in sources]

    async def search_questions_historical(
        self,
        *,
        kp: str,
        kp_vec: list[float],
        q_vec: list[float],
        q_type: str | None,
        majors: list[str] | None,
        course_code: str | None,
        university: str | None,
        limit: int,
    ) -> list[Question]:
        filters: list[dict[str, Any]] = [
            {"match_phrase": {"content": {"query": kp, "slop": 0}}},
        ]
        if q_type:
            filters.append({"term": {"question_type": q_type}})
        if majors:
            filters.append({"terms": {"major.raw": majors}})
        must_nots: list[dict[str, Any]] = []
        if course_code and university:
            must_nots.append(
                {
                    "bool": {
                        "must": [
                            {"term": {"course_code": {"value": course_code}}},
                            {"term": {"university.raw": {"value": university}}},
                        ]
                    }
                }
            )
        resp = await self._knn_search(q_vec, self._make_size(limit), filters, must_nots)
        sources = self._rerank_questions(kp_vec, resp, limit)
        if university:
            qs: list[Question] = []
            for it in sources:
                q_src = QuestionSource.SameCourse if it["university"] == university else QuestionSource.Historical
                qs.append(self._make_question(it, q_src))
            return qs
        else:
            return [self._make_question(it, QuestionSource.Historical) for it in sources]

    def _make_size(self, limit: int) -> int:
        return min(max(10, limit * 2), limit + 10)

    async def _knn_search(
        self,
        q_vec: list[float],
        size: int,
        filters: list[dict[str, Any]],
        must_nots: list[dict[str, Any]] | None = None,
    ) -> Any:
        if must_nots is None:
            must_nots = []
        resp = await self._client.search(
            index="question-retrieval",
            knn={
                "field": "embedding",
                "query_vector": q_vec,
                "k": size,
                "num_candidates": size,
                "filter": {"bool": {"filter": filters, "must_not": must_nots}},
            },
            size=size,
            source_includes=[
                "question_id",
                "content",
                "question_type",
                "university",
                "major",
                "course_name",
                "course_code",
                "key_kps",
            ],
        )
        return resp

    def _rerank_questions(self, kp_vec: list[float], resp: Any, limit: int) -> list[dict[str, Any]]:
        if not resp["hits"]["hits"]:
            return []
        kp_arr = numpy.array(kp_vec)
        pairs: list[tuple[dict[str, Any], float]] = []
        for it in resp["hits"]["hits"]:
            score = it["_score"]
            source = it["_source"]
            if "key_kps" in source:
                if key_kps := source.pop("key_kps"):
                    score = 0.25 * score + 0.75 * max(
                        similarity(kp_arr, it["embedding"]) for it in key_kps if "embedding" in it
                    )
            pairs.append((source, score))
        pairs.sort(key=lambda it: it[1], reverse=True)
        return [it[0] for it in pairs[:limit]]

    def _make_question(self, d: dict[str, Any], src: QuestionSource) -> Question:
        meta_info = "{} - {} - {}".format(
            d.get("university") or "/", d.get("major") or "/", d.get("course_code") or "/"
        )
        q_type = QuestionType.from_elasticsearch_keyword(d.get("question_type"))
        question_id = d.get("question_id")
        if isinstance(question_id, int):
            q_id = UUID(int=question_id)
        else:
            q_id = uuid4()
        return Question(id=q_id, content=d.get("content", ""), type=q_type, source=src, meta_info=meta_info)
