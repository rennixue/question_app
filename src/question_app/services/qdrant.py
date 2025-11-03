import logging
import warnings
from types import TracebackType
from typing import cast

import pylcs
import qdrant_client.models as qm
from qdrant_client import AsyncQdrantClient
from rapidfuzz.distance import Levenshtein

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, message="Api key is used with an insecure connection.")


class QdrantService:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def __aenter__(self) -> "QdrantService":
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.close()

    async def is_healthy(self) -> bool:
        try:
            await self._client.info()
        except Exception:
            return False
        return True

    async def query_closest_major(self, vec: list[float]) -> str | None:
        # SELECT * FROM db_major_cluster WHERE sim_rank=2;
        resp = await self._client.query_points(
            "db_major", vec, using="major", with_payload=["major"], score_threshold=0.7, limit=1
        )
        if not resp.points:
            return None
        point = resp.points[0]
        if not point.payload:
            return None
        return point.payload.get("major")

    async def query_chunks(self, kp: str, vec: list[float], order_id: int, limit: int) -> list[tuple[str, str]]:
        resp = await self._client.query_points(
            "file-preprocess",
            vec,
            query_filter=qm.Filter(must=[qm.FieldCondition(key="order_id", match=qm.MatchValue(value=order_id))]),
            with_payload=["chunk"],
            score_threshold=0.5,
            limit=min(max(4, limit * 2), limit + 4),
        )
        if len(resp.points) == 0:
            logger.info("no chunks retrieved for order_id=%d", order_id)
        pairs: list[tuple[str, str]] = []
        lowered_kp = kp.strip().lower()
        for point in resp.points:
            if payload := point.payload:
                if chunk := payload.get("chunk"):
                    if self._kp_in_chunk(lowered_kp, chunk):
                        pairs.append((str(point.id), chunk))
        return pairs[:limit]

    def _kp_in_chunk(self, lowered_kp: str, chunk: str) -> bool:
        return True
        lowered_chunk = chunk.lower()
        # NOTE not longest common subsequence
        # TODO find a better library
        lcs_len = cast(int, pylcs.lcs_string_length(lowered_kp, chunk))  # type: ignore
        return (
            Levenshtein.distance(lowered_kp, lowered_chunk, weights=(0, 1, 1), score_cutoff=2) <= 2
            and lcs_len >= len(lowered_kp) // 2
        )
