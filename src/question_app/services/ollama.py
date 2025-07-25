from collections.abc import Iterable
from itertools import batched
from types import TracebackType

from ollama import AsyncClient


class OllamaService:
    def __init__(self, client: AsyncClient, model: str) -> None:
        self._client = client
        self._model = model

    async def __aenter__(self) -> "OllamaService":
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client._client.aclose()  # type: ignore

    async def is_healthy(self) -> bool:
        try:
            await self._client._request_raw("GET", "/", timeout=10)  # type: ignore
        except Exception:
            return False
        return True

    async def embed_one(self, text: str) -> list[float]:
        resp = await self._client.embed(self._model, [text], truncate=True)
        return list(resp.embeddings[0])

    async def embed_several(self, *texts: str) -> tuple[list[float], ...]:
        resp = await self._client.embed(self._model, texts, truncate=True)
        return tuple(map(list, resp.embeddings))

    async def embed_many(self, texts: Iterable[str], batch_size: int) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in batched(texts, batch_size):
            resp = await self._client.embed(self._model, batch, truncate=True)
            embeddings.extend(map(list, resp.embeddings))
        return embeddings
