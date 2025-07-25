from collections.abc import AsyncGenerator

import pytest

from question_app.dependencies import (
    Settings,
    create_elasticsearch,
    create_ollama,
    create_qdrant,
    create_question_search,
    get_or_create_agent,
    get_or_create_callback,
    get_or_create_mysql,
    get_or_create_settings,
)
from question_app.services import (
    AgentService,
    CallbackService,
    ElasticsearchService,
    MysqlService,
    OllamaService,
    QdrantService,
    QuestionGenerateService,
    QuestionImitateService,
    QuestionRewriteService,
    QuestionSearchService,
)


@pytest.fixture
async def settings() -> Settings:
    return await get_or_create_settings()


@pytest.fixture
async def callback(settings: Settings) -> CallbackService:
    return await get_or_create_callback(settings)


@pytest.fixture
async def agent(settings: Settings) -> AgentService:
    return await get_or_create_agent(settings)


@pytest.fixture
async def elasticsearch(settings: Settings) -> AsyncGenerator[ElasticsearchService]:
    async with await create_elasticsearch(settings) as service:
        yield service


@pytest.fixture
async def mysql(settings: Settings) -> MysqlService:
    return await get_or_create_mysql(settings)


@pytest.fixture
async def ollama(settings: Settings) -> AsyncGenerator[OllamaService]:
    async with await create_ollama(settings) as service:
        yield service


@pytest.fixture
async def qdrant(settings: Settings) -> AsyncGenerator[QdrantService]:
    async with await create_qdrant(settings) as service:
        yield service


@pytest.fixture
async def question_search(
    elasticsearch: ElasticsearchService, mysql: MysqlService, ollama: OllamaService, qdrant: QdrantService
) -> AsyncGenerator[QuestionSearchService]:
    async with elasticsearch, ollama, qdrant:
        yield await create_question_search(elasticsearch, mysql, ollama, qdrant)


@pytest.fixture
async def question_imitate(agent: AgentService, ollama: OllamaService, qdrant: QdrantService) -> QuestionImitateService:
    return QuestionImitateService(agent, ollama, qdrant)


@pytest.fixture
async def question_generate(
    agent: AgentService, ollama: OllamaService, qdrant: QdrantService
) -> QuestionGenerateService:
    return QuestionGenerateService(agent, ollama, qdrant)


@pytest.fixture
async def question_rewrite(agent: AgentService) -> QuestionRewriteService:
    return QuestionRewriteService(agent)
