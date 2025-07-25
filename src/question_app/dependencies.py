from asyncio import current_task
from pathlib import Path
from typing import Annotated

from elasticsearch import AsyncElasticsearch
from fastapi import Depends
from ollama import AsyncClient as AsyncOllamaClient
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import async_scoped_session, async_sessionmaker, create_async_engine

from .models import Settings
from .services import (
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
from .services.prompt import JinjaTemplateManager

settings: Settings | None = None


async def get_or_create_settings() -> Settings:
    global settings
    if settings is None:
        settings = Settings()  # type: ignore
    return settings


SettingsDep = Annotated[Settings, Depends(get_or_create_settings)]


agent_service: AgentService | None = None


async def get_or_create_agent(settings: SettingsDep) -> AgentService:
    global agent_service
    if agent_service is None:
        provider = OpenAIProvider(base_url=settings.openai.base_url, api_key=settings.openai.api_key)
        chat_model = OpenAIModel(settings.openai.chat_model, provider=provider)
        reason_model = OpenAIModel(settings.openai.reason_model, provider=provider)
        chat_agent = Agent(chat_model)
        reason_agent = Agent(reason_model)
        tmpl_mngr = JinjaTemplateManager(
            Path(__file__).parent / "services/prompts", trim_blocks=True, lstrip_blocks=True
        )
        agent_service = AgentService(chat_agent, reason_agent, tmpl_mngr)
    return agent_service


AgentDep = Annotated[AgentService, Depends(get_or_create_agent)]


async def create_elasticsearch(settings: SettingsDep) -> ElasticsearchService:
    # will be sent to async task, do not close in dependency definition
    client = AsyncElasticsearch(settings.elasticsearch.url, api_key=settings.elasticsearch.api_key, verify_certs=False)
    return ElasticsearchService(client)


ElasticsearchDep = Annotated[ElasticsearchService, Depends(create_elasticsearch)]


callback_service: CallbackService | None = None


async def get_or_create_callback(settings: SettingsDep) -> CallbackService:
    global callback_service
    if callback_service is None:
        callback_service = CallbackService(str(settings.callback_base_url), settings.skip_callback)
    return callback_service


CallbackDep = Annotated[CallbackService, Depends(get_or_create_callback)]


mysql_service: MysqlService | None = None


async def get_or_create_mysql(settings: SettingsDep) -> MysqlService:
    global mysql_service
    if mysql_service is None:
        engine = create_async_engine(settings.sqlalchemy.url, pool_size=5, pool_recycle=60)
        session_maker = async_sessionmaker(engine)
        session_factory = async_scoped_session(session_maker, current_task)
        mysql_service = MysqlService(session_factory)
    return mysql_service


MysqlDep = Annotated[MysqlService, Depends(get_or_create_mysql)]


async def create_ollama(settings: SettingsDep) -> OllamaService:
    # will be sent to async task, do not close in dependency definition
    client = AsyncOllamaClient(settings.ollama.url, timeout=20)
    return OllamaService(client, "bge-m3")


OllamaDep = Annotated[OllamaService, Depends(create_ollama)]


async def create_qdrant(settings: SettingsDep) -> QdrantService:
    # will be sent to async task, do not close in dependency definition
    client = AsyncQdrantClient(settings.qdrant.url, api_key=settings.qdrant.api_key)
    return QdrantService(client)


QdrantDep = Annotated[QdrantService, Depends(create_qdrant)]


async def create_question_generate(agent: AgentDep, ollama: OllamaDep, qdrant: QdrantDep) -> QuestionGenerateService:
    return QuestionGenerateService(agent, ollama, qdrant)


QuestionGenerateDep = Annotated[QuestionGenerateService, Depends(create_question_generate)]


async def create_question_imitate(agent: AgentDep, ollama: OllamaDep, qdrant: QdrantDep) -> QuestionImitateService:
    return QuestionImitateService(agent, ollama, qdrant)


QuestionImitateDep = Annotated[QuestionImitateService, Depends(create_question_imitate)]


async def create_question_rewrite(agent: AgentDep) -> QuestionRewriteService:
    return QuestionRewriteService(agent)


QuestionRewriteDep = Annotated[QuestionRewriteService, Depends(create_question_rewrite)]


async def create_question_search(
    elasticsearch: ElasticsearchDep, mysql: MysqlDep, ollama: OllamaDep, qdrant: QdrantDep
) -> QuestionSearchService:
    return QuestionSearchService(elasticsearch, mysql, ollama, qdrant)


QuestionSearchDep = Annotated[QuestionSearchService, Depends(create_question_search)]
