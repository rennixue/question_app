import logging
import re

from ..models import Question, QuestionType
from .agent import AgentService
from .elasticsearch import ElasticsearchService
from .mysql import MysqlService
from .ollama import OllamaService
from .qdrant import QdrantService

logger = logging.getLogger(__name__)


class QuestionSearchService:
    def __init__(
        self, elasticsearch: ElasticsearchService, mysql: MysqlService, ollama: OllamaService, qdrant: QdrantService
    ) -> None:
        self._elasticsearch = elasticsearch
        self._mysql = mysql
        self._ollama = ollama
        self._qdrant = qdrant

    async def find_questions(
        self,
        exam_kp: str,
        context: str | None,
        question_type: QuestionType,
        major_name: str | None = None,
        course_name: str | None = None,
        course_code: str | None = None,
        university: str | None = None,
        limit_same_course: int = 10,
        limit_historical: int = 10,
    ) -> tuple[list[Question], list[Question]]:
        kp = exam_kp.strip().lower()
        if course_code:
            course_code = self._normalize_course_code(course_code)
        kp_vec, q_vec = await self._create_embeddings(kp, context, major_name, course_name)
        q_type = question_type.to_elasticsearch_keyword()
        majors = await self.find_majors(major_name) if major_name else None
        if not majors:
            return [], []
        if course_code and university:
            questions_same_course = await self._elasticsearch.search_questions_same_course(
                kp=kp,
                kp_vec=kp_vec,
                q_vec=q_vec,
                q_type=q_type,
                course_code=course_code,
                university=university,
                limit=limit_same_course,
            )
        else:
            questions_same_course = []
        if questions_same_course:
            int_ids = [it.id.int for it in questions_same_course]
            pairs = await self._mysql.select_question_contents(int_ids)
            for int_id, q_content in pairs:
                for it in questions_same_course:
                    if it.id.int == int_id:
                        if q_content:
                            it.content = q_content
                        break
        questions_historical = await self._elasticsearch.search_questions_historical(
            kp=kp,
            kp_vec=kp_vec,
            q_vec=q_vec,
            q_type=q_type,
            majors=majors,
            course_code=course_code,
            university=university,
            limit=limit_historical,
        )
        if questions_historical:
            int_ids = [it.id.int for it in questions_historical]
            pairs = await self._mysql.select_question_contents(int_ids)
            for int_id, q_content in pairs:
                for it in questions_historical:
                    if it.id.int == int_id:
                        if q_content:
                            it.content = q_content
                        break
        return questions_same_course, questions_historical

    async def _create_embeddings(
        self, kp: str, context: str | None, major: str | None, course_name: str | None
    ) -> tuple[list[float], list[float]]:
        parts: list[str] = []
        if major:
            parts.append(f"<major>{major.strip().lower()}</major>")
        if course_name:
            parts.append(f"<course-name>{course_name.strip().lower()}</course-name>")
        if context:
            parts.append(f"This is a question related to the following context:\n{context.strip()}")
        if parts:
            q_text = "\n\n".join(parts)
        else:
            q_text = f"This is a question related to {kp}"
        kp_vec, q_vec = await self._ollama.embed_several(kp, q_text)
        return kp_vec, q_vec

    def _normalize_course_code(self, s: str | None) -> str | None:
        if not s:
            return None
        s = s.upper()
        s = re.sub(r"[^A-Z0-9]", "", s)
        s = re.sub(r"^0+", "", s)
        if not s:
            return None
        return s

    async def find_majors(self, major: str | None) -> list[str]:
        if not major:
            return []
        vec = await self._ollama.embed_one(major)
        closest_major = await self._qdrant.query_closest_major(vec)
        logger.debug("closest major: %r", closest_major)
        if not closest_major:
            return []
        sim_majors = await self._mysql.select_sim_majors(closest_major, 0.8)
        logger.debug("similar majors: %r", sim_majors)
        return sim_majors


class QuestionImitateService:
    # TODO imitative writing
    def __init__(self, agent: AgentService, ollama: OllamaService, qdrant: QdrantService) -> None:
        self._agent = agent
        self._ollama = ollama
        self._qdrant = qdrant

    async def verify(
        self, questions: list[Question], course_id: int, exam_kp: str, context: str | None
    ) -> list[Question]:
        kp = exam_kp.strip().lower()
        vec = await self._create_embedding(kp, context)
        pairs = await self._qdrant.query_chunks(kp, vec, course_id, 12)
        logger.debug("len(chunks)=%d", len(pairs))
        key_points = await self._agent.analyze_chunks(kp, [it[1] for it in pairs])
        relevances = ["weak", "medium", "strong"]
        key_points.sort(key=lambda it: relevances.index(it.relevance), reverse=True)
        qs_verified = await self._agent.verify_questions(questions, exam_kp, key_points)
        return qs_verified

    async def _create_embedding(self, kp: str, context: str | None) -> list[float]:
        text = f"Definition or explanation of {kp}."
        if context:
            text += f"\nKnowledge of {kp} related to the following context:\n{context.strip()}"
        vec = await self._ollama.embed_one(text)
        return vec


class QuestionGenerateService:
    def __init__(self, agent: AgentService, ollama: OllamaService, qdrant: QdrantService) -> None:
        self._agent = agent
        self._ollama = ollama
        self._qdrant = qdrant

    async def generate(
        self,
        course_id: int,
        exam_kp: str,
        context: str | None,
        question_type: QuestionType,
        major: str | None,
        course_name: str | None,
        num: int,
    ) -> list[Question]:
        kp = exam_kp.strip().lower()
        vec = await self._create_embedding(kp, context)
        pairs = await self._qdrant.query_chunks(kp, vec, course_id, 8)
        logger.debug("len(chunks)=%d", len(pairs))
        key_points = await self._agent.analyze_chunks(kp, [it[1] for it in pairs])
        relevances = ["weak", "medium", "strong"]
        key_points.sort(key=lambda it: relevances.index(it.relevance), reverse=True)
        key_points = [it for it in key_points if it.relevance != "weak"]
        questions = await self._agent.generate(exam_kp, context, question_type, major, course_name, key_points, num)
        logger.debug("questions: %r", questions)
        return questions

    async def _create_embedding(self, kp: str, context: str | None) -> list[float]:
        text = f"Definition or explanation of {kp}."
        if context:
            text += f"\nKnowledge of {kp} related to the following context:\n{context.strip()}"
        vec = await self._ollama.embed_one(text)
        return vec


class QuestionRewriteService:
    def __init__(self, agent: AgentService) -> None:
        self._agent = agent

    async def rewrite(
        self, rewrite_prompt: str, old_question: str, exam_kp: str, context: str | None, question_type: QuestionType
    ) -> Question:
        background = f"Write {question_type.to_natural_language()} about {exam_kp}."
        if context:
            background += f" Refer to the following for more information:\n{context}"
        new_question = await self._agent.rewrite(background=background, prompt=rewrite_prompt, question=old_question)
        logger.debug("new question: %r", new_question)
        return new_question
