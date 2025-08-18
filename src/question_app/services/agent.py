import json
import re
from collections.abc import AsyncIterator

from pydantic_ai import Agent

from ..models import KeyPoint, Question, QuestionSource, QuestionType
from .prompt import TemplateManager


class AgentService:
    def __init__(self, chat_agent: Agent, reason_agent: Agent, tmpl_mngr: TemplateManager) -> None:
        self._chat_agent = chat_agent
        self._reason_agent = reason_agent
        self._tmpl_mngr = tmpl_mngr
        self._tmpl_rewrite = tmpl_mngr.load_template("rewrite")
        self._tmpl_analyze_chunks = tmpl_mngr.load_template("analyze_chunks")
        self._tmpl_verify_questions = tmpl_mngr.load_template("verify_questions")
        self._tmpl_generate = tmpl_mngr.load_template("generate")

    async def analyze_chunks(self, query: str, chunks: list[str]) -> list[KeyPoint]:
        user_msg = self._tmpl_analyze_chunks.render(query=query, chunks=chunks)
        asst_msg = ""
        async with self._chat_agent.run_stream(user_msg, model_settings={"max_tokens": 8192}) as result:
            async for asst_msg in result.stream_text():
                pass
        if m := re.search(r"(?s)<summary>(.+?)</summary>", asst_msg):
            explanation = m.group(1).strip()
            offset = m.end()
        else:
            explanation = ""
            offset = 0
        kps: list[KeyPoint] = [KeyPoint(name=query, explanation=explanation, relevance="strong")]
        for it in re.finditer(r"(?s)<entity>(.+?)</entity>", asst_msg[offset:]):
            kp = self._parse_entity(it.group())
            if kp:
                kps.append(kp)
        return kps

    async def verify_questions(
        self, questions: list[Question], exam_kp: str, key_points: list[KeyPoint]
    ) -> list[Question]:
        user_msg = self._tmpl_verify_questions.render(
            kp=exam_kp, key_points=key_points, questions=[it.content[:1000] for it in questions]
        )
        asst_msg = ""
        async with self._chat_agent.run_stream(user_msg, model_settings={"max_tokens": 1024}) as result:
            async for asst_msg in result.stream_text():
                pass
        goods = [True for _ in range(len(questions))]
        if m := re.search(r"(?s)```json\n(.+?)\n```", asst_msg):
            obj = json.loads(m.group(1))
            for it in obj["judgements"]:
                try:
                    q_idx = int(it["question_index"])
                    q_good = bool(it["can_be_solved"])
                    if not q_good:
                        goods[q_idx] = False
                except Exception:
                    pass
        verified_questions = [q for q, flag in zip(questions, goods) if flag]
        return verified_questions

    async def generate(
        self,
        exam_kp: str,
        context: str | None,
        question_type: QuestionType,
        major: str | None,
        course: str | None,
        key_points: list[KeyPoint],
        number: int,
    ) -> list[Question]:
        user_msg = self._tmpl_generate.render(
            exam_kp=exam_kp,
            context=context,
            question_type=question_type.to_natural_language(),
            major=major,
            course=course,
            key_points=key_points,
            number=number,
        )
        asst_msg = ""
        async with self._chat_agent.run_stream(user_msg, model_settings={"max_tokens": 8192}) as result:
            async for asst_msg in result.stream_text():
                pass
        questions: list[Question] = []
        for it in re.finditer(r"(?s)<question>(.+?)</question>", asst_msg):
            content, q_type = self._parse_question(it.group())
            questions.append(Question(content=content, type=q_type, source=QuestionSource.Generated))
        return questions

    async def generate_stream(
        self,
        exam_kp: str,
        context: str | None,
        question_type: QuestionType,
        major: str | None,
        course: str | None,
        key_points: list[KeyPoint],
        number: int,
    ) -> AsyncIterator[str]:
        user_msg = self._tmpl_generate.render(
            exam_kp=exam_kp,
            context=context,
            question_type=question_type.to_natural_language(),
            major=major,
            course=course,
            key_points=key_points,
            number=number,
        )
        async with self._chat_agent.run_stream(user_msg, model_settings={"max_tokens": 8192}) as result:
            async for asst_msg in result.stream_text(delta=True, debounce_by=0.2):
                yield asst_msg

    async def rewrite(self, *, background: str, prompt: str, question: str) -> Question:
        user_msg = self._tmpl_rewrite.render(background=background, prompt=prompt, question=question)
        asst_msg = ""
        async with self._chat_agent.run_stream(user_msg, model_settings={"max_tokens": 4096}) as result:
            async for asst_msg in result.stream_text():
                pass
        content, q_type = self._parse_question(asst_msg)
        return Question(content=content, type=q_type, source=QuestionSource.Rewritten)

    def _parse_entity(self, s: str) -> KeyPoint | None:
        if m := re.search(r"(?s)<name>(.+?)</name>", s):
            name = m.group(1).strip()
            offset = m.end()
        else:
            return None
        if m := re.search(r"(?s)<explanation>(.+?)</explanation>", s[offset:]):
            explanation = m.group(1).strip()
            offset = m.end()
        else:
            explanation = ""
        if m := re.search(r"(?s)<strength>(.+?)</strength>", s[offset:]):
            relevance = m.group(1).strip()
        else:
            relevance = "weak"
        return KeyPoint(name=name, explanation=explanation, relevance=relevance)

    def _parse_question(self, s: str) -> tuple[str, QuestionType]:
        if m := re.search(r"(?s)<content>(.+?)</content>", s):
            content = m.group(1).strip()
            offset = m.end()
        else:
            content = s
            offset = 0
        if m := re.search(r"(?s)<type>(.+?)</type>", s[offset:]):
            q_type = m.group(1).strip()
        else:
            q_type = "open"
        return content, QuestionType.from_value(q_type)
